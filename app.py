"""
Crypto Analysis Dashboard — Flask Server
"""

from flask import Flask, jsonify, render_template, request
from analysis import full_analysis, chart_for_timeframe
from datetime import datetime, timezone
import time
import threading
import uuid
import os
import learning
import notifications
import ai_analysis
import market_data
import backtester
import regime

app          = Flask(__name__)
_cache       = {}
_lock        = threading.Lock()
TTL          = 300   # 5-min analysis cache

_chart_cache = {}
_clk         = threading.Lock()
CHART_TTL    = 120   # 2-min chart cache

VALID_INTERVALS  = ["15m", "30m", "1h", "4h", "1d", "1w"]
DEFAULT_SYMBOLS  = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT",  "DOGEUSDT",
    "SOLUSDT", "BNBUSDT", "ADAUSDT",  "AVAXUSDT",
    "LINKUSDT","LTCUSDT", "DOTUSDT",  "NEARUSDT", "ATOMUSDT",
]

# Two-tier scanning: Day entries (15m) + Swing entries (4h) only.
# This keeps max 1 open trade per symbol per tier and eliminates stacking.
SCAN_INTERVALS   = ["15m", "4h"]

# Grows as users view additional coins — persists for the life of the server process
_known_symbols   = set(DEFAULT_SYMBOLS)

# Scanner status — visible in /api/scanner-status
_scanner_status  = {
    "running":          False,
    "scans_completed":  0,
    "last_scan_at":     None,   # ISO timestamp
    "last_scan_ago_s":  None,   # seconds since last scan
    "signals_logged":   0,
    "trades_closed":    0,
}

# Backtest jobs — keyed by job_id (uuid)
_backtest_jobs: dict = {}
_bt_lock = threading.Lock()

# Research campaigns — keyed by campaign_id (uuid)
_campaigns: dict = {}
_camp_lock = threading.Lock()


# ── Session filter ────────────────────────────────────────────────────────────
def _in_active_session() -> bool:
    """
    True during European or US trading sessions (7am–10pm UTC).
    Day (15m) signals are only logged during active sessions when volume is highest.
    Swing (4h) signals fire any time.
    """
    h = datetime.now(timezone.utc).hour
    return 7 <= h < 22


# ── Price-based auto-close helper ────────────────────────────────────────────
def _auto_close_from_data(data: dict, sym: str, interval: str) -> None:
    """Auto-close trades whose TP/SL has been hit based on current price.
    Signal logging is handled ONLY by the background scanner (with bias filter).
    """
    cur_price = data.get("current_price", 0)
    if cur_price:
        learning.auto_close(sym, interval, float(cur_price))


# ── Higher-TF bias filter ─────────────────────────────────────────────────────
def _bias_agrees(signal_dir: str, htf_data: dict) -> bool:
    """
    Returns True only if the higher-timeframe context agrees with the signal.
    Day (15m) signals filtered by 1h context.
    Swing (4h) signals filtered by 1d context.

    Agreement = HTF price above 200 EMA (for LONG) or below (for SHORT),
                OR a confirmed BOS on the HTF in the same direction.
    """
    regime    = htf_data.get("regime", {})
    structure = htf_data.get("structure", {})
    above_200 = regime.get("above_200", False)
    bull_bos  = structure.get("bullish_bos", False) if structure else False
    bear_bos  = structure.get("bearish_bos", False) if structure else False

    if signal_dir == "LONG":
        return above_200 or bull_bos
    else:  # SHORT
        return (not above_200) or bear_bos


# ── Background scanner ────────────────────────────────────────────────────────
def _background_scanner() -> None:
    """
    Runs forever in a daemon thread.
    Every 5 minutes scans ALL known symbols × Day (15m) + Swing (4h) tiers.
    Each signal is filtered by higher-TF bias before logging:
      - 15m LONG  → only logged if 1h is bullish (above 200 EMA or bullish BOS)
      - 15m SHORT → only logged if 1h is bearish
      - 4h  LONG  → only logged if 1d is bullish
      - 4h  SHORT → only logged if 1d is bearish
    """
    _scanner_status["running"] = True
    time.sleep(10)          # short startup pause, then begin immediately

    while True:
        syms = list(_known_symbols)
        scan_signals = 0
        scan_closes  = 0

        for sym in syms:
            # Fetch bias TFs first (1h = bias for 15m day trades, 1d = bias for 4h swings)
            bias_cache: dict[str, dict] = {}
            for bias_tf in ("1h", "1d"):
                try:
                    bias_data = full_analysis(sym, bias_tf)
                    bias_cache[bias_tf] = bias_data
                    # Cache the bias data too
                    with _lock:
                        _cache[f"{sym}_{bias_tf}"] = (bias_data, time.time())
                except Exception:
                    bias_cache[bias_tf] = {}
                time.sleep(1)

            for interval in SCAN_INTERVALS:
                try:
                    data = full_analysis(sym, interval)
                    now  = time.time()
                    data["cache_age"] = 0
                    with _lock:
                        _cache[f"{sym}_{interval}"] = (data, now)

                    # ── Detect per-symbol market regime from already-computed data ──
                    try:
                        detected_regime = regime.detect_regime_from_data(data)
                        learning.save_regime(sym, detected_regime)
                        # Use per-symbol params from research campaign if available,
                        # otherwise fall back to global adaptive defaults.
                        sym_p = learning.get_symbol_params(sym, interval) or {}
                        base_p = {
                            "score_threshold": sym_p.get("score_threshold", learning.get_threshold()),
                            "min_rr":          sym_p.get("min_rr",          3.0),
                            "adx_threshold":   sym_p.get("adx_threshold",   25),
                            "body_ratio_min":  sym_p.get("body_ratio_min",  0.30),
                        }
                        regime_p = regime.get_regime_params(detected_regime, base_p)
                    except Exception:
                        regime_p = {}

                    # Log signal only if all filters pass
                    sig    = data.get("signal")
                    htf    = "1h" if interval == "15m" else "1d"
                    htf_d  = bias_cache.get(htf, {})
                    is_day = interval in ("15m", "30m", "1h")
                    session_ok = (not is_day) or _in_active_session()

                    if sig:
                        # ── Regime-adjusted score filter ──────────────────────
                        regime_threshold = regime_p.get("score_threshold")
                        if regime_threshold and sig.get("score", 0) < regime_threshold:
                            sig = None   # doesn't clear the regime-adjusted bar

                    if sig and _bias_agrees(sig["direction"], htf_d) and session_ok:
                        # ── Medium: 1H confirmation candle for 4H swing signals ──
                        # For 4H entries, the most recent 1H candle must confirm
                        # direction with a real body (not a doji or opposite colour).
                        if interval == "4h":
                            h1_candles = bias_cache.get("1h", {}).get("chart", {}).get("candles", [])
                            if h1_candles:
                                h1 = h1_candles[-1]
                                h1_rng   = h1["high"] - h1["low"]
                                h1_body  = abs(h1["close"] - h1["open"])
                                h1_ratio = h1_body / h1_rng if h1_rng > 0 else 0
                                h1_bull  = h1["close"] > h1["open"]
                                h1_bear  = h1["close"] < h1["open"]
                                if sig["direction"] == "LONG"  and not (h1_bull and h1_ratio >= 0.40):
                                    continue  # No 1H bullish confirmation
                                if sig["direction"] == "SHORT" and not (h1_bear and h1_ratio >= 0.40):
                                    continue  # No 1H bearish confirmation

                        trade_id  = f"{sym}_{interval}_{sig['direction']}_{int(time.time())}"
                        cur_px    = sig.get("current_price", sig["entry"])
                        vwap_val  = data.get("vwap") or 0
                        trade_data = {
                            "id":               trade_id,
                            "symbol":           sym,
                            "interval":         interval,
                            "direction":        sig["direction"],
                            "entry":            sig["entry"],      # structural retest level
                            "current_price":    cur_px,            # where BOS happened
                            "tp":               sig["target"],
                            "sl":               sig["stop"],
                            "score":            sig["score"],
                            "effective_score":  sig["score"],
                            "reason":           sig.get("reason", ""),
                            "factors_snapshot": sig.get("factors_snapshot", {}),
                            "target_basis":     sig.get("target_basis", ""),
                            "opened_at":        datetime.now(timezone.utc).isoformat(),
                            "status":           "pending",  # activated when retest level hit
                            "adx_value":        data.get("adx", {}).get("value", 0),
                            "vwap_side":        ("above" if vwap_val and cur_px > vwap_val
                                                 else "below"),
                        }
                        # ── Claude AI REQUIRED before logging ─────────────
                        # Every trade must pass Claude review.
                        # If API key is set but Claude fails → hold signal (no unreviewed trades).
                        has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
                        ohlcv = data.get("chart", {}).get("candles", [])[-20:]
                        try:
                            ctx       = market_data.get_market_context(sym)
                            history   = learning.get_trades()
                            ai_acc    = learning.get_ai_accuracy()
                            ai_result = ai_analysis.analyze_signal(
                                trade_data, history, ctx,
                                ai_accuracy=ai_acc,
                                ohlcv_candles=ohlcv,
                                htf_context=htf_d,
                            )
                        except Exception:
                            ai_result = {}

                        if has_api_key and not ai_result:
                            pass  # Claude unavailable — skip this scan cycle
                        elif ai_result.get("recommendation") == "skip":
                            pass  # Claude vetoed
                        else:
                            logged = learning.log_trade(trade_data)
                            if logged:
                                scan_signals += 1
                                if ai_result:
                                    learning.update_trade_ai(trade_id, ai_result)
                                    trade_data["ai_analysis"] = ai_result
                                notifications.send_signal_alert({
                                    "symbol":        sym,
                                    "interval":      interval,
                                    "direction":     sig["direction"],
                                    "entry":         sig["entry"],
                                    "tp":            sig["target"],
                                    "sl":            sig["stop"],
                                    "score":         sig["score"],
                                    "reason":        sig.get("reason", ""),
                                    "target_basis":  sig.get("target_basis", ""),
                                    "ai_analysis":   trade_data.get("ai_analysis", {}),
                                    "pending":       True,
                                    "current_price": cur_px,
                                })

                    # Auto-close trades (TP / SL / time / stagnation)
                    cur_price = data.get("current_price", 0)
                    if cur_price:
                        closed, partials = learning.auto_close(sym, interval, float(cur_price))
                        for p in partials:
                            notifications.send_partial_alert(p, p["partial_price"])
                        for c in closed:
                            notifications.send_close_alert(
                                c, c["status"], c["close_price"], c["roi_pct"]
                            )
                        scan_closes += len(closed)

                        # Trailing stop: after 2R, trail SL to structural swing
                        swings     = data.get("swings", {})
                        sw_highs   = [v for _, v in swings.get("highs", [])]
                        sw_lows    = [v for _, v in swings.get("lows",  [])]
                        learning.update_trailing_stops(
                            sym, interval, float(cur_price), sw_highs, sw_lows
                        )

                except Exception:
                    pass
                time.sleep(1)   # respect Binance public API rate limits

        # Update status after each full sweep
        now_iso = datetime.now(timezone.utc).isoformat()
        _scanner_status["scans_completed"] += 1
        _scanner_status["last_scan_at"]    = now_iso
        _scanner_status["signals_logged"]  += scan_signals
        _scanner_status["trades_closed"]   += scan_closes

        time.sleep(300)         # 5 minutes between full sweeps


threading.Thread(target=_background_scanner, daemon=True).start()


# ── Weekly review job ─────────────────────────────────────────────────────────

def _weekly_review_job() -> dict:
    """
    Run the weekly Claude strategy review:
      1. Collect all closed trades (min 10 required).
      2. Run backtester baseline on BTCUSDT 4H with current params.
      3. Ask Claude Sonnet for parameter proposals.
      4. Validate each proposal with backtester — deploy if win_rate improves ≥5%
         AND drawdown doesn't worsen more than 10%.
      5. Send Discord summary.

    Returns a summary dict (also used by /api/weekly-review manual trigger).
    """
    trades = learning.get_trades()
    closed = [t for t in trades if t.get("status") in ("win", "loss")]
    if len(closed) < 10:
        return {"status": "skipped", "reason": f"Need ≥10 closed trades (have {len(closed)})"}

    state = learning.get_learning_state()
    current_params = {
        "signal_threshold": state.get("signal_threshold", 7.0),
        "stop_multiplier":  state.get("stop_multiplier",  0.1),
        "adx_threshold":    25,
        "body_ratio_min":   0.30,
        "min_rr":           3.0,
        **{f"weight_{k}": v["current"]
           for k, v in state.get("weights", {}).items()},
    }

    # Baseline backtest (current params, BTCUSDT 4H, 1 year for speed)
    try:
        baseline = backtester.run_backtest("BTCUSDT", "4h", current_params, years=1)
    except Exception as e:
        baseline = {"win_rate": 0, "max_drawdown": 0, "error": str(e)}

    # Claude weekly review
    review = ai_analysis.get_weekly_review(trades, current_params, state)
    if not review or "error" in review:
        return {"status": "error", "reason": review.get("error", "Claude unavailable")}

    # Validate proposals and deploy if backtester confirms improvement
    deployed = []
    for proposal in review.get("parameter_proposals", []):
        try:
            test_params = dict(current_params)
            test_params[proposal["param"]] = float(proposal["proposed"])
            result = backtester.run_backtest("BTCUSDT", "4h", test_params, years=1)

            baseline_wr = baseline.get("win_rate", 0)
            baseline_dd = baseline.get("max_drawdown", 0) or 1.0
            improved_wr = result.get("win_rate", 0) >= baseline_wr + 0.05
            safe_dd     = result.get("max_drawdown", 0) <= baseline_dd * 1.10

            if improved_wr and safe_dd:
                deployed_ok = learning.auto_deploy_params(
                    {proposal["param"]: float(proposal["proposed"])},
                    reason=proposal.get("reason", "Weekly review"),
                    improvement={"baseline": baseline, "proposed": result},
                )
                if deployed_ok:
                    deployed.append(proposal)
        except Exception:
            pass

    notifications.send_weekly_review_alert(review, deployed, baseline)
    return {
        "status":   "completed",
        "deployed": len(deployed),
        "baseline": baseline,
        "review":   review,
        "deployed_params": deployed,
    }


def _weekly_review_thread() -> None:
    """Daemon thread: sleeps until next Sunday noon UTC, then runs the review weekly."""
    from datetime import timedelta
    while True:
        now               = datetime.now(timezone.utc)
        days_to_sunday    = (6 - now.weekday()) % 7
        next_sunday_noon  = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if days_to_sunday == 0 and now.hour >= 12:
            days_to_sunday = 7   # already past noon this Sunday — wait until next
        next_sunday_noon += timedelta(days=days_to_sunday)
        sleep_s = (next_sunday_noon - now).total_seconds()
        time.sleep(max(sleep_s, 1))
        try:
            _weekly_review_job()
        except Exception:
            pass


threading.Thread(target=_weekly_review_thread, daemon=True).start()


# ── Analysis cache ────────────────────────────────────────────────────────────
def cached_analysis(symbol: str, interval: str = "4h") -> dict:
    key = f"{symbol}_{interval}"
    now = time.time()
    with _lock:
        if key in _cache and now - _cache[key][1] < TTL:
            cached, ts = _cache[key]
            cached["cache_age"] = int(now - ts)
            return cached
    data = full_analysis(symbol, interval)
    data["cache_age"] = 0
    with _lock:
        _cache[key] = (data, now)
    return data


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analysis/<symbol>")
@app.route("/api/analysis/<symbol>/<interval>")
def analysis(symbol: str, interval: str = "4h"):
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    _known_symbols.add(sym)     # register so the scanner picks it up too
    try:
        data = cached_analysis(sym, interval)
        if "error" in data:
            return jsonify(data), 400
        _auto_close_from_data(data, sym, interval)   # TP/SL check only; scanner logs signals
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "symbol": sym}), 500


@app.route("/api/refresh/<symbol>")
@app.route("/api/refresh/<symbol>/<interval>")
def refresh(symbol: str, interval: str = "4h"):
    """Force-clear cache and re-fetch."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    with _lock:
        _cache.pop(f"{sym}_{interval}", None)
    return analysis(sym.replace("USDT", ""), interval)


@app.route("/api/chart/<symbol>/<interval>")
def chart_data(symbol, interval):
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval. Use: {VALID_INTERVALS}"}), 400
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    key = f"{sym}_{interval}"
    now = time.time()
    with _clk:
        if key in _chart_cache and now - _chart_cache[key][1] < CHART_TTL:
            return jsonify(_chart_cache[key][0])
    try:
        data = chart_for_timeframe(sym, interval)
        with _clk:
            _chart_cache[key] = (data, now)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trades")
def get_trades():
    return jsonify(learning.get_trades())


@app.route("/api/trades/<trade_id>/close", methods=["POST"])
def close_trade(trade_id):
    body   = request.get_json() or {}
    status = body.get("status", "cancelled")
    price  = body.get("price")
    if status not in ("win", "loss", "cancelled"):
        return jsonify({"error": "status must be win, loss, or cancelled"}), 400
    try:
        close_px = float(price) if price else 0.0
        result   = learning.close_trade(trade_id, close_px, status)
        if result is None:
            return jsonify({"error": "trade not found or already closed"}), 404
        notifications.send_close_alert(
            result, status, result.get("close_price", close_px), result.get("roi_pct", 0.0)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning")
def get_learning():
    return jsonify(learning.get_learning_state())


@app.route("/api/symbols")
def symbols():
    return jsonify({"symbols": DEFAULT_SYMBOLS})


@app.route("/api/scanner-status")
def scanner_status():
    st = dict(_scanner_status)
    if st["last_scan_at"]:
        last = datetime.fromisoformat(st["last_scan_at"])
        st["last_scan_ago_s"] = int((datetime.now(timezone.utc) - last).total_seconds())
    return jsonify(st)


@app.route("/api/ai-chart/<symbol>/<interval>")
@app.route("/api/ai-chart/<symbol>")
def ai_chart(symbol: str, interval: str = "4h"):
    """On-demand Claude TA analysis for the currently viewed chart."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    if interval not in VALID_INTERVALS:
        interval = "4h"
    try:
        data   = cached_analysis(sym, interval)
        result = ai_analysis.analyze_chart(sym, interval, data)
        if not result:
            return jsonify({"error": "AI unavailable — check ANTHROPIC_API_KEY"}), 503
        if "_error" in result:
            return jsonify({"error": result["_error"]}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-insights")
def get_ai_insights():
    """Ask Claude Sonnet to analyze full trade history and surface patterns."""
    try:
        history   = learning.get_trades()
        ai_acc    = learning.get_ai_accuracy()
        insights  = ai_analysis.get_performance_insights(history, ai_accuracy=ai_acc)
        return jsonify(insights)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/test-discord")
def test_discord():
    """Send a test Discord notification to verify the webhook is configured."""
    import notifications
    if not notifications.WEBHOOK_URL:
        return jsonify({
            "status": "error",
            "message": "DISCORD_WEBHOOK_URL environment variable is not set on Render."
        }), 500
    notifications.send_signal_alert({
        "symbol":       "BTCUSDT",
        "interval":     "15m",
        "direction":    "LONG",
        "entry":        94500.0,
        "tp":           97200.0,
        "sl":           92800.0,
        "score":        8.5,
        "reason":       "TEST — Discord webhook is working correctly ✅",
        "target_basis": "Next major resistance",
    })
    return jsonify({"status": "ok", "message": "Test alert sent to Discord."})


@app.route("/api/test-ai")
def test_ai():
    """Test ANTHROPIC_API_KEY and surface the real error if any."""
    import anthropic as _ant
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return jsonify({"status": "error", "message": "ANTHROPIC_API_KEY env var is not set."}), 500
    try:
        client = _ant.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        return jsonify({"status": "ok", "message": "API key is valid and working.",
                        "model": "claude-haiku-4-5-20251001"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ── Backtester routes ─────────────────────────────────────────────────────────

@app.route("/api/backtest/run", methods=["POST"])
def backtest_run():
    """
    Start an async backtest job.
    Body (JSON): {symbol, interval, years=2, param_grid=null}
      - param_grid=null → single run with default params
      - param_grid={...} → grid search over all combinations

    Returns: {job_id, status: "started", message}
    """
    body       = request.get_json() or {}
    symbol     = body.get("symbol", "BTCUSDT").upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    interval   = body.get("interval", "4h")
    years      = int(body.get("years", 2))
    param_grid = body.get("param_grid")   # None → single default run

    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval. Use: {VALID_INTERVALS}"}), 400

    job_id = str(uuid.uuid4())[:8]
    with _bt_lock:
        _backtest_jobs[job_id] = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    def _run():
        try:
            if param_grid:
                result = backtester.grid_search(symbol, interval, param_grid, years)
            else:
                result = [backtester.run_backtest(
                    symbol, interval, backtester.DEFAULT_PARAMS, years=years
                )]
            with _bt_lock:
                _backtest_jobs[job_id] = {
                    "status":       "done",
                    "results":      result,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            with _bt_lock:
                _backtest_jobs[job_id] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id, "status": "started",
                    "message": f"Backtesting {symbol} {interval} ({years}yr). Poll /api/backtest/status/{job_id}"})


@app.route("/api/backtest/status/<job_id>")
def backtest_status(job_id: str):
    with _bt_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({"job_id": job_id, "status": job["status"]})


@app.route("/api/backtest/results/<job_id>")
def backtest_results(job_id: str):
    with _bt_lock:
        job = _backtest_jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    if job["status"] == "running":
        return jsonify({"status": "running", "message": "Still in progress"}), 202
    if job["status"] == "error":
        return jsonify({"status": "error", "error": job.get("error")}), 500
    return jsonify({"status": "done", "results": job.get("results", [])})


# ── Research campaign routes ──────────────────────────────────────────────────

@app.route("/api/backtest/campaign", methods=["POST"])
def backtest_campaign():
    """
    Launch a full research campaign: grid search for each symbol × interval pair.

    Body (JSON, all optional):
        symbols   — list of symbols (default: CAMPAIGN_SYMBOLS)
        intervals — list of intervals (default: CAMPAIGN_INTERVALS)

    Returns {campaign_id, jobs: {sym_interval: {job_id, status}}}
    Each job uses CAMPAIGN_GRID and MAX_HISTORY_YEARS[interval] for data depth.
    """
    body      = request.get_json() or {}
    symbols   = body.get("symbols",   backtester.CAMPAIGN_SYMBOLS)
    intervals = body.get("intervals", backtester.CAMPAIGN_INTERVALS)

    # Normalise symbols
    symbols = [s.upper() if s.upper().endswith("USDT") else s.upper() + "USDT"
               for s in symbols]

    campaign_id = str(uuid.uuid4())[:8]
    jobs        = {}

    for sym in symbols:
        for interval in intervals:
            job_id = str(uuid.uuid4())[:8]
            years  = backtester.MAX_HISTORY_YEARS.get(interval, 2)
            key    = f"{sym}_{interval}"
            jobs[key] = {"job_id": job_id, "status": "running",
                         "symbol": sym, "interval": interval}

            with _bt_lock:
                _backtest_jobs[job_id] = {
                    "status":     "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }

            def _run(s=sym, iv=interval, jid=job_id, yr=years):
                try:
                    result = backtester.grid_search(
                        s, iv, backtester.CAMPAIGN_GRID, years=yr
                    )
                    with _bt_lock:
                        _backtest_jobs[jid] = {
                            "status":       "done",
                            "results":      result,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }
                except Exception as e:
                    with _bt_lock:
                        _backtest_jobs[jid] = {"status": "error", "error": str(e)}

            threading.Thread(target=_run, daemon=True).start()

    with _camp_lock:
        _campaigns[campaign_id] = {
            "jobs":       jobs,
            "symbols":    symbols,
            "intervals":  intervals,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    return jsonify({
        "campaign_id": campaign_id,
        "jobs":        jobs,
        "status":      "started",
        "message":     (f"Campaign started: {len(symbols)} symbols × "
                        f"{len(intervals)} intervals = {len(jobs)} jobs running in parallel."),
    })


@app.route("/api/backtest/campaign/status/<campaign_id>")
def campaign_status(campaign_id: str):
    """Return per-job completion status for a running campaign."""
    with _camp_lock:
        camp = _campaigns.get(campaign_id)
    if not camp:
        return jsonify({"error": "campaign not found"}), 404

    jobs_out = {}
    with _bt_lock:
        for key, meta in camp["jobs"].items():
            job = _backtest_jobs.get(meta["job_id"], {})
            jobs_out[key] = {
                "job_id":   meta["job_id"],
                "symbol":   meta["symbol"],
                "interval": meta["interval"],
                "status":   job.get("status", "unknown"),
            }

    all_done = all(v["status"] in ("done", "error") for v in jobs_out.values())
    return jsonify({
        "campaign_id": campaign_id,
        "status":      "done" if all_done else "running",
        "jobs":        jobs_out,
        "started_at":  camp["started_at"],
    })


@app.route("/api/backtest/campaign/results/<campaign_id>")
def campaign_results(campaign_id: str):
    """Return best params + top-5 results for each completed job in the campaign."""
    with _camp_lock:
        camp = _campaigns.get(campaign_id)
    if not camp:
        return jsonify({"error": "campaign not found"}), 404

    results_out = {}
    with _bt_lock:
        for key, meta in camp["jobs"].items():
            job = _backtest_jobs.get(meta["job_id"], {})
            st  = job.get("status", "unknown")
            if st == "done":
                all_r = [r for r in (job.get("results") or []) if not r.get("error")]
                best  = all_r[0] if all_r else None
                results_out[key] = {
                    "status":   "done",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                    "best":     best,
                    "top5":     all_r[:5],
                }
            elif st == "error":
                results_out[key] = {
                    "status":   "error",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                    "error":    job.get("error"),
                }
            else:
                results_out[key] = {
                    "status":   "running",
                    "symbol":   meta["symbol"],
                    "interval": meta["interval"],
                }

    all_done = all(v["status"] in ("done", "error") for v in results_out.values())
    return jsonify({
        "campaign_id": campaign_id,
        "status":      "done" if all_done else "running",
        "results":     results_out,
    })


@app.route("/api/apply-symbol-params", methods=["POST"])
def apply_symbol_params():
    """
    Save approved backtest params for a specific symbol+interval.
    The scanner will use these on its next sweep instead of global defaults.

    Body: {symbol, interval, params: {adx_threshold, score_threshold, min_rr, ...}}
    """
    body     = request.get_json() or {}
    symbol   = body.get("symbol", "").upper()
    interval = body.get("interval", "")
    params   = body.get("params", {})

    if not symbol or not interval or not params:
        return jsonify({"error": "symbol, interval, and params are required"}), 400
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"interval must be one of {VALID_INTERVALS}"}), 400

    learning.save_symbol_params(symbol, interval, params)
    return jsonify({
        "status":   "saved",
        "symbol":   symbol,
        "interval": interval,
        "params":   params,
        "message":  f"Scanner will now use these params for {symbol} {interval}.",
    })


# ── Regime route ──────────────────────────────────────────────────────────────

@app.route("/api/regime/<symbol>")
def get_regime(symbol: str):
    """Return the most recently detected market regime for a symbol."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    stored = learning.get_regime(sym)
    if stored:
        return jsonify({"symbol": sym, **stored})
    # Compute on-demand if not yet detected
    try:
        data = cached_analysis(sym, "4h")
        detected = regime.detect_regime_from_data(data)
        learning.save_regime(sym, detected)
        return jsonify({"symbol": sym, **detected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Weekly review manual trigger ──────────────────────────────────────────────

@app.route("/api/weekly-review")
def weekly_review():
    """Manually trigger the weekly Claude strategy review (synchronous)."""
    try:
        result = _weekly_review_job()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  Crypto Analysis Dashboard")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
