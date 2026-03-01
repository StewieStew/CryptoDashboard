"""
Crypto Analysis Dashboard — Flask Server
"""

from flask import Flask, jsonify, render_template, request
from analysis import full_analysis, chart_for_timeframe
from datetime import datetime, timezone
import time
import threading
import os
import learning
import notifications
import ai_analysis
import market_data

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

                    # Log signal only if higher-TF bias agrees + session check
                    sig = data.get("signal")
                    if sig:
                        htf   = "1h" if interval == "15m" else "1d"
                        htf_d = bias_cache.get(htf, {})
                        is_day = interval in ("15m", "30m", "1h")
                        session_ok = (not is_day) or _in_active_session()

                        if _bias_agrees(sig["direction"], htf_d) and session_ok:
                            trade_id   = f"{sym}_{interval}_{sig['direction']}_{int(time.time())}"
                            trade_data = {
                                "id":               trade_id,
                                "symbol":           sym,
                                "interval":         interval,
                                "direction":        sig["direction"],
                                "entry":            sig["entry"],
                                "tp":               sig["target"],
                                "sl":               sig["stop"],
                                "score":            sig["score"],
                                "effective_score":  sig["score"],
                                "reason":           sig.get("reason", ""),
                                "factors_snapshot": sig.get("factors_snapshot", {}),
                                "target_basis":     sig.get("target_basis", ""),
                                "opened_at":        datetime.now(timezone.utc).isoformat(),
                                # Pass ADX/VWAP context for AI evaluation
                                "adx_value":        data.get("adx", {}).get("value", 0),
                                "vwap_side":        ("above" if data.get("vwap") and
                                                     sig["entry"] > data.get("vwap", 0) else "below"),
                            }
                            # ── Claude AI evaluates BEFORE logging ───────
                            # If Claude says skip → signal is vetoed entirely.
                            # If API is down (empty result) → fall back to log.
                            try:
                                ctx       = market_data.get_market_context(sym)
                                history   = learning.get_trades()
                                ai_acc    = learning.get_ai_accuracy()
                                ai_result = ai_analysis.analyze_signal(
                                    trade_data, history, ctx, ai_accuracy=ai_acc)
                            except Exception:
                                ai_result = {}

                            if ai_result.get("recommendation") == "skip":
                                pass  # Claude vetoed — don't log, don't notify
                            else:
                                logged = learning.log_trade(trade_data)
                                if logged:
                                    scan_signals += 1
                                    if ai_result:
                                        learning.update_trade_ai(trade_id, ai_result)
                                        trade_data["ai_analysis"] = ai_result
                                    notifications.send_signal_alert({
                                        "symbol":       sym,
                                        "interval":     interval,
                                        "direction":    sig["direction"],
                                        "entry":        sig["entry"],
                                        "tp":           sig["target"],
                                        "sl":           sig["stop"],
                                        "score":        sig["score"],
                                        "reason":       sig.get("reason", ""),
                                        "target_basis": sig.get("target_basis", ""),
                                        "ai_analysis":  trade_data.get("ai_analysis", {}),
                                    })

                    # Auto-close trades
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  Crypto Analysis Dashboard")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
