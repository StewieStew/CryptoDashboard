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

app          = Flask(__name__)
_cache       = {}
_lock        = threading.Lock()
TTL          = 300   # 5-min analysis cache

_chart_cache = {}
_clk         = threading.Lock()
CHART_TTL    = 120   # 2-min chart cache

VALID_INTERVALS  = ["15m", "30m", "1h", "4h", "1d", "1w"]
DEFAULT_SYMBOLS  = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]

# Intervals the background scanner checks automatically
SCAN_INTERVALS   = ["1h", "4h", "1d"]

# Grows as users view additional coins — persists for the life of the server process
_known_symbols   = set(DEFAULT_SYMBOLS)


# ── Shared signal-logging helper ─────────────────────────────────────────────
def _log_signal_from_data(data: dict, sym: str, interval: str) -> None:
    """Log a signal + auto-close trades based on current price."""
    sig = data.get("signal")
    if sig:
        trade_id = f"{sym}_{interval}_{sig['direction']}_{int(time.time())}"
        learning.log_trade({
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
        })
    cur_price = data.get("current_price", 0)
    if cur_price:
        learning.auto_close(sym, interval, float(cur_price))


# ── Background scanner ────────────────────────────────────────────────────────
def _background_scanner() -> None:
    """
    Runs forever in a daemon thread.
    Every 5 minutes scans all known symbols × SCAN_INTERVALS for new signals
    and auto-closes trades — no user interaction required.
    """
    time.sleep(30)          # let gunicorn fully start before the first scan
    while True:
        syms = list(_known_symbols)
        for sym in syms:
            for interval in SCAN_INTERVALS:
                try:
                    data = full_analysis(sym, interval)
                    now  = time.time()
                    data["cache_age"] = 0
                    with _lock:
                        _cache[f"{sym}_{interval}"] = (data, now)
                    _log_signal_from_data(data, sym, interval)
                except Exception:
                    pass
                time.sleep(2)   # small gap — respect Binance rate limits
        time.sleep(300)         # wait 5 min before the next full sweep


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
        _log_signal_from_data(data, sym, interval)
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
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/learning")
def get_learning():
    return jsonify(learning.get_learning_state())


@app.route("/api/symbols")
def symbols():
    return jsonify({"symbols": DEFAULT_SYMBOLS})


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
