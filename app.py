"""
Crypto Analysis Dashboard — Flask Server
"""

from flask import Flask, jsonify, render_template, request
from analysis import full_analysis, chart_for_timeframe
from datetime import datetime, timezone
import time
import threading
import os

app          = Flask(__name__)
_cache       = {}
_lock        = threading.Lock()
TTL          = 300   # 5-min analysis cache

_chart_cache = {}
_clk         = threading.Lock()
CHART_TTL    = 120   # 2-min chart cache

VALID_INTERVALS = ["15m", "30m", "1h", "4h", "1d", "1w"]

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]

# ── In-memory trade log ──────────────────────────────────
_trades      = []   # list of trade dicts, newest appended last
_trades_lock = threading.Lock()


def _close_trade_inner(t: dict, price: float, status: str) -> None:
    """Mutate a trade dict in-place. Must be called while holding _trades_lock."""
    t["status"]      = status
    t["closed_at"]   = datetime.now(timezone.utc).isoformat()
    t["close_price"] = round(float(price), 8)
    entry = t["entry"]
    if t["direction"] == "LONG":
        t["roi_pct"] = round((float(price) - entry) / entry * 100, 2)
    else:
        t["roi_pct"] = round((entry - float(price)) / entry * 100, 2)


def log_signal(signal: dict, symbol: str, interval: str) -> None:
    """Append a new trade entry — skips if an open trade already exists for this key."""
    if not signal:
        return
    direction = signal["direction"]
    with _trades_lock:
        for t in _trades:
            if (t["symbol"] == symbol and t["interval"] == interval
                    and t["direction"] == direction and t["status"] == "open"):
                return  # already tracking this signal
        _trades.append({
            "id":         f"{symbol}_{interval}_{direction}_{int(time.time())}",
            "symbol":     symbol,
            "interval":   interval,
            "direction":  direction,
            "entry":      signal["entry"],
            "tp":         signal["target"],
            "sl":         signal["stop"],
            "score":      signal["score"],
            "reason":     signal.get("reason", ""),
            "status":     "open",
            "opened_at":  datetime.now(timezone.utc).isoformat(),
            "closed_at":  None,
            "close_price":None,
            "roi_pct":    None,
        })


def auto_close_trades(symbol: str, interval: str, current_price: float) -> None:
    """Auto-close open trades when current price crosses TP or SL."""
    if not current_price:
        return
    with _trades_lock:
        for t in _trades:
            if t["symbol"] != symbol or t["interval"] != interval or t["status"] != "open":
                continue
            if t["direction"] == "LONG":
                if current_price >= t["tp"]:
                    _close_trade_inner(t, current_price, "win")
                elif current_price <= t["sl"]:
                    _close_trade_inner(t, current_price, "loss")
            elif t["direction"] == "SHORT":
                if current_price <= t["tp"]:
                    _close_trade_inner(t, current_price, "win")
                elif current_price >= t["sl"]:
                    _close_trade_inner(t, current_price, "loss")


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
    try:
        data = cached_analysis(sym, interval)
        if "error" in data:
            return jsonify(data), 400
        # Log new signal and auto-close existing open trades
        if data.get("signal"):
            log_signal(data["signal"], sym, interval)
        auto_close_trades(sym, interval, data.get("current_price", 0))
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
    key = f"{sym}_{interval}"
    with _lock:
        _cache.pop(key, None)
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
    with _trades_lock:
        return jsonify(list(reversed(_trades)))  # newest first


@app.route("/api/trades/<trade_id>/close", methods=["POST"])
def close_trade(trade_id):
    body   = request.get_json() or {}
    status = body.get("status", "cancelled")
    price  = body.get("price")
    if status not in ("win", "loss", "cancelled"):
        return jsonify({"error": "status must be win, loss, or cancelled"}), 400
    with _trades_lock:
        for t in _trades:
            if t["id"] == trade_id:
                if t["status"] != "open":
                    return jsonify({"error": "trade already closed"}), 400
                close_px = float(price) if price else t["entry"]
                _close_trade_inner(t, close_px, status)
                return jsonify(t)
    return jsonify({"error": "trade not found"}), 404


@app.route("/api/symbols")
def symbols():
    return jsonify({"symbols": DEFAULT_SYMBOLS})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("  Crypto Analysis Dashboard")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
