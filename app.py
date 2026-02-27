"""
Crypto Analysis Dashboard — Flask Server
"""

from flask import Flask, jsonify, render_template
from analysis import full_analysis
import time
import threading
import os

app   = Flask(__name__)
_cache = {}
_lock  = threading.Lock()
TTL    = 300  # 5-minute cache

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT"]


def cached_analysis(symbol: str) -> dict:
    now = time.time()
    with _lock:
        if symbol in _cache and now - _cache[symbol][1] < TTL:
            cached, ts = _cache[symbol]
            cached["cache_age"] = int(now - ts)
            return cached
    data = full_analysis(symbol)
    data["cache_age"] = 0
    with _lock:
        _cache[symbol] = (data, now)
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analysis/<symbol>")
def analysis(symbol: str):
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    try:
        data = cached_analysis(sym)
        if "error" in data:
            return jsonify(data), 400
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "symbol": sym}), 500


@app.route("/api/refresh/<symbol>")
def refresh(symbol: str):
    """Force-clear cache and re-fetch."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    with _lock:
        _cache.pop(sym, None)
    return analysis(sym.replace("USDT", ""))


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
