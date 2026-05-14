"""
External market context — Fear & Greed, BTC Dominance, Funding Rates, Open Interest.
All data is cached for 1 hour to avoid hammering external APIs.
"""
from __future__ import annotations

import time
import threading
import requests

_lock  = threading.Lock()
_cache = {}
_TTL   = 3600   # 1 hour


def _cached(key: str, fetch_fn, ttl: int = _TTL):
    """Generic cache helper."""
    now = time.time()
    with _lock:
        if key in _cache and now - _cache[key][1] < ttl:
            return _cache[key][0]
    try:
        data = fetch_fn()
    except Exception:
        with _lock:
            return _cache.get(key, (None, 0))[0]   # return stale on error
    with _lock:
        _cache[key] = (data, now)
    return data


# ── Fear & Greed Index ────────────────────────────────────────────────────────

def get_fear_greed() -> dict:
    """Returns {value: int, label: str} or {}."""
    def _fetch():
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    result = _cached("fear_greed", _fetch)
    return result or {}


# ── BTC Dominance ─────────────────────────────────────────────────────────────

def get_btc_dominance() -> float:
    """Returns BTC market cap dominance as a percentage (e.g. 52.3), or 0."""
    def _fetch():
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        return r.json()["data"]["market_cap_percentage"]["btc"]
    result = _cached("btc_dominance", _fetch)
    return result or 0.0


# ── Funding Rate ──────────────────────────────────────────────────────────────

def get_funding_rate(symbol: str) -> float:
    """
    Returns the latest perpetual funding rate as a percentage for `symbol`
    (e.g. 0.0100 = 0.01%).  Returns 0 on failure.
    Uses Binance global futures API (public, no auth needed).
    """
    def _fetch():
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 1},
            timeout=8,
        )
        data = r.json()
        if data:
            return float(data[-1]["fundingRate"]) * 100
        return 0.0
    result = _cached(f"funding_{symbol}", _fetch, ttl=900)   # 15-min cache
    return result or 0.0


# ── Open Interest ─────────────────────────────────────────────────────────────

def get_open_interest(symbol: str) -> float:
    """
    Returns current open interest in contracts for `symbol`, or 0.
    """
    def _fetch():
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=8,
        )
        return float(r.json()["openInterest"])
    result = _cached(f"oi_{symbol}", _fetch, ttl=300)   # 5-min cache
    return result or 0.0


# ── Live price + historical bar helpers ──────────────────────────────────────

_BINANCE_SPOT = "https://api.binance.us/api/v3"

def get_live_price(symbol: str) -> float | None:
    """Fetch current spot price from Binance. No cache — always fresh."""
    try:
        r = requests.get(
            f"{_BINANCE_SPOT}/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        return float(r.json()["price"])
    except Exception:
        return None


def get_recent_1m_extreme(symbol: str, since_ms: int | None = None) -> dict:
    """
    Fetch the last 10 1m candles (including the forming one) and return merged
    {high, low}.  Used by the live monitor to catch brief TP/SL wicks that
    occur between the 60-second price checks.  10 candles gives a ~10-minute
    lookback window, ensuring a wick is still visible even if the monitor skips
    a cycle or runs slightly late.

    since_ms: if provided, only include candles whose open timestamp (ms) is
              >= since_ms.  Pass the trade's opened_at milliseconds to prevent
              pre-open candles from falsely triggering SL/TP hits.

    Returns {} on error or if no qualifying candles.
    """
    try:
        r = requests.get(
            f"{_BINANCE_SPOT}/klines",
            params={"symbol": symbol, "interval": "1m", "limit": 10},
            timeout=5,
        )
        bars = r.json()
        # Include the forming candle — its low/high are real traded prices.
        # Skipping it caused missed TP wicks that happened in the live candle.
        all_bars = [b for b in bars if isinstance(b, list)]
        if since_ms is not None:
            all_bars = [b for b in all_bars if int(b[0]) >= since_ms]
        if not all_bars:
            return {}
        highs = [float(b[2]) for b in all_bars]
        lows  = [float(b[3]) for b in all_bars]
        return {"high": max(highs), "low": min(lows)}
    except Exception:
        return {}


def fetch_1m_bars_since(symbol: str, since_ms: int) -> list[dict]:
    """
    Fetch ALL 1m OHLCV bars from Binance from since_ms until now.
    Uses pagination so trades open longer than ~16 hours (1000-bar limit) are
    fully covered.  Each bar: {high, low}.
    """
    result: list[dict] = []
    start = since_ms
    while True:
        try:
            r = requests.get(
                f"{_BINANCE_SPOT}/klines",
                params={"symbol": symbol, "interval": "1m",
                        "startTime": start, "limit": 1000},
                timeout=10,
            )
            bars = r.json()
            if not isinstance(bars, list) or not bars:
                break
            parsed = [
                {"high": float(b[2]), "low": float(b[3]), "_ts": int(b[0])}
                for b in bars if isinstance(b, list)
            ]
            if not parsed:
                break
            result.extend(parsed)
            if len(parsed) < 1000:
                break   # fetched everything available — done
            # Advance start past the last bar's open time (1-minute step)
            start = parsed[-1]["_ts"] + 60_000
            time.sleep(0.1)   # mild rate-limit pacing
        except Exception:
            break
    return [{"high": b["high"], "low": b["low"]} for b in result]


# ── Full context bundle ───────────────────────────────────────────────────────

def get_market_context(symbol: str) -> dict:
    """
    Returns all market context for a single symbol.
    Called once per signal before AI analysis.
    """
    return {
        "fear_greed":    get_fear_greed(),
        "btc_dominance": get_btc_dominance(),
        "funding_rate":  get_funding_rate(symbol),
        "open_interest": get_open_interest(symbol),
    }
