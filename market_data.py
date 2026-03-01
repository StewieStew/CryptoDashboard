"""
External market context — Fear & Greed, BTC Dominance, Funding Rates, Open Interest.
All data is cached for 1 hour to avoid hammering external APIs.
"""

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
