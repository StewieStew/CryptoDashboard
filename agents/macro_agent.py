"""
MACRO AGENT — runs every 30 minutes
Job: Read the big picture. News, sentiment, Fear & Greed, BTC dominance,
     funding rates. Determine overall market regime and coin biases.
Output: regime, coin biases, news highlights, avoid list
"""
from __future__ import annotations
import json, os, time
from datetime import datetime, timezone

import requests
import anthropic

from agents.state import set_state, get_state, add_report, add_knowledge, post_to_render

COINS = ["BTC", "ETH", "XRP", "DOGE", "SOL"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def fetch_fear_greed() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=3", timeout=8)
        data = r.json()["data"]
        return {
            "current":   {"value": int(data[0]["value"]), "label": data[0]["value_classification"]},
            "yesterday": {"value": int(data[1]["value"]), "label": data[1]["value_classification"]},
        }
    except Exception:
        return {}


def fetch_btc_dominance() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        return float(r.json()["data"]["market_cap_percentage"].get("btc", 0))
    except Exception:
        return 0.0


def fetch_funding_rates() -> dict:
    rates = {}
    for coin in COINS:
        sym = f"{coin}USDT"
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": sym, "limit": 1},
                timeout=8,
            )
            d = r.json()
            if d:
                rates[sym] = float(d[-1].get("fundingRate", 0))
        except Exception:
            rates[sym] = 0.0
        time.sleep(0.3)
    return rates


def fetch_news_all() -> dict:
    """Fetch news for all coins at once."""
    news_by_coin = {}
    for coin in COINS:
        try:
            r = requests.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={"auth_token": "free", "currencies": coin,
                        "public": "true", "filter": "hot"},
                timeout=8,
            )
            items = r.json().get("results", [])[:6]
            news_by_coin[coin] = [
                {
                    "title": item.get("title", ""),
                    "sentiment": (
                        "bullish" if item.get("votes", {}).get("positive", 0) >
                                     item.get("votes", {}).get("negative", 0)
                        else "bearish" if item.get("votes", {}).get("negative", 0) >
                                          item.get("votes", {}).get("positive", 0)
                        else "neutral"
                    ),
                }
                for item in items
            ]
            time.sleep(0.5)
        except Exception:
            news_by_coin[coin] = []
    return news_by_coin


def fetch_btc_price_context() -> dict:
    """Get BTC 1d candles to understand macro trend."""
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1d", "limit": 14},
            timeout=10,
        )
        candles = r.json()
        closes = [float(c[4]) for c in candles]
        cur    = closes[-1]
        week   = closes[-7]
        month  = closes[0]
        return {
            "current":      cur,
            "change_7d":    (cur - week)  / week  * 100,
            "change_14d":   (cur - month) / month * 100,
            "above_ema20":  cur > sum(closes) / len(closes),
            "trend":        "up" if closes[-1] > closes[-3] > closes[-7] else
                            "down" if closes[-1] < closes[-3] < closes[-7] else "sideways",
        }
    except Exception:
        return {}


def run() -> dict:
    """Execute macro analysis. Returns findings dict."""
    print("[MACRO AGENT] Running...", flush=True)

    fg       = fetch_fear_greed()
    btc_dom  = fetch_btc_dominance()
    funding  = fetch_funding_rates()
    news     = fetch_news_all()
    btc_ctx  = fetch_btc_price_context()

    # Format for Claude
    fg_str = f"Fear & Greed: {fg.get('current',{}).get('value','?')} ({fg.get('current',{}).get('label','?')}) | Yesterday: {fg.get('yesterday',{}).get('value','?')}"
    fund_str = "\n".join(f"  {sym}: {rate:+.4f}% ({'longs paying' if rate>0 else 'shorts paying'})"
                         for sym, rate in funding.items())
    news_str = ""
    for coin, items in news.items():
        if items:
            news_str += f"\n{coin}:\n"
            news_str += "\n".join(f"  [{i['sentiment']}] {i['title']}" for i in items[:4])

    btc_str = (f"BTC ${btc_ctx.get('current',0):,.0f} | "
               f"7d: {btc_ctx.get('change_7d',0):+.1f}% | "
               f"14d: {btc_ctx.get('change_14d',0):+.1f}% | "
               f"Trend: {btc_ctx.get('trend','?')}")

    # Get prior regime for context
    prior = get_state("macro_regime", {})
    prior_str = f"Prior regime: {prior.get('regime_type','unknown')} ({prior.get('timestamp','')})" if prior else "First run."

    prompt = f"""You are the Macro Analyst on a crypto trading desk. Your job is to assess the big-picture market environment.

{fg_str}
BTC Dominance: {btc_dom:.1f}%
{btc_str}

Funding Rates:
{fund_str}

Recent News:
{news_str}

{prior_str}

Assess the macro environment and output ONLY this JSON:
{{
  "regime_type": "<bull_trending|bear_trending|ranging|uncertain>",
  "regime_strength": <1-10>,
  "macro_summary": "<2-3 sentences on overall market state>",
  "coin_bias": {{
    "BTC": "<long|short|neutral>",
    "ETH": "<long|short|neutral>",
    "XRP": "<long|short|neutral>",
    "DOGE": "<long|short|neutral>",
    "SOL": "<long|short|neutral>"
  }},
  "coin_bias_reasons": {{
    "BTC": "<one reason>",
    "ETH": "<one reason>",
    "XRP": "<one reason>",
    "DOGE": "<one reason>",
    "SOL": "<one reason>"
  }},
  "avoid_longs": ["<coin if applicable>"],
  "avoid_shorts": ["<coin if applicable>"],
  "key_news": "<most impactful news item, or none>",
  "risk_level": "<low|medium|high>",
  "trading_advice": "<one sentence — what the trading desk should do right now>"
}}"""

    client = _claude()
    result = {}
    if client:
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[MACRO AGENT] Claude error: {e}", flush=True)

    result["timestamp"]  = datetime.now(timezone.utc).isoformat()
    result["fear_greed"] = fg
    result["btc_dom"]    = btc_dom
    result["funding"]    = funding
    result["btc_price"]  = btc_ctx

    # Store in shared state
    set_state("macro_regime", result)
    add_report("macro", "regime_analysis", result)
    add_knowledge("macro_snapshots", {
        "regime":   result.get("regime_type"),
        "strength": result.get("regime_strength"),
        "summary":  result.get("macro_summary"),
    })

    # Post to Render
    post_to_render("/api/agent/insight", {
        "type":      "macro_analysis",
        "agent":     "macro",
        "timestamp": result["timestamp"],
        "regime":    result.get("regime_type"),
        "strength":  result.get("regime_strength"),
        "summary":   result.get("macro_summary"),
        "coin_bias": result.get("coin_bias"),
        "risk_level": result.get("risk_level"),
        "advice":    result.get("trading_advice"),
    })

    print(f"[MACRO AGENT] Done. Regime: {result.get('regime_type')} | "
          f"Risk: {result.get('risk_level')} | "
          f"Advice: {result.get('trading_advice','')[:60]}", flush=True)
    return result
