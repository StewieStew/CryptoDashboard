"""
ANALYST AGENT — runs every 15 minutes
Job: Deep price action analysis per coin. Identify key levels, structure,
     liquidity clusters, orderbook imbalances. Rate each coin for setup quality.
Output: coin ratings 1-10, key levels, liquidity map, setup quality
"""
from __future__ import annotations
import json, os, time
from datetime import datetime, timezone

import requests
import anthropic
import numpy as np

from agents.state import (set_state, get_state, add_report, add_knowledge,
                          post_to_render)

COINS        = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BINANCE_BASE  = "https://api.binance.us/api/v3"


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def fetch_candles(symbol: str, interval: str, limit: int) -> list:
    try:
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        return [{"time": c[0], "open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
                for c in r.json()]
    except Exception:
        return []


def fetch_orderbook(symbol: str, limit: int = 50) -> dict:
    try:
        r = requests.get(f"{BINANCE_BASE}/depth",
                         params={"symbol": symbol, "limit": limit}, timeout=8)
        return r.json()
    except Exception:
        return {}


def calc_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period+2):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[-period:])
    avg_l  = np.mean(losses[-period:])
    return float(100.0 - 100.0 / (1.0 + avg_g / avg_l)) if avg_l > 0 else 50.0


def find_liquidity_clusters(candles: list, price: float) -> list:
    """Detect equal highs/lows within 0.3% of each other."""
    clusters = []
    try:
        highs = [c["high"] for c in candles[-150:]]
        lows  = [c["low"]  for c in candles[-150:]]

        seen = set()
        for h in highs:
            key = round(h / price * 1000)
            if key in seen:
                continue
            count = sum(1 for hh in highs if abs(hh - h) / h < 0.003)
            if count >= 2:
                seen.add(key)
                clusters.append({
                    "type": "sell_stops",
                    "price": round(h, 8),
                    "strength": count,
                    "dist_pct": round(abs(h - price) / price * 100, 2),
                    "side": "above" if h > price else "below",
                })

        seen = set()
        for l in lows:
            key = round(l / price * 1000)
            if key in seen:
                continue
            count = sum(1 for ll in lows if abs(ll - l) / l < 0.003)
            if count >= 2:
                seen.add(key)
                clusters.append({
                    "type": "buy_stops",
                    "price": round(l, 8),
                    "strength": count,
                    "dist_pct": round(abs(l - price) / price * 100, 2),
                    "side": "above" if l > price else "below",
                })

        clusters.sort(key=lambda x: x["dist_pct"])
        return clusters[:10]
    except Exception:
        return []


def analyze_orderbook(ob: dict, price: float) -> dict:
    try:
        bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:30]]
        asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:30]]

        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        ratio   = bid_vol / ask_vol if ask_vol > 0 else 1.0

        # Largest walls
        top_bid = max(bids, key=lambda x: x[1]) if bids else (0, 0)
        top_ask = min(asks, key=lambda x: x[1]) if asks else (0, 0)  # lowest ask with most volume
        top_ask = max(asks, key=lambda x: x[1]) if asks else (0, 0)

        return {
            "bid_ask_ratio": round(ratio, 3),
            "pressure":      "buy" if ratio > 1.3 else "sell" if ratio < 0.7 else "neutral",
            "largest_bid":   {"price": top_bid[0], "size": round(top_bid[1], 2)},
            "largest_ask":   {"price": top_ask[0], "size": round(top_ask[1], 2)},
            "spread_pct":    round((asks[0][0] - bids[0][0]) / price * 100, 4) if bids and asks else 0,
        }
    except Exception:
        return {}


def format_candles_brief(candles: list, n: int = 20) -> str:
    lines = []
    for c in candles[-n:]:
        chg = (c["close"] - c["open"]) / c["open"] * 100
        lines.append(f"O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} "
                     f"C={c['close']:.4f} V={c.get('volume',0):,.0f} ({chg:+.2f}%)")
    return "\n".join(lines)


def run() -> dict:
    """Execute analyst pass on all coins."""
    print("[ANALYST AGENT] Running...", flush=True)

    macro = get_state("macro_regime", {})
    macro_regime = macro.get("regime_type", "uncertain")
    coin_bias    = macro.get("coin_bias", {})

    all_ratings  = {}
    coin_details = {}

    for sym in COINS:
        coin = sym.replace("USDT", "")
        print(f"[ANALYST AGENT]   Analyzing {sym}...", flush=True)

        c1h  = fetch_candles(sym, "1h",  100)
        c4h  = fetch_candles(sym, "4h",   50)
        c1d  = fetch_candles(sym, "1d",   30)
        ob   = fetch_orderbook(sym, 50)

        if not c1h:
            continue

        price    = c1h[-1]["close"]
        closes1h = [c["close"] for c in c1h]
        closes4h = [c["close"] for c in c4h]

        rsi_1h = calc_rsi(closes1h)
        rsi_4h = calc_rsi(closes4h)

        change_24h = (price - closes1h[-25]) / closes1h[-25] * 100 if len(closes1h) >= 25 else 0
        change_7d  = (price - closes1h[-169]) / closes1h[-169] * 100 if len(closes1h) >= 169 else 0

        clusters   = find_liquidity_clusters(c1h, price)
        ob_analysis = analyze_orderbook(ob, price)

        # Detect BOS
        recent_high = max(c["high"]  for c in c1h[-20:])
        prior_high  = max(c["high"]  for c in c1h[-40:-20])
        recent_low  = min(c["low"]   for c in c1h[-20:])
        prior_low   = min(c["low"]   for c in c1h[-40:-20])
        bullish_bos = price > prior_high
        bearish_bos = price < prior_low

        bias = coin_bias.get(coin, "neutral")

        coin_details[sym] = {
            "price":        price,
            "change_24h":   round(change_24h, 2),
            "change_7d":    round(change_7d, 2),
            "rsi_1h":       round(rsi_1h, 1),
            "rsi_4h":       round(rsi_4h, 1),
            "bullish_bos":  bullish_bos,
            "bearish_bos":  bearish_bos,
            "ob":           ob_analysis,
            "clusters":     clusters[:5],
            "macro_bias":   bias,
            "candles_brief_1h": format_candles_brief(c1h, 15),
            "candles_brief_4h": format_candles_brief(c4h, 8),
        }
        time.sleep(1)

    # ── Ask Claude to rate all coins and identify best setups ──────────────
    coins_prompt = ""
    for sym, d in coin_details.items():
        coin = sym.replace("USDT", "")
        liq_str = "\n".join(
            f"    {c['type']} @ {c['price']:.6f} (strength={c['strength']}, {c['dist_pct']:.1f}% away, {c['side']})"
            for c in d.get("clusters", [])[:4]
        ) or "    None detected"

        coins_prompt += f"""
── {sym} ──────────────────────
Price: ${d['price']:.6f} | 24h: {d['change_24h']:+.1f}% | 7d: {d['change_7d']:+.1f}%
RSI 1h: {d['rsi_1h']:.0f} | RSI 4h: {d['rsi_4h']:.0f}
Structure: {'BULLISH BOS ✓' if d['bullish_bos'] else ''} {'BEARISH BOS ✓' if d['bearish_bos'] else ''} {'No BOS' if not d['bullish_bos'] and not d['bearish_bos'] else ''}
Order book pressure: {d['ob'].get('pressure','?')} (bid/ask ratio: {d['ob'].get('bid_ask_ratio','?')})
Macro bias from Macro Agent: {d['macro_bias'].upper()}

Liquidity clusters:
{liq_str}

1H candles (last 15):
{d.get('candles_brief_1h','')}

4H candles (last 8):
{d.get('candles_brief_4h','')}
"""

    client = _claude()
    ratings = {}
    if client:
        prompt = f"""You are the Lead Technical Analyst on a crypto trading desk.
Macro regime: {macro_regime}

Analyze all 5 coins and rate each for trade setup quality RIGHT NOW.

{coins_prompt}

Rate each coin 1-10 for setup quality (10 = ideal setup, 1 = avoid).
Identify the #1 best trade setup if one exists.

Respond with ONLY this JSON:
{{
  "ratings": {{
    "BTCUSDT": <1-10>,
    "ETHUSDT": <1-10>,
    "XRPUSDT": <1-10>,
    "DOGEUSDT": <1-10>,
    "SOLUSDT": <1-10>
  }},
  "rating_reasons": {{
    "BTCUSDT": "<one sentence>",
    "ETHUSDT": "<one sentence>",
    "XRPUSDT": "<one sentence>",
    "DOGEUSDT": "<one sentence>",
    "SOLUSDT": "<one sentence>"
  }},
  "best_setup": "<coin or none>",
  "best_setup_direction": "<long|short|none>",
  "best_setup_reason": "<2-3 sentences on why this is the best setup>",
  "key_levels": {{
    "BTCUSDT": {{"support": <price>, "resistance": <price>}},
    "ETHUSDT": {{"support": <price>, "resistance": <price>}},
    "XRPUSDT": {{"support": <price>, "resistance": <price>}},
    "DOGEUSDT": {{"support": <price>, "resistance": <price>}},
    "SOLUSDT": {{"support": <price>, "resistance": <price>}}
  }},
  "liquidity_targets": "<where smart money will hunt next, 1-2 sentences>",
  "market_structure_note": "<overall structure observation, 1-2 sentences>"
}}"""

        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            ratings = json.loads(raw)
        except Exception as e:
            print(f"[ANALYST AGENT] Claude error: {e}", flush=True)

    result = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "ratings":      ratings.get("ratings", {}),
        "rating_reasons": ratings.get("rating_reasons", {}),
        "best_setup":   ratings.get("best_setup"),
        "best_direction": ratings.get("best_setup_direction"),
        "best_reason":  ratings.get("best_setup_reason"),
        "key_levels":   ratings.get("key_levels", {}),
        "liquidity_targets": ratings.get("liquidity_targets"),
        "structure_note": ratings.get("market_structure_note"),
        "coin_details": {s: {k: v for k, v in d.items()
                             if k not in ("candles_brief_1h", "candles_brief_4h")}
                         for s, d in coin_details.items()},
    }

    # Save to shared state
    set_state("analyst_ratings", result)
    add_report("analyst", "coin_ratings", result)
    add_knowledge("setup_ratings", {
        "best":    ratings.get("best_setup"),
        "ratings": ratings.get("ratings", {}),
        "note":    ratings.get("market_structure_note"),
    })

    # Post to Render
    post_to_render("/api/agent/insight", {
        "type":       "analyst_ratings",
        "agent":      "analyst",
        "timestamp":  result["timestamp"],
        "ratings":    result["ratings"],
        "best_setup": result["best_setup"],
        "best_direction": result["best_direction"],
        "best_reason": result["best_reason"],
        "key_levels": result["key_levels"],
        "liquidity_targets": result["liquidity_targets"],
    })

    best = result.get("best_setup", "none")
    print(f"[ANALYST AGENT] Done. Best setup: {best} {result.get('best_direction','')} | "
          f"Ratings: {result.get('ratings')}", flush=True)
    return result
