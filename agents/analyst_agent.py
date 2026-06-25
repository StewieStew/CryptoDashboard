"""
ANALYST AGENT — runs every 15 minutes
Job: Read the market like a real trader. Assess confluence across all available
     data. Always produce the single best available trade — even in choppy
     markets, pick the highest-probability setup and note the context.

Philosophy:
  - 40% win rate + 2.5:1 R:R = profitable. We don't need perfect setups.
  - Confluence over perfection: 3+ factors agreeing beats waiting for 10/10.
  - Always trade, always learn, always improve.

Output: best trade signal with full confluence reasoning, key levels, risk notes
"""
from __future__ import annotations
import json, os, time
from datetime import datetime, timezone

import requests
import anthropic
import numpy as np

from agents.state import (set_state, get_state, add_report, add_knowledge,
                           get_knowledge, post_to_render)

COINS         = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT"]
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BINANCE_BASE  = "https://api.binance.us/api/v3"
MIN_RR        = 2.0   # minimum risk:reward — non-negotiable


def _claude():
    if not ANTHROPIC_KEY:
        return None
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def fetch_candles(symbol: str, interval: str, limit: int) -> list:
    try:
        r = requests.get(f"{BINANCE_BASE}/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=10)
        return [{"time": c[0], "open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
                for c in r.json()]
    except Exception:
        return []


def fetch_orderbook(symbol: str) -> dict:
    try:
        r = requests.get(f"{BINANCE_BASE}/depth",
                         params={"symbol": symbol, "limit": 50}, timeout=8)
        return r.json()
    except Exception:
        return {}


def calc_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 2):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = np.mean(gains[-period:])
    avg_l  = np.mean(losses[-period:])
    return float(100.0 - 100.0 / (1.0 + avg_g / avg_l)) if avg_l > 0 else 50.0


def calc_atr(candles: list, period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:]))


def find_swing_levels(candles: list) -> dict:
    """Find recent swing highs and lows — natural support/resistance."""
    if len(candles) < 10:
        return {}
    highs = [c["high"]  for c in candles]
    lows  = [c["low"]   for c in candles]
    price = candles[-1]["close"]

    # Find swing highs above price (resistance)
    resistances = sorted(
        [h for h in highs[-50:] if h > price],
        key=lambda x: abs(x - price)
    )[:3]

    # Find swing lows below price (support)
    supports = sorted(
        [l for l in lows[-50:] if l < price],
        key=lambda x: abs(x - price)
    )[:3]

    return {
        "price":       price,
        "resistance":  resistances[0] if resistances else price * 1.03,
        "support":     supports[0]    if supports    else price * 0.97,
        "resistances": resistances,
        "supports":    supports,
    }


def find_liquidity_clusters(candles: list, price: float) -> list:
    clusters = []
    if len(candles) < 20:
        return clusters
    highs = [c["high"] for c in candles[-100:]]
    lows  = [c["low"]  for c in candles[-100:]]
    seen  = set()
    for h in highs:
        key = round(h / price * 500)
        if key in seen:
            continue
        count = sum(1 for hh in highs if abs(hh - h) / h < 0.005)
        if count >= 2 and abs(h - price) / price < 0.10:
            seen.add(key)
            clusters.append({"type": "sell_stops", "price": round(h, 8),
                              "strength": count, "side": "above" if h > price else "below"})
    seen = set()
    for l in lows:
        key = round(l / price * 500)
        if key in seen:
            continue
        count = sum(1 for ll in lows if abs(ll - l) / l < 0.005)
        if count >= 2 and abs(l - price) / price < 0.10:
            seen.add(key)
            clusters.append({"type": "buy_stops", "price": round(l, 8),
                              "strength": count, "side": "above" if l > price else "below"})
    return sorted(clusters, key=lambda x: abs(x["price"] - price))[:6]


def fmt_candles(candles: list, n: int = 20) -> str:
    lines = []
    for c in candles[-n:]:
        chg  = (c["close"] - c["open"]) / c["open"] * 100
        body = abs(c["close"] - c["open"])
        wick_up   = c["high"]  - max(c["close"], c["open"])
        wick_down = min(c["close"], c["open"]) - c["low"]
        lines.append(
            f"O={c['open']:.4f} H={c['high']:.4f} L={c['low']:.4f} "
            f"C={c['close']:.4f} V={c['volume']:,.0f} ({chg:+.2f}%) "
            f"{'▲' if c['close'] > c['open'] else '▼'}"
        )
    return "\n".join(lines)


def run() -> dict:
    print("[ANALYST AGENT] Running...", flush=True)

    macro        = get_state("macro_regime", {})
    macro_regime = macro.get("regime_type", "uncertain")
    coin_bias    = macro.get("coin_bias", {})
    btc_onchain  = macro.get("btc_onchain", {})
    defi_tvl     = macro.get("defi_tvl", {})
    fear_greed   = macro.get("fear_greed", {}).get("current", {})
    whales       = macro.get("whales", [])

    # Learning agent rules (informational context, not hard blocks)
    strategy_rules = get_state("strategy_rules", [])
    rules_str = ""
    if strategy_rules:
        rules_str = "LEARNED STRATEGY RULES (from past trades — use as context):\n"
        for r in strategy_rules[:8]:
            rules_str += f"  [{r.get('rule_type')}] {r.get('coins',[])} {r.get('direction','')} — {r.get('reason','')[:80]}\n"

    # Gather full market data for all coins
    coin_blocks = []
    coin_data   = {}

    for sym in COINS:
        coin = sym.replace("USDT", "")
        c15m = fetch_candles(sym, "15m", 80)   # ~20 hours of 15m data
        c1h  = fetch_candles(sym, "1h",  100)
        c4h  = fetch_candles(sym, "4h",   50)
        c1d  = fetch_candles(sym, "1d",   30)
        ob   = fetch_orderbook(sym)

        if not c1h:
            time.sleep(0.5)
            continue

        price    = c15m[-1]["close"] if c15m else c1h[-1]["close"]
        closes   = [c["close"] for c in c1h]
        closes15m= [c["close"] for c in c15m]
        closes4h = [c["close"] for c in c4h]
        closes1d = [c["close"] for c in c1d]

        rsi_15m = calc_rsi(closes15m)
        rsi_1h  = calc_rsi(closes)
        rsi_4h  = calc_rsi(closes4h)
        rsi_1d  = calc_rsi(closes1d)
        atr_15m = calc_atr(c15m) if c15m else 0.0
        atr_1h  = calc_atr(c1h)

        chg_1h  = (price - closes[-2])  / closes[-2]  * 100 if len(closes)  >= 2  else 0
        chg_24h = (price - closes[-25]) / closes[-25] * 100 if len(closes)  >= 25 else 0
        chg_7d  = (price - closes1d[-8])/ closes1d[-8]* 100 if len(closes1d)>=  8 else 0

        levels   = find_swing_levels(c15m if c15m else c1h)
        clusters = find_liquidity_clusters(c15m if c15m else c1h, price)

        # Order book
        bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:30]]
        asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:30]]
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        ob_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
        ob_pressure = "BUY" if ob_ratio > 1.3 else "SELL" if ob_ratio < 0.7 else "NEUTRAL"

        # Trend: is price above/below key MAs?
        ema20_15m = float(np.mean(closes15m[-20:])) if len(closes15m) >= 20 else price
        ema50_15m = float(np.mean(closes15m[-50:])) if len(closes15m) >= 50 else price
        ema20_1h  = float(np.mean(closes[-20:]))    if len(closes)    >= 20 else price
        ema50_1h  = float(np.mean(closes[-50:]))    if len(closes)    >= 50 else price
        ema20_4h  = float(np.mean(closes4h[-20:]))  if len(closes4h)  >= 20 else price

        # Volume trend (15m)
        vols15m  = [c["volume"] for c in c15m[-20:]] if c15m else []
        if len(vols15m) >= 6:
            vol_avg15m = float(np.mean(vols15m[:-3]))
            vol_now15m = float(np.mean(vols15m[-3:]))
            vol_trend  = "expanding" if vol_now15m > vol_avg15m * 1.2 else "contracting" if vol_now15m < vol_avg15m * 0.8 else "normal"
        else:
            vols   = [c["volume"] for c in c1h[-20:]]
            vol_avg= float(np.mean(vols[:-3]))
            vol_now= float(np.mean(vols[-3:]))
            vol_trend = "expanding" if vol_now > vol_avg * 1.2 else "contracting" if vol_now < vol_avg * 0.8 else "normal"

        coin_data[sym] = {
            "price": price, "rsi_15m": rsi_15m, "rsi_1h": rsi_1h, "rsi_4h": rsi_4h, "rsi_1d": rsi_1d,
            "atr_15m": atr_15m, "atr_1h": atr_1h, "chg_1h": chg_1h, "chg_24h": chg_24h, "chg_7d": chg_7d,
            "levels": levels, "clusters": clusters, "ob_ratio": ob_ratio,
            "ob_pressure": ob_pressure, "ema20_1h": ema20_1h, "ema50_1h": ema50_1h,
            "ema20_4h": ema20_4h, "vol_trend": vol_trend, "bias": coin_bias.get(coin, "neutral"),
        }

        liq_str = "\n".join(
            f"  {c['type']} @ {c['price']:.6f} (x{c['strength']}, {c['side']})"
            for c in clusters[:3]
        ) or "  None clear"

        whale_for_coin = [w for w in whales if w.get("symbol") == coin]
        whale_str = "\n".join(
            f"  {'🔴' if w['signal']=='BEARISH_INFLOW' else '🟢'} ${w['amount_usd']}M {w['from']}→{w['to']} ({w['signal']})"
            for w in whale_for_coin[:2]
        ) or "  No whale activity"

        coin_blocks.append(f"""
══ {sym} ══════════════════════════════════════
Price: ${price:.6f}  |  1h: {chg_1h:+.2f}%  |  24h: {chg_24h:+.2f}%  |  7d: {chg_7d:+.2f}%
RSI: 15m={rsi_15m:.0f}  1h={rsi_1h:.0f}  4h={rsi_4h:.0f}  1d={rsi_1d:.0f}
Trend (15m): {'ABOVE' if price > ema20_15m else 'BELOW'} 20EMA  |  {'ABOVE' if price > ema50_15m else 'BELOW'} 50EMA
Trend (1h):  {'ABOVE' if price > ema20_1h  else 'BELOW'} 20EMA  |  {'ABOVE' if price > ema50_1h  else 'BELOW'} 50EMA  |  {'ABOVE' if price > ema20_4h else 'BELOW'} 20EMA(4h)
Volume: {vol_trend}  |  Order book pressure: {ob_pressure} (ratio={ob_ratio:.2f})
ATR(15m): {atr_15m:.6f}  |  ATR(1h): {atr_1h:.6f}  |  Macro bias: {coin_bias.get(coin,'neutral').upper()}
Support: ${levels.get('support',0):.6f}  |  Resistance: ${levels.get('resistance',0):.6f}

Liquidity clusters:
{liq_str}

Whale activity:
{whale_str}

15M candles (last 24 = 6 hours):
{fmt_candles(c15m, 24)}

1H candles (last 12):
{fmt_candles(c1h, 12)}

4H candles (last 6):
{fmt_candles(c4h, 6)}

1D candles (last 5):
{fmt_candles(c1d, 5)}
""")
        time.sleep(0.5)

    # Load prior trades for context
    try:
        import requests as _r
        render_url = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")
        trades_r = _r.get(f"{render_url}/api/trades", timeout=10)
        all_trades = trades_r.json() if trades_r.status_code == 200 else []
    except Exception:
        all_trades = []

    closed = [t for t in all_trades if t.get("status") in ("win","loss")]
    open_t = [t for t in all_trades if t.get("status") == "open"]

    trade_ctx = f"Open positions: {len(open_t)} | Closed trades: {len(closed)}"
    if closed:
        wins = sum(1 for t in closed if t.get("status") == "win")
        trade_ctx += f" | Win rate: {wins/len(closed)*100:.0f}%"
        recent = closed[-5:]
        trade_ctx += "\nRecent: " + " ".join(
            f"{'W' if t.get('status')=='win' else 'L'}({t.get('symbol','').replace('USDT','')})"
            for t in recent
        )

    open_str = ""
    if open_t:
        open_str = "CURRENTLY OPEN (do NOT duplicate these):\n"
        for t in open_t:
            open_str += f"  {t.get('symbol')} {t.get('direction')} @ {t.get('entry')} (TP={t.get('tp')} SL={t.get('sl')})\n"

    # Postmortems context — pull both patterns and specific checks
    loss_pms = get_knowledge("loss_patterns", 6)
    win_pms  = get_knowledge("win_patterns",  4)
    all_pms  = loss_pms + win_pms

    loss_str = "\n".join(
        f"  LOSS [{p.get('pattern','')}] {p.get('symbol','')} {p.get('direction','')} — "
        f"lesson: {p.get('lesson','')[:80]} | check: {p.get('check','')[:60]}"
        for p in loss_pms if p.get("pattern")
    ) or "  No losses analyzed yet."

    win_str = "\n".join(
        f"  WIN  [{p.get('pattern','')}] {p.get('symbol','')} {p.get('direction','')} — "
        f"lesson: {p.get('lesson','')[:80]}"
        for p in win_pms if p.get("pattern")
    ) or "  No wins analyzed yet."

    pm_str = f"LOSSES:\n{loss_str}\nWINS:\n{win_str}"

    # Also pull improvement suggestions
    improvements = get_knowledge("improvements", 3)
    improve_str = "\n".join(
        f"  - {imp.get('suggestion','')[:100]}"
        for imp in improvements if imp.get("suggestion")
    ) or ""

    client = _claude()
    result = {}

    if client:
        prompt = f"""You are an experienced CRYPTO trader doing your market read. This is crypto — not stocks. BTC leads everything, alts follow or diverge, and the 24/7 market means momentum can shift fast. You read candle patterns on the actual trading timeframe, project where the NEXT few candles are headed, and set entries at the level that confirms your projection. You never chase. You wait for price to come to you.

MARKET CONTEXT:
Macro regime: {macro_regime} | Fear & Greed: {fear_greed.get('value','?')} ({fear_greed.get('label','?')})
DeFi TVL: ${defi_tvl.get('total_tvl_bn',0):.1f}B ({defi_tvl.get('tvl_change_24h',0):+.1f}% 24h) — {defi_tvl.get('tvl_signal','?')}
BTC active addresses: {btc_onchain.get('active_addresses',0):,} ({btc_onchain.get('active_addr_change',0):+.1f}% vs yesterday)

BOT STATUS:
{trade_ctx}
{open_str}

WHAT THE BOT HAS LEARNED FROM PAST TRADES (apply these now):
{pm_str}
{f"IMPROVEMENTS SUGGESTED:{chr(10)}{improve_str}" if improve_str else ""}
{rules_str}

FULL MARKET DATA:
{''.join(coin_blocks)}

YOUR PROCESS — think like a crypto trader reading live charts:

STEP 1 - READ BTC FIRST (everything in crypto follows BTC)
Start with BTCUSDT 15M candles, then check 1H for confirmation:
- What are the last 24 x 15M candles doing? Trending, coiling, distributing, recovering?
- What do the LAST 4 x 15M candles show right now: momentum building, fading, or reversing?
- Where is BTC relative to its nearest 15M and 1H key levels?
This is your master bias. When BTC 15M is breaking down, alt shorts are favored. When BTC stabilizes and bounces off a level, alt longs become viable. Counter-BTC trades need very strong coin-specific justification.

STEP 2 - FOR EACH ALT: READ THE 15M CANDLES AS A STORY (past 6 hours > now > next 1-2 hours)
The 15M chart is where trades actually form. Read the last 24 x 15M candles as a sequence:
  PAST (candles 1-16, the first ~4 hours): What was happening? Trending, ranging, fake breakout, reversal?
  NOW (candles 17-24, last 2 hours, especially last 4): What is price doing right now? Building toward a level? Rejecting a key zone? Volume picking up or dying?
  NEXT (your projection for the next 4-8 x 15M candles = next 1-2 hours):
    - 15M range tightening (smaller candles, highs/lows converging) -> coiling for breakout. Pick direction based on BTC bias and 1H trend.
    - 15M lower highs on each bounce approaching a resistance -> distribution. Short the next bounce.
    - 15M higher lows with increasing volume at a support -> accumulation. Long on next dip.
    - 15M big move then shrinking candles with volume dying -> exhaustion, likely reversal.
    - 15M strong close through a key level with volume -> continuation, enter on first pullback.
Then cross-check: do the 1H candles (last 12) confirm the 15M story and direction?

STEP 3 - CONFIRM WITH 1H AND 4H
- Does your 15M setup align with the 1H trend direction?
- Are there 4H or 1D levels nearby that would block the projected TP before it reaches the target?
- 15M setups WITH the 1H trend are high probability. Counter-trend 15M trades need exceptional justification.

STEP 4 - SET ENTRY AT A MEANINGFUL 15M LEVEL
The entry must be at a level that means something on the 15M chart:
- Price is AT a key 15M level right now with confirmation (wick rejection, engulfing close, RSI divergence) -> entry = current price
- Price needs to pull back to the 15M level first before the move starts -> set entry = that pullback level and wait
- Price is mid-range on 15M with no nearby structure -> skip this coin
Set timeframe to "15m" when entry and TP/SL are based on 15M structure. Use "1h" when levels are wider 1H structures.

STEP 5 - SET TP AND SL AT STRUCTURAL LEVELS ON THE TRADING TIMEFRAME
- TP: next significant 15M or 1H level that price is projected to reach (prior swing high for SHORT, swing low for LONG)
- SL: just beyond the 15M candle that proves this read was wrong (beyond the wick that invalidates the setup)
- R:R >= 1.8:1. If 15M levels are too tight, check if 1H levels give a viable setup with wider TP.

STEP 6 - RETURN YOUR BEST 1-3 SETUPS
Return only setups you would actually take. Quality over quantity.
- If 3 genuine setups exist across different coins -> return all 3
- If only 1 good setup exists -> return 1 (do not manufacture bad trades)
- ALWAYS return at least 1. Never return empty.
- Do NOT return two trades on the same coin.
- Do NOT suggest a trade if the same coin+direction is already open.

CRYPTO CONTEXT:
- Fear and Greed above 75 = overleveraged long market -> SHORT setups on 15M rejection candles are high probability
- Fear and Greed below 25 = capitulation -> LONG setups if BTC 15M shows a base forming (higher lows)
- Volume expanding on a 15M move = real conviction. Volume shrinking = likely to reverse or stall.
- Rejection wicks on the 15M at a key level with high volume = strong signal that level matters
- Lower highs on 15M = distribution (favor shorts on bounces). Higher lows on 15M = accumulation (favor longs on dips).

Respond with ONLY this JSON:
{{
  "trades": [
    {{
      "symbol": "<XYZUSDT>",
      "direction": "<LONG|SHORT>",
      "timeframe": "<15m|1h|4h>",
      "entry": <entry_price - current price if at the level now, or the 15M level you are waiting for>,
      "tp": <take_profit - next real structural level on the 15M or 1H that price is heading to>,
      "sl": <stop_loss - just beyond the 15M level that invalidates this read>,
      "rr_ratio": <tp_dist / sl_dist>,
      "confidence": <1-10>,
      "confluence_factors": ["<factor1>", "<factor2>", "<factor3>"],
      "reason": "<2-3 sentences: what the 15M candles are building toward, what you project happens in the next 1-2 hours, and why this entry level makes sense>",
      "risk_note": "<what would prove this 15M read wrong>",
      "setup_quality": "<strong|moderate|marginal>"
    }}
  ],
  "market_summary": "<1-2 sentences: BTC 15M structure right now and what it means for alts>",
  "coins_to_avoid": ["<sym>"],
  "next_check_focus": "<specific 15M level or pattern to watch next cycle>"
}}"""

        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                open_b = raw.count("{") - raw.count("}")
                open_a = raw.count("[") - raw.count("]")
                raw += "]" * max(open_a, 0) + "}" * max(open_b, 0)
                try:
                    result = json.loads(raw)
                except Exception:
                    result = {}
        except Exception as e:
            print(f"[ANALYST AGENT] Claude error: {e}", flush=True)

    # Handle both old single-trade and new multi-trade response formats
    trades = result.get("trades", [])
    if not trades and result.get("best_trade"):
        trades = [result["best_trade"]]  # backward compat

    # Use the highest-confidence trade as the primary signal for dashboard display
    trade = max(trades, key=lambda t: t.get("confidence", 0)) if trades else {}

    output = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "trade_signal":    trade,       # primary signal (highest confidence)
        "all_signals":     trades,      # all signals this cycle
        "market_summary":  result.get("market_summary"),
        "coins_to_avoid":  result.get("coins_to_avoid", []),
        "next_focus":      result.get("next_check_focus"),
        "coin_data":       {s: {k: v for k, v in d.items()
                                if k not in ("clusters",)} for s, d in coin_data.items()},
    }

    set_state("analyst_ratings", output)
    add_report("analyst", "trade_signal", output)

    for t in trades:
        add_knowledge("trade_signals", {
            "symbol":     t.get("symbol"),
            "direction":  t.get("direction"),
            "timeframe":  t.get("timeframe"),
            "quality":    t.get("setup_quality"),
            "rr":         t.get("rr_ratio"),
            "confidence": t.get("confidence"),
            "reason":     t.get("reason","")[:100],
        })

    post_to_render("/api/agent/insight", {
        "type":          "analyst_signal",
        "agent":         "analyst",
        "timestamp":     output["timestamp"],
        "trade_signal":  trade,
        "all_signals":   trades,
        "market_summary": result.get("market_summary"),
        "coins_to_avoid": result.get("coins_to_avoid", []),
    })

    print(f"[ANALYST AGENT] Done. {len(trades)} signal(s):", flush=True)
    for t in trades:
        print(f"  {t.get('symbol')} {t.get('direction')} {t.get('timeframe')} | "
              f"R:R {t.get('rr_ratio',0):.1f}:1 | {t.get('setup_quality')} | "
              f"confidence {t.get('confidence')}/10", flush=True)
    return output
