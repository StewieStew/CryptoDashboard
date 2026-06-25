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
                           get_knowledge, post_to_render, get_session)

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
    """Wilder's RSI — matches TradingView and every real charting platform."""
    if len(closes) < period + 2:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Seed with simple average for the first period
    avg_g = float(np.mean(gains[:period]))
    avg_l = float(np.mean(losses[:period]))
    # Wilder's smoothing over remaining bars
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
    return float(100.0 - 100.0 / (1.0 + avg_g / avg_l)) if avg_l > 0 else 100.0


def calc_ema(closes: list, period: int) -> float:
    """True Exponential Moving Average — matches TradingView."""
    if len(closes) < period:
        return float(np.mean(closes)) if closes else 0.0
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))  # seed with SMA
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_atr(candles: list, period: int = 14) -> float:
    """Wilder's ATR — exponential smoothing, not simple average."""
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h  = candles[i]["high"]
        l  = candles[i]["low"]
        pc = candles[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    # Seed with SMA, then Wilder's smoothing
    atr = float(np.mean(trs[:period]))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def find_swing_levels(candles_15m: list, candles_4h: list = None, candles_1d: list = None) -> dict:
    """
    Find structural support/resistance from higher timeframes.
    Uses daily swing highs/lows first (major levels), supplemented by 4h.
    These are the levels a real trader draws on their chart.
    """
    price = candles_15m[-1]["close"] if candles_15m else 0
    if not price:
        return {}

    all_highs = []
    all_lows  = []

    # Daily levels — most significant (last 60 days)
    if candles_1d and len(candles_1d) >= 5:
        for c in candles_1d[-60:]:
            all_highs.append(c["high"])
            all_lows.append(c["low"])

    # 4h levels — intermediate structure (last 90 candles = ~15 days)
    if candles_4h and len(candles_4h) >= 5:
        for c in candles_4h[-90:]:
            all_highs.append(c["high"])
            all_lows.append(c["low"])

    # Fallback: use 4h or 1h data if that's all we have
    if not all_highs:
        for c in (candles_15m or [])[-100:]:
            all_highs.append(c["high"])
            all_lows.append(c["low"])

    # Nearest structural resistance above price
    resistances = sorted(
        list(set(round(h, 8) for h in all_highs if h > price * 1.001)),
        key=lambda x: abs(x - price)
    )[:5]

    # Nearest structural support below price
    supports = sorted(
        list(set(round(l, 8) for l in all_lows if l < price * 0.999)),
        key=lambda x: abs(x - price)
    )[:5]

    return {
        "price":       price,
        "resistance":  resistances[0] if resistances else price * 1.03,
        "support":     supports[0]    if supports    else price * 0.97,
        "resistances": resistances,
        "supports":    supports,
    }


def find_liquidity_clusters(candles_15m: list, price: float,
                            candles_4h: list = None, candles_1d: list = None) -> list:
    """
    Find liquidity pools (equal highs / equal lows) across all timeframes.
    4h and daily clusters are structurally significant — that's where stops actually pool.
    """
    all_highs = []
    all_lows  = []

    # Daily highs/lows carry the most weight — swing traders' stops live here
    if candles_1d:
        for c in candles_1d[-60:]:
            all_highs.append(("1d", c["high"]))
            all_lows.append(("1d", c["low"]))

    # 4h highs/lows — intermediate pools
    if candles_4h:
        for c in candles_4h[-90:]:
            all_highs.append(("4h", c["high"]))
            all_lows.append(("4h", c["low"]))

    # 15m for short-term micro clusters
    for c in (candles_15m or [])[-100:]:
        all_highs.append(("15m", c["high"]))
        all_lows.append(("15m", c["low"]))

    clusters = []
    seen = set()

    for tf, h in all_highs:
        key = round(h / price * 200)
        if key in seen:
            continue
        count = sum(1 for _, hh in all_highs if abs(hh - h) / h < 0.005)
        # Weight: daily clusters need 2 touches, 4h need 3, 15m need 4
        min_touches = 2 if tf == "1d" else 3 if tf == "4h" else 4
        if count >= min_touches and abs(h - price) / price < 0.15:
            seen.add(key)
            clusters.append({"type": "sell_stops", "price": round(h, 8),
                              "strength": count, "tf": tf,
                              "side": "above" if h > price else "below"})

    seen = set()
    for tf, l in all_lows:
        key = round(l / price * 200)
        if key in seen:
            continue
        count = sum(1 for _, ll in all_lows if abs(ll - l) / l < 0.005)
        min_touches = 2 if tf == "1d" else 3 if tf == "4h" else 4
        if count >= min_touches and abs(l - price) / price < 0.15:
            seen.add(key)
            clusters.append({"type": "buy_stops", "price": round(l, 8),
                              "strength": count, "tf": tf,
                              "side": "above" if l > price else "below"})

    return sorted(clusters, key=lambda x: abs(x["price"] - price))[:8]


def render_chart_image(symbol: str, c1h: list, c15m: list,
                       ema20_1h: float, ema50_1h: float,
                       ema20_15m: float, ema50_15m: float,
                       supports: list, resistances: list) -> str | None:
    """
    Render a dark-theme 2-panel candlestick chart (1H top, 15M bottom).
    Draws EMA20/50 and key S/R levels. Returns base64 PNG or None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from io import BytesIO
        import base64

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            facecolor="#0d0d1a",
            gridspec_kw={"height_ratios": [1.5, 1]},
        )
        fig.subplots_adjust(hspace=0.08)

        def _draw(ax, candles, n, tf_label):
            candles = candles[-n:]
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#888899", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#222244")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.12, color="#333355", linewidth=0.5)

            price_min = min(c["low"]  for c in candles)
            price_max = max(c["high"] for c in candles)
            pad = (price_max - price_min) * 0.04
            ax.set_ylim(price_min - pad, price_max + pad)
            ax.set_xlim(-1, len(candles) + 1)

            for i, c in enumerate(candles):
                o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
                color = "#26a69a" if cl >= o else "#ef5350"
                ax.add_patch(Rectangle(
                    (i - 0.38, min(o, cl)), 0.76, max(abs(cl - o), (price_max - price_min) * 0.002),
                    color=color, zorder=2
                ))
                ax.plot([i, i], [l, h], color=color, linewidth=0.9, zorder=1)

            ax.set_title(
                f"{symbol}  {tf_label}",
                color="#ccccdd", fontsize=9, loc="left", pad=5, fontweight="bold"
            )
            return len(candles)

        n1h  = _draw(ax1, c1h,  48, "1H — last 48 bars (2 days)")
        n15m = _draw(ax2, c15m, 96, "15M — last 96 bars (24 hours)")

        # EMA lines
        for ax, e20, e50 in [(ax1, ema20_1h, ema50_1h), (ax2, ema20_15m, ema50_15m)]:
            ax.axhline(e20, color="#00e5ff", linewidth=1.1, alpha=0.85, label=f"EMA20  {e20:.4f}")
            ax.axhline(e50, color="#ff9800", linewidth=1.1, alpha=0.85, label=f"EMA50  {e50:.4f}")
            ax.legend(loc="upper left", fontsize=6.5, facecolor="#0d0d1a",
                      labelcolor="white", framealpha=0.6, edgecolor="#333355")

        # S/R lines on both panels
        for ax in (ax1, ax2):
            for r in resistances[:4]:
                ax.axhline(r, color="#ef5350", linewidth=0.75, linestyle="--", alpha=0.65, zorder=0)
            for s in supports[:4]:
                ax.axhline(s, color="#26a69a", linewidth=0.75, linestyle="--", alpha=0.65, zorder=0)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0d0d1a")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print(f"  [CHART] {symbol} render failed: {e}", flush=True)
        return None


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
    session = get_session()
    print(f"  Session: {session['session']}  ({session['quality']} quality)  "
          f"{'⚠ CAUTION' if session['caution'] else '✓ active'}", flush=True)
    if session["caution"]:
        print(f"  ⚠  {session['note']}", flush=True)
    print("  Scanning charts...", flush=True)

    macro        = get_state("macro_regime", {})
    macro_regime = macro.get("regime_type", "uncertain")
    coin_bias    = macro.get("coin_bias", {})
    btc_onchain  = macro.get("btc_onchain", {})
    defi_tvl     = macro.get("defi_tvl", {})
    fear_greed   = macro.get("fear_greed", {}).get("current", {})
    whales       = macro.get("whales", [])

    # No blocking rules — learning is observation only, never restriction

    # Gather full market data for all coins
    coin_blocks  = []
    coin_data    = {}
    chart_images = {}  # sym -> base64 PNG

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

        levels   = find_swing_levels(c15m if c15m else c1h, c4h, c1d)
        clusters = find_liquidity_clusters(c15m if c15m else c1h, price, c4h, c1d)

        # Order book
        bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:30]]
        asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:30]]
        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        ob_ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
        ob_pressure = "BUY" if ob_ratio > 1.3 else "SELL" if ob_ratio < 0.7 else "NEUTRAL"

        # Trend: price vs real EMAs (Wilder exponential, matches TradingView)
        ema20_15m = calc_ema(closes15m, 20) if len(closes15m) >= 20 else price
        ema50_15m = calc_ema(closes15m, 50) if len(closes15m) >= 50 else price
        ema20_1h  = calc_ema(closes,    20) if len(closes)    >= 20 else price
        ema50_1h  = calc_ema(closes,    50) if len(closes)    >= 50 else price
        ema20_4h  = calc_ema(closes4h,  20) if len(closes4h)  >= 20 else price
        ema50_4h  = calc_ema(closes4h,  50) if len(closes4h)  >= 50 else price
        ema20_1d  = calc_ema(closes1d,  20) if len(closes1d)  >= 20 else price
        ema50_1d  = calc_ema(closes1d,  50) if len(closes1d)  >= 50 else price

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
            "ob_pressure": ob_pressure,
            "ema20_15m": ema20_15m, "ema50_15m": ema50_15m,
            "ema20_1h": ema20_1h,   "ema50_1h": ema50_1h,
            "ema20_4h": ema20_4h,   "ema50_4h": ema50_4h,
            "ema20_1d": ema20_1d,   "ema50_1d": ema50_1d,
            "vol_trend": vol_trend, "bias": coin_bias.get(coin, "neutral"),
        }

        # Print what we found for this coin
        def _pos(p, e): return "▲" if p > e else "▼"
        ema_str = (f"15m {_pos(price,ema20_15m)}20/{_pos(price,ema50_15m)}50  "
                   f"1h {_pos(price,ema20_1h)}20/{_pos(price,ema50_1h)}50  "
                   f"4h {_pos(price,ema20_4h)}20/{_pos(price,ema50_4h)}50  "
                   f"1d {_pos(price,ema20_1d)}20/{_pos(price,ema50_1d)}50")
        ob_str  = f"OB {ob_pressure} ({ob_ratio:.2f})"
        macro_b = coin_bias.get(coin, "neutral").upper()
        print(f"  {coin:4s}  ${price:<12,.4f}  RSI 15m={rsi_15m:.0f}  1h={rsi_1h:.0f}  4h={rsi_4h:.0f}  1d={rsi_1d:.0f}  |  Vol: {vol_trend}  |  {ob_str}  |  Macro: {macro_b}", flush=True)
        print(f"       EMA: {ema_str}", flush=True)
        sups = levels.get("supports", [])
        ress = levels.get("resistances", [])
        sup_disp = "  ".join(f"${s:.4f}" for s in sups[:3])
        res_disp = "  ".join(f"${r:.4f}" for r in ress[:3])
        print(f"       Sup(1D/4H): {sup_disp or 'none'}  |  Res(1D/4H): {res_disp or 'none'}  |  ATR(15m): {atr_15m:.4f}", flush=True)

        liq_str = "\n".join(
            f"  {c['type']} @ {c['price']:.6f} (x{c['strength']}, {c['side']})"
            for c in clusters[:3]
        ) or "  None clear"

        whale_for_coin = [w for w in whales if w.get("symbol") == coin]
        whale_str = "\n".join(
            f"  {'🔴' if w['signal']=='BEARISH_INFLOW' else '🟢'} ${w['amount_usd']}M {w['from']}→{w['to']} ({w['signal']})"
            for w in whale_for_coin[:2]
        ) or "  No whale activity"

        # Build multi-timeframe level stack for Claude
        res_stack = "  ".join(f"${r:.4f}" for r in levels.get("resistances", [])[:4])
        sup_stack = "  ".join(f"${s:.4f}" for s in levels.get("supports",    [])[:4])
        liq_str2  = "\n".join(
            f"  [{c['tf']}] {c['type']} @ ${c['price']:.4f} (x{c['strength']}, {c['side']})"
            for c in clusters[:6]
        ) or "  None"

        # Generate chart image so Claude can visually read the candles
        chart_b64 = render_chart_image(
            sym, c1h, c15m,
            ema20_1h, ema50_1h, ema20_15m, ema50_15m,
            levels.get("supports", []), levels.get("resistances", [])
        )
        if chart_b64:
            chart_images[sym] = chart_b64
            print(f"       Chart: ✓ rendered", flush=True)
        else:
            print(f"       Chart: ✗ failed (text-only)", flush=True)

        coin_blocks.append(f"""
══ {sym} ══════════════════════════════════════
Price: ${price:.6f}  |  1h: {chg_1h:+.2f}%  |  24h: {chg_24h:+.2f}%  |  7d: {chg_7d:+.2f}%
RSI:     15m={rsi_15m:.0f}   1h={rsi_1h:.0f}   4h={rsi_4h:.0f}   1d={rsi_1d:.0f}
EMA(20): 15m=${ema20_15m:.4f}  1h=${ema20_1h:.4f}  4h=${ema20_4h:.4f}  1d=${ema20_1d:.4f}
EMA(50): 15m=${ema50_15m:.4f}  1h=${ema50_1h:.4f}  4h=${ema50_4h:.4f}  1d=${ema50_1d:.4f}
Price vs EMA: 15m={'▲' if price>ema20_15m else '▼'}20/{'▲' if price>ema50_15m else '▼'}50  |  1h={'▲' if price>ema20_1h else '▼'}20/{'▲' if price>ema50_1h else '▼'}50  |  4h={'▲' if price>ema20_4h else '▼'}20/{'▲' if price>ema50_4h else '▼'}50  |  1d={'▲' if price>ema20_1d else '▼'}20/{'▲' if price>ema50_1d else '▼'}50
Volume: {vol_trend}  |  Order book: {ob_pressure} ({ob_ratio:.2f})  |  Macro bias: {coin_bias.get(coin,'neutral').upper()}
ATR(15m): {atr_15m:.6f}  |  ATR(1h): {atr_1h:.6f}

KEY LEVELS (from 1D + 4H structure):
  Resistances above: {res_stack or 'none found'}
  Supports below:    {sup_stack or 'none found'}

Liquidity pools (stop clusters across timeframes):
{liq_str2}

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

TRADING SESSION (critical for setup quality):
Current session: {session['session']}  |  Quality: {session['quality']}  |  UTC hour: {session['hour_utc']}:00
{session['note']}
{"⚠ CAUTION SESSION: Only take setups with exceptional clarity. Widen SL slightly — this is a low-liquidity window where stop hunts are common." if session['caution'] else "✓ Active session: volume and liquidity support directional moves."}

BOT STATUS:
{trade_ctx}
{open_str}

WHAT THE BOT HAS LEARNED FROM PAST TRADES (apply these now):
{pm_str}
{f"IMPROVEMENTS SUGGESTED:{chr(10)}{improve_str}" if improve_str else ""}

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
- R:R >= 1.5:1 on 15M trades. If levels are too tight and RR is under 1.3, skip and find a better coin. For 1H setups, aim for 1.8:1+.

STEP 6 - CROSS-CHECK: DOES THE CHART MATCH THE NUMBERS? (CRITICAL)
Before finalizing any setup, look back at the chart image and verify:
- Does the chart VISUALLY confirm what the numbers suggest? If the numbers say RSI=28 (oversold) but the chart shows price still in a clean downtrend with no base, do NOT long — the numbers are early, the chart is truth.
- Is the entry level visible on the chart as a real level? If the suggested entry is in the middle of open air with no wicks or structure near it, reject it.
- Is the TP blocked by a visible level on the chart before it gets there? A daily resistance drawn on the chart between entry and TP kills the trade — adjust TP to just below that level.
- Does the candle pattern visually match the direction? If you're calling SHORT but the last 4 candles on the 15M chart are clear green momentum candles with no wick rejection, that's a mismatch — pass on it.
- Does volume (bar height at bottom of candles) support the move? Visually expanding volume on the setup candle = conviction. Dying volume = no edge.
- Flag any coin where the chart and numbers are telling different stories — note it in market_summary so it's visible.

STEP 7 - RETURN YOUR BEST 1-3 SETUPS
Return only setups you would actually take. Quality over quantity.
- If 3 genuine setups exist across different coins -> return all 3
- If only 1 good setup exists -> return 1 (do not manufacture bad trades)
- If nothing is clean -> return empty trades array. Never force a bad trade.
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
  "next_check_focus": "<specific 15M level or pattern to watch next cycle>",
  "chart_warnings": ["<any coin where chart and numbers conflict, e.g. 'ETH: RSI oversold but chart shows no base, skip longs'>"]
}}"""

        try:
            # Build message content: chart images first, then the text prompt
            # Claude sees the charts visually AND reads the raw numbers
            content = []
            for sym in COINS:
                img = chart_images.get(sym)
                if img:
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img}
                    })
                    content.append({
                        "type": "text",
                        "text": (f"↑ {sym} chart — Top panel: 1H (48 bars = 2 days). "
                                 f"Bottom panel: 15M (96 bars = 24h). "
                                 f"Cyan line = EMA20. Orange line = EMA50. "
                                 f"Red dashed = resistance levels (from 1D+4H). "
                                 f"Green dashed = support levels (from 1D+4H).")
                    })
            content.append({"type": "text", "text": prompt})

            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": content}],
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
            print(f"  Claude error: {e}", flush=True)

    # Handle both old single-trade and new multi-trade response formats
    trades = result.get("trades", [])
    if not trades and result.get("best_trade"):
        trades = [result["best_trade"]]  # backward compat

    # Hard filter: drop any trade that doesn't meet minimum R:R
    before = len(trades)
    trades = [t for t in trades if float(t.get("rr_ratio", 0)) >= MIN_RR]
    if len(trades) < before:
        print(f"  [FILTER] Dropped {before - len(trades)} trade(s) below R:R {MIN_RR}:1", flush=True)

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

    # Print any chart/number conflicts Claude flagged
    warnings = result.get("chart_warnings", [])
    if warnings:
        print(f"", flush=True)
        print(f"  ── CHART WARNINGS ⚠️  ───────────────────────────────────", flush=True)
        for w in warnings:
            print(f"  ⚠  {w}", flush=True)

    summary = result.get("market_summary", "")
    if summary:
        print(f"", flush=True)
        print(f"  ── MARKET SUMMARY ─────────────────────────────────────", flush=True)
        print(f"  {summary}", flush=True)
        nxt = result.get("next_check_focus", "")
        if nxt:
            print(f"  Next watch: {nxt}", flush=True)

    if trades:
        print(f"", flush=True)
        print(f"  ── SETUPS FOUND ({len(trades)}) ─────────────────────────────────", flush=True)
        for i, t in enumerate(trades, 1):
            sym   = t.get('symbol','').replace('USDT','')
            dirn  = t.get('direction','')
            tf    = t.get('timeframe','')
            rr    = t.get('rr_ratio', 0)
            conf  = t.get('confidence', 0)
            qual  = t.get('setup_quality','')
            entry = t.get('entry', 0)
            tp    = t.get('tp', 0)
            sl    = t.get('sl', 0)
            arrow = "SHORT ↓" if dirn == "SHORT" else "LONG ↑"
            stars = "★★★" if qual == "strong" else "★★☆" if qual == "moderate" else "★☆☆"
            print(f"  [{i}] {sym} {arrow} ({tf})  —  R:R {rr:.1f}:1  —  Confidence {conf}/10  {stars}", flush=True)
            print(f"      Entry (limit): ${entry:,.4f}  →  fires when price touches this level", flush=True)
            print(f"      TP: ${tp:,.4f}  |  SL: ${sl:,.4f}", flush=True)
            if t.get('reason'):
                print(f"      Why: {t.get('reason','')[:150]}", flush=True)
            if t.get('risk_note'):
                print(f"      Invalidated if: {t.get('risk_note','')[:100]}", flush=True)
            factors = t.get('confluence_factors', [])
            if factors:
                print(f"      Confluence: {' · '.join(factors[:4])}", flush=True)
    else:
        print(f"  No clean setups found this cycle — waiting for better levels.", flush=True)

    return output
