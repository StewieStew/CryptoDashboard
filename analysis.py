"""
Crypto Technical Analysis Engine
Fetches OHLCV from Binance public API and computes all 7 framework sections.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"

# Maps each chart interval to (higher-TF for regime, candle limit for analysis)
INTERVAL_HTF = {
    "15m": ("1h",  400),
    "30m": ("4h",  300),
    "1h":  ("4h",  300),
    "4h":  ("1d",  200),
    "1d":  ("1w",  200),
    "1w":  ("1w",  200),
}


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    try:
        r = requests.get(
            BINANCE_KLINES,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Binance fetch failed [{symbol} {interval}]: {e}")

    cols = ["ts","open","high","low","close","volume",
            "ct","qv","trades","tbb","tbq","ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]


# ─────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()


# ─────────────────────────────────────────────
# SWING DETECTION
# ─────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, n: int = 5):
    """Return (highs, lows) as lists of (timestamp, price)."""
    h = df["high"].values
    l = df["low"].values
    idx = df.index
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if h[i] >= max(h[i - n: i + n + 1]):
            highs.append((idx[i], float(h[i])))
        if l[i] <= min(l[i - n: i + n + 1]):
            lows.append((idx[i], float(l[i])))
    return highs, lows


# ─────────────────────────────────────────────
# STRUCTURE ANALYSIS
# ─────────────────────────────────────────────

def detect_structure(df: pd.DataFrame, swing_highs, swing_lows) -> dict:
    cur = float(df["close"].iloc[-1])

    last_sh  = swing_highs[-1][1] if swing_highs else None
    prev_sh  = swing_highs[-2][1] if len(swing_highs) >= 2 else None
    last_sl  = swing_lows[-1][1]  if swing_lows  else None
    prev_sl  = swing_lows[-2][1]  if len(swing_lows)  >= 2 else None

    bullish_bos = bool(last_sh and cur > last_sh)
    bearish_bos = bool(last_sl and cur < last_sl)

    hh_hl = bool(
        last_sh and prev_sh and last_sl and prev_sl
        and last_sh > prev_sh and last_sl > prev_sl
    )
    lh_ll = bool(
        last_sh and prev_sh and last_sl and prev_sl
        and last_sh < prev_sh and last_sl < prev_sl
    )

    return dict(
        bullish_bos=bullish_bos, bearish_bos=bearish_bos,
        hh_hl=hh_hl, lh_ll=lh_ll,
        last_swing_high=last_sh, prev_swing_high=prev_sh,
        last_swing_low=last_sl,  prev_swing_low=prev_sl,
    )


def find_level_touches(df: pd.DataFrame, price: float, side: str = "low",
                       tolerance: float = 0.005, max_results: int = 5) -> list:
    """Return unix timestamps of 4H candles that came within tolerance% of price on the given side."""
    col = "low" if side == "low" else "high"
    hits = df[np.abs(df[col] - price) / price <= tolerance]
    return [int(ts.timestamp()) for ts in hits.index[-max_results:]]


def detect_sweeps(df: pd.DataFrame, swing_highs, swing_lows, lookback: int = 12) -> list:
    recent = df.iloc[-lookback:]
    sweeps = []
    for ts, sh in (swing_highs[-4:] or []):
        wicked = recent[recent["high"] > sh]
        if not wicked.empty:
            bar = wicked.iloc[-1]
            if bar["close"] < sh:
                sweeps.append({"type": "bearish_sweep", "level": round(sh, 2),
                                "ts": str(bar.name), "desc": "Wick above then rejected down"})
    for ts, sl in (swing_lows[-4:] or []):
        wicked = recent[recent["low"] < sl]
        if not wicked.empty:
            bar = wicked.iloc[-1]
            if bar["close"] > sl:
                sweeps.append({"type": "bullish_sweep", "level": round(sl, 2),
                                "ts": str(bar.name), "desc": "Wick below then rejected up"})
    return sweeps


# ─────────────────────────────────────────────
# FIBONACCI
# ─────────────────────────────────────────────

def fibonacci_levels(low: float, high: float) -> dict:
    d = high - low
    return {
        "0.0":   round(low, 2),
        "0.236": round(high - 0.236 * d, 2),
        "0.382": round(high - 0.382 * d, 2),
        "0.5":   round(high - 0.5   * d, 2),
        "0.618": round(high - 0.618 * d, 2),
        "0.786": round(high - 0.786 * d, 2),
        "1.0":   round(high, 2),
    }


def fib_analysis(df: pd.DataFrame, swing_highs, swing_lows) -> dict | None:
    if not swing_highs or not swing_lows:
        return None
    last_sh = swing_highs[-1][1]
    last_sl = swing_lows[-1][1]
    levels  = fibonacci_levels(last_sl, last_sh)
    cur     = float(df["close"].iloc[-1])

    nearest_k, nearest_v, nearest_d = None, None, float("inf")
    for k, v in levels.items():
        d = abs(cur - v) / cur
        if d < nearest_d:
            nearest_d, nearest_k, nearest_v = d, k, v

    return dict(
        levels=levels,
        swing_low=round(last_sl, 2),
        swing_high=round(last_sh, 2),
        at_fib=nearest_d < 0.015,
        nearest_level=nearest_k,
        nearest_price=round(nearest_v, 2) if nearest_v else None,
        distance_pct=round(nearest_d * 100, 2),
    )


# ─────────────────────────────────────────────
# VOLUME & OBV
# ─────────────────────────────────────────────

def volume_analysis(df: pd.DataFrame) -> dict:
    vol   = df["volume"]
    avg20 = vol.iloc[-20:].mean()
    avg5  = vol.iloc[-5:].mean()
    expanding = bool(avg5 > vol.iloc[-20:-5].mean())
    recent_spike = bool((vol.iloc[-10:] > 2 * avg20).any())

    obv_s = obv(df)
    obv_up = bool(obv_s.iloc[-1] > obv_s.iloc[-20])

    ph = df["close"].iloc[-1] > df["close"].iloc[-20:-1].max()
    pl = df["close"].iloc[-1] < df["close"].iloc[-20:-1].min()
    oh = obv_s.iloc[-1] > obv_s.iloc[-20:-1].max()
    ol = obv_s.iloc[-1] < obv_s.iloc[-20:-1].min()

    return dict(
        expanding=expanding,
        recent_spike=recent_spike,
        obv_bullish=obv_up,
        bearish_obv_div=bool(ph and not oh),
        bullish_obv_div=bool(pl and not ol),
        current=round(float(vol.iloc[-1]), 0),
        avg20=round(float(avg20), 0),
    )


# ─────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────

def rsi_analysis(df: pd.DataFrame) -> dict:
    rsi_s = rsi(df["close"])
    cur   = float(rsi_s.iloc[-1])
    rng   = "Bullish" if cur > 60 else "Bearish" if cur < 40 else "Neutral"

    ph = df["close"].iloc[-1] > df["close"].iloc[-14:-1].max()
    pl = df["close"].iloc[-1] < df["close"].iloc[-14:-1].min()
    rh = rsi_s.iloc[-1] > rsi_s.iloc[-14:-1].max()
    rl = rsi_s.iloc[-1] < rsi_s.iloc[-14:-1].min()

    return dict(
        value=round(cur, 1),
        range=rng,
        bearish_div=bool(ph and not rh),
        bullish_div=bool(pl and not rl),
        reset_overbought=bool((rsi_s.iloc[-8:-1] > 70).any() and cur < 70),
        reset_oversold=bool((rsi_s.iloc[-8:-1] < 30).any()  and cur > 30),
        overbought=cur > 70,
        oversold=cur < 30,
    )


# ─────────────────────────────────────────────
# MARKET REGIME
# ─────────────────────────────────────────────

def market_regime(daily_df: pd.DataFrame) -> dict:
    cur    = float(daily_df["close"].iloc[-1])
    ema200 = float(ema(daily_df["close"], 200).iloc[-1])
    atr_s  = atr(daily_df)
    atr_c  = float(atr_s.iloc[-1])
    atr_a  = float(atr_s.iloc[-20:].mean())
    above  = cur > ema200
    expnd  = atr_c > atr_a

    sh, sl = detect_swings(daily_df, n=5)
    st     = detect_structure(daily_df, sh, sl)

    if   above and st["hh_hl"] and expnd:  regime = "Strong Uptrend"
    elif above and st["hh_hl"]:            regime = "Weak Uptrend"
    elif not above and st["lh_ll"] and expnd: regime = "Strong Downtrend"
    elif not above and st["lh_ll"]:        regime = "Weak Downtrend"
    else:                                  regime = "Range / Compression"

    return dict(
        regime=regime,
        above_200=above,
        ema200=round(ema200, 2),
        atr_expanding=expnd,
        atr=round(atr_c, 2),
        hh_hl=st["hh_hl"],
        lh_ll=st["lh_ll"],
    )


# ─────────────────────────────────────────────
# CONFLUENCE SCORE
# ─────────────────────────────────────────────

def confluence_score(regime, structure, vol, rsi_data, sweeps, fib,
                     interval: str = "4h") -> dict:
    score   = 0
    reasons = []

    # 1. Daily trend aligned (2 pts)
    trending = "Uptrend" in regime["regime"] or "Downtrend" in regime["regime"]
    if trending:
        score += 2
        reasons.append({"pts": 2, "earned": True,
                        "text": f"Daily regime is '{regime['regime']}' — directional trend active"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"Regime is '{regime['regime']}' — no directional trend to align"})

    # 2. Clear BOS (2 pts)
    tf = interval.upper()
    bos = structure and (structure["bullish_bos"] or structure["bearish_bos"])
    if bos:
        score += 2
        d = "Bullish" if structure["bullish_bos"] else "Bearish"
        reasons.append({"pts": 2, "earned": True,
                        "text": f"{d} Break of Structure confirmed on {tf} — price outside last swing"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"No confirmed Break of Structure on {tf} — price inside structure"})

    # 3. Liquidity sweep (2 pts)
    if sweeps:
        score += 2
        s = sweeps[0]
        reasons.append({"pts": 2, "earned": True,
                        "text": f"Liquidity sweep: {s['type']} @ {s['level']} — {s['desc']}"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "No recent liquidity sweep detected on 4H"})

    # 4. Volume expansion or spike (1 pt)
    if vol["expanding"] or vol["recent_spike"]:
        score += 1
        vtxt = "Volume expanding" if vol["expanding"] else "Volume spike (2× avg detected)"
        reasons.append({"pts": 1, "earned": True,
                        "text": f"{vtxt} — real participation behind the move"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "Volume declining, no spike — weak participation on current move"})

    # 5. OBV confirms bias or diverges (1 pt)
    bias_up = "Uptrend" in regime["regime"] or bool(structure and structure.get("bullish_bos"))
    bias_dn = "Downtrend" in regime["regime"] or bool(structure and structure.get("bearish_bos"))
    obv_ok = (
        (bias_up and vol["obv_bullish"]) or
        (bias_dn and not vol["obv_bullish"]) or
        vol["bullish_obv_div"] or vol["bearish_obv_div"]
    )
    if obv_ok:
        score += 1
        if vol["bullish_obv_div"] or vol["bearish_obv_div"]:
            obv_dir = "Bullish" if vol["bullish_obv_div"] else "Bearish"
            obv_txt = f"{obv_dir} OBV divergence — smart money accumulating/distributing"
        elif bias_up:
            obv_txt = "OBV trending up — confirms bullish bias, institutional buying"
        else:
            obv_txt = "OBV trending down — confirms bearish bias, institutional selling"
        reasons.append({"pts": 1, "earned": True, "text": obv_txt})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "OBV not confirming trade direction — volume flow opposing bias"})

    # 6. RSI confirmation (1 pt)
    rsi_ok = (
        (rsi_data["range"] == "Bullish" and "Uptrend"   in regime["regime"]) or
        (rsi_data["range"] == "Bearish" and "Downtrend" in regime["regime"]) or
        rsi_data["reset_oversold"] or rsi_data["reset_overbought"]
    )
    if rsi_ok:
        score += 1
        reasons.append({"pts": 1, "earned": True,
                        "text": f"RSI {rsi_data['value']} — confirms direction or reset from extreme"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"RSI {rsi_data['value']} ({rsi_data['range']}) — no directional confirmation"})

    # 7. Fib + structure confluence (1 pt)
    if fib and fib["at_fib"]:
        score += 1
        reasons.append({"pts": 1, "earned": True,
                        "text": f"Price at Fib {fib['nearest_level']} (${fib['nearest_price']:,.2f}), {fib['distance_pct']}% away — structure + fib confluence"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "Price not at a significant Fibonacci level"})

    if   score >= 7: strength = "High Probability Swing Environment"
    elif score >= 5: strength = "Moderate Setup"
    else:            strength = "Low Quality / Avoid"

    # What would improve it
    missing = [r["text"] for r in reasons if not r["earned"]]
    improve = missing[:3] if missing else ["All confluence factors are satisfied"]

    return dict(score=score, max=10, strength=strength, reasons=reasons, improve=improve)


# ─────────────────────────────────────────────
# RISK CONTEXT
# ─────────────────────────────────────────────

def risk_context(df: pd.DataFrame, structure, swing_highs, swing_lows,
                 interval: str = "4h") -> dict:
    cur     = float(df["close"].iloc[-1])
    atr_val = float(atr(df).iloc[-1])

    if structure and structure["bullish_bos"]:
        # Stop: last swing low below entry (wide structural stop)
        inval  = structure["last_swing_low"] or (cur - 2.0 * atr_val)
        # Target: MUST be above entry — use prev swing high, fallback to +2 ATR
        prev_sh = structure.get("prev_swing_high")
        target  = prev_sh if (prev_sh and prev_sh > cur) else round(cur + 2.0 * atr_val, 6)
        bias    = "Long"

    elif structure and structure["bearish_bos"]:
        # Stop: last swing high above entry (wide structural stop)
        inval  = structure["last_swing_high"] or (cur + 2.0 * atr_val)
        # Target: MUST be below entry — use prev swing low, fallback to -2 ATR
        prev_sl = structure.get("prev_swing_low")
        target  = prev_sl if (prev_sl and prev_sl < cur) else round(cur - 2.0 * atr_val, 6)
        bias    = "Short"

    else:
        inval  = swing_lows[-1][1]  if swing_lows  else cur * 0.95
        target = swing_highs[-1][1] if swing_highs else cur * 1.05
        bias   = "Neutral"

    risk_d   = abs(cur - inval)
    reward_d = abs(target - cur)
    rr       = round(reward_d / risk_d, 2) if risk_d > 0 else 0

    inval_dir  = "below" if bias == "Long" else "above"
    inval_swing = "low" if bias == "Long" else "high"
    invalidation_note = (
        f"{interval.upper()} close {inval_dir} ${inval:,.2f} (last swing {inval_swing})"
    )

    return dict(
        bias=bias,
        current=round(cur, 2),
        invalidation=round(inval, 2),
        target=round(target, 2),
        rr=rr,
        favorable=rr >= 2.0,
        invalidation_note=invalidation_note,
    )


# ─────────────────────────────────────────────
# CHART DATA FOR ANY TIMEFRAME
# ─────────────────────────────────────────────

def chart_for_timeframe(symbol: str, interval: str) -> dict:
    """Return OHLCV + indicators for any timeframe (chart display only)."""
    limit = 300 if interval in ["15m", "30m", "1h"] else 200
    df = fetch_ohlcv(symbol, interval, limit)

    ema50_s  = ema(df["close"], 50)
    ema200_s = ema(df["close"], 200)
    rsi_s    = rsi(df["close"])
    obv_s    = obv(df)
    obv_ema_s = ema(obv_s, 20)

    candles = [{"time": int(r.Index.timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in df.itertuples()]
    volumes = [{"time": int(r.Index.timestamp()), "value": r.volume,
                "color": "#26a69a" if r.close >= r.open else "#ef5350"}
               for r in df.itertuples()]
    ema50_pts  = [{"time": int(ts.timestamp()), "value": round(v, 6)}
                  for ts, v in ema50_s.items()]
    ema200_pts = [{"time": int(ts.timestamp()), "value": round(v, 6)}
                  for ts, v in ema200_s.items()]
    rsi_pts    = [{"time": int(ts.timestamp()), "value": round(v, 2)}
                  for ts, v in rsi_s.items() if not np.isnan(v)]
    obv_pts    = [{"time": int(ts.timestamp()), "value": round(float(v), 2)}
                  for ts, v in obv_s.items() if not np.isnan(v)]
    obv_ema_pts = [{"time": int(ts.timestamp()), "value": round(float(v), 2)}
                   for ts, v in obv_ema_s.items() if not np.isnan(v)]

    return dict(candles=candles, volume=volumes,
                ema50=ema50_pts, ema200=ema200_pts, rsi=rsi_pts,
                obv=obv_pts, obv_ema=obv_ema_pts)


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signal(confluence: dict, structure, risk: dict, h4_df) -> dict | None:
    """Returns a signal dict if score >= 7 and BOS is confirmed, else None."""
    if confluence["score"] < 7:
        return None
    if not structure:
        return None

    if structure["bullish_bos"]:
        direction = "LONG"
        reason    = "Bullish BOS — price closed above last swing high"
    elif structure["bearish_bos"]:
        direction = "SHORT"
        reason    = "Bearish BOS — price closed below last swing low"
    else:
        return None

    top_reasons = [r["text"] for r in confluence["reasons"] if r["earned"]]

    # Entry zone: if fib confluence, use fib level, else current price
    entry = risk["current"]

    return dict(
        direction=direction,
        score=confluence["score"],
        entry=entry,
        target=risk["target"],
        stop=risk["invalidation"],
        rr=risk["rr"],
        favorable=risk["favorable"],
        reason=reason,
        top_reasons=top_reasons,
        bar_time=int(h4_df.index[-1].timestamp()),
    )


# ─────────────────────────────────────────────
# PRICE PREDICTION
# ─────────────────────────────────────────────

def price_prediction(h4_df: pd.DataFrame, risk: dict, confluence: dict,
                     volatility: dict, regime: dict, swing_highs, swing_lows,
                     interval: str = "4h") -> dict:
    """Near-term (ATR-based) and swing (structure-based) price targets with scenarios."""
    cur         = float(h4_df["close"].iloc[-1])
    h4_atr_val  = float(atr(h4_df).iloc[-1])
    score       = confluence["score"]
    bias        = risk["bias"]

    confidence  = "High" if score >= 7 else "Medium" if score >= 5 else "Low"

    # Near-term: ±1.5×ATR
    near_bull = round(cur + 1.5 * h4_atr_val, 2)
    near_bear = round(cur - 1.5 * h4_atr_val, 2)

    # Extended target: first structural swing beyond the immediate risk target
    extended_target = None
    if bias == "Long":
        for _, p in reversed(swing_highs[:-1]):
            if p > risk["target"]:
                extended_target = round(p, 2)
                break
    elif bias == "Short":
        for _, p in reversed(swing_lows[:-1]):
            if p < risk["target"]:
                extended_target = round(p, 2)
                break

    # Scenarios
    if bias == "Long":
        scenarios = dict(
            bull={"price": risk["target"],
                  "condition": f"Bullish momentum holds — target resistance at ${risk['target']:,.2f}"},
            base={"price": near_bull,
                  "condition": f"Mild continuation over 1-3 bars to ${near_bull:,.2f} (1.5\u00d7ATR)"},
            bear={"price": risk["invalidation"],
                  "condition": f"Reversal — close below ${risk['invalidation']:,.2f} invalidates long"},
        )
    elif bias == "Short":
        scenarios = dict(
            bull={"price": risk["invalidation"],
                  "condition": f"Reversal — close above ${risk['invalidation']:,.2f} invalidates short"},
            base={"price": near_bear,
                  "condition": f"Mild continuation over 1-3 bars to ${near_bear:,.2f} (1.5\u00d7ATR)"},
            bear={"price": risk["target"],
                  "condition": f"Bearish momentum holds — target support at ${risk['target']:,.2f}"},
        )
    else:
        scenarios = dict(
            bull={"price": near_bull,
                  "condition": f"Upside breakout — ATR-based move toward ${near_bull:,.2f}"},
            base={"price": cur,
                  "condition": "Range-bound — no clear directional bias, wait for structure"},
            bear={"price": near_bear,
                  "condition": f"Downside breakdown — ATR-based move toward ${near_bear:,.2f}"},
        )

    # Narrative
    trend_dir = "above" if regime["above_200"] else "below"
    atr_state = "expanding" if volatility["expanding"] else "compressing"
    near_target = near_bull if bias == "Long" else near_bear if bias == "Short" else None

    tf = interval.upper()
    parts = [
        f"Price is trading {trend_dir} the {risk.get('htf', 'daily').upper()} 200 EMA in a '{regime['regime']}' environment.",
        f"{tf} ATR is {atr_state} at ${h4_atr_val:,.2f}.",
    ]
    if bias in ("Long", "Short"):
        dir_word = "resistance" if bias == "Long" else "support"
        parts.append(
            f"Near-term target: ${near_target:,.2f} (1.5\u00d7ATR from entry)."
        )
        parts.append(
            f"Swing target: ${risk['target']:,.2f} — next structural {dir_word}."
        )
        if extended_target:
            parts.append(
                f"Extended target: ${extended_target:,.2f} if momentum continues beyond the swing target."
            )
        inval_dir = "below" if bias == "Long" else "above"
        parts.append(
            f"Setup invalidated on a {tf} close {inval_dir} ${risk['invalidation']:,.2f}."
        )
    else:
        parts.append(
            f"No directional bias. Expected range: ${near_bear:,.2f}\u2013${near_bull:,.2f} (\u00b11.5\u00d7ATR). "
            "Wait for structure to resolve before entering."
        )

    return dict(
        bias=bias,
        confidence=confidence,
        near_term=dict(
            bull_target=near_bull,
            bear_target=near_bear,
            timeframe="1-3 four-hour bars",
            atr=round(h4_atr_val, 2),
            method="1.5 \u00d7 4H ATR from current price",
        ),
        swing=dict(
            target=risk["target"],
            invalidation=risk["invalidation"],
            extended_target=extended_target,
        ),
        scenarios=scenarios,
        narrative=" ".join(parts),
    )


# ─────────────────────────────────────────────
# CHANNEL ANALYSIS
# ─────────────────────────────────────────────

def channel_analysis(df: pd.DataFrame, swing_highs, swing_lows) -> dict | None:
    """Detect ascending / descending / horizontal channel using linear regression."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    n = min(4, len(swing_highs), len(swing_lows))

    h_ts = np.array([float(ts.timestamp()) for ts, _ in swing_highs[-n:]])
    h_pr = np.array([float(p) for _, p in swing_highs[-n:]])
    l_ts = np.array([float(ts.timestamp()) for ts, _ in swing_lows[-n:]])
    l_pr = np.array([float(p) for _, p in swing_lows[-n:]])

    # Normalise timestamps to avoid floating-point blow-up in polyfit
    t0 = min(h_ts[0], l_ts[0])
    m_h, b_h = np.polyfit(h_ts - t0, h_pr, 1)
    m_l, b_l = np.polyfit(l_ts - t0, l_pr, 1)

    t_end = float(df.index[-1].timestamp())

    def eval_line(m, b, t): return round(m * (t - t0) + b, 6)

    upper_line = [
        {"time": int(h_ts[0]),  "value": eval_line(m_h, b_h, h_ts[0])},
        {"time": int(t_end),    "value": eval_line(m_h, b_h, t_end)},
    ]
    lower_line = [
        {"time": int(l_ts[0]),  "value": eval_line(m_l, b_l, l_ts[0])},
        {"time": int(t_end),    "value": eval_line(m_l, b_l, t_end)},
    ]

    # Slope as % per bar (normalised to current price)
    mid = float(df["close"].iloc[-1])
    bar_s = max((df.index[-1] - df.index[-2]).total_seconds(), 1)
    sh_pct = m_h * bar_s / mid * 100
    sl_pct = m_l * bar_s / mid * 100
    THRESH = 0.025   # % per bar to be considered directional

    if abs(sh_pct) < THRESH and abs(sl_pct) < THRESH:
        ch_type, bias = "Horizontal Range", "Neutral"
        note = "Price consolidating between flat support and resistance. Expect breakout; direction TBD."
    elif sh_pct > THRESH and sl_pct > THRESH:
        ch_type, bias = "Ascending Channel", "Bullish"
        note = "Higher highs and higher lows forming a rising channel. Look for longs near the lower boundary."
    elif sh_pct < -THRESH and sl_pct < -THRESH:
        ch_type, bias = "Descending Channel", "Bearish"
        note = "Lower highs and lower lows forming a falling channel. Look for shorts near the upper boundary."
    else:
        ch_type, bias = "Mixed Structure", "Neutral"
        note = "Upper and lower trendlines diverge — no clean channel. Rely on horizontal S/R levels instead."

    return dict(
        channel_type=ch_type,
        bias=bias,
        note=note,
        upper_line=upper_line,
        lower_line=lower_line,
        slope_h_pct=round(sh_pct, 4),
        slope_l_pct=round(sl_pct, 4),
        n_points=n,
    )


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def full_analysis(symbol: str, interval: str = "4h") -> dict:
    # Resolve timeframes
    htf, candle_limit = INTERVAL_HTF.get(interval, ("1d", 200))

    # Fetch data — HTF for regime context, selected TF for everything else
    htf_df = fetch_ohlcv(symbol, htf, 300)
    df     = fetch_ohlcv(symbol, interval, candle_limit)

    # Section 1: Regime (always uses HTF for broader market context)
    regime = market_regime(htf_df)

    # Section 2: Structure
    sh, sl = detect_swings(df, n=5)
    structure = detect_structure(df, sh, sl)
    sweeps    = detect_sweeps(df, sh, sl)

    # Section 3: Volume & RSI
    vol      = volume_analysis(df)
    rsi_data = rsi_analysis(df)

    # Section 4: Fibonacci (only in trending regimes)
    fib = fib_analysis(df, sh, sl) if "Trend" in regime["regime"] else None

    # Section 5: Volatility
    df_atr = atr(df)
    volatility = dict(
        current=round(float(df_atr.iloc[-1]), 2),
        avg=round(float(df_atr.iloc[-20:].mean()), 2),
        expanding=bool(df_atr.iloc[-1] > df_atr.iloc[-20:].mean()),
        compressing=bool(df_atr.iloc[-1] < df_atr.iloc[-20:].mean() * 0.7),
    )

    # Section 6: Confluence
    confluence = confluence_score(regime, structure, vol, rsi_data, sweeps, fib, interval)

    # Section 7: Risk
    risk = risk_context(df, structure, sh, sl, interval)

    # Signal (score >= 7 + BOS)
    signal = generate_signal(confluence, structure, risk, df)

    # Price Prediction
    prediction = price_prediction(df, risk, confluence, volatility, regime, sh, sl, interval)

    # Channel detection
    channels = channel_analysis(df, sh, sl)

    # Key levels
    key_support    = [round(sl[1], 2) for sl in sl[-2:]] if len(sl) >= 2 else ([round(sl[-1][1], 2)] if sl else [])
    key_resistance = [round(sh[1], 2) for sh in sh[-2:]] if len(sh) >= 2 else ([round(sh[-1][1], 2)] if sh else [])

    # Level touch timestamps (for multi-circle highlighting)
    support_touches    = {str(p): find_level_touches(df, p, "low")  for p in key_support}
    resistance_touches = {str(p): find_level_touches(df, p, "high") for p in key_resistance}

    # Chart data
    ema50_s   = ema(df["close"], 50)
    ema200_s  = ema(df["close"], 200)
    rsi_s     = rsi(df["close"])
    obv_s     = obv(df)
    obv_ema_s = ema(obv_s, 20)

    candles = [{"time": int(r.Index.timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in df.itertuples()]

    volumes = [{"time": int(r.Index.timestamp()), "value": r.volume,
                "color": "#26a69a" if r.close >= r.open else "#ef5350"}
               for r in df.itertuples()]

    ema50_pts   = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in ema50_s.items()]
    ema200_pts  = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in ema200_s.items()]
    rsi_pts     = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in rsi_s.items() if not np.isnan(v)]
    obv_pts     = [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in obv_s.items() if not np.isnan(v)]
    obv_ema_pts = [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in obv_ema_s.items() if not np.isnan(v)]

    return dict(
        symbol=symbol,
        interval=interval,
        htf=htf,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        current_price=risk["current"],
        regime=regime,
        structure=structure,
        swings=dict(
            highs=[(str(ts), round(v, 2)) for ts, v in sh[-5:]],
            lows= [(str(ts), round(v, 2)) for ts, v in sl[-5:]],
        ),
        sweeps=sweeps,
        volume=vol,
        rsi=rsi_data,
        fibonacci=fib,
        volatility=volatility,
        confluence=confluence,
        risk=risk,
        signal=signal,
        prediction=prediction,
        channels=channels,
        key_support=key_support,
        key_resistance=key_resistance,
        chart=dict(
            candles=candles,
            volume=volumes,
            ema50=ema50_pts,
            ema200=ema200_pts,
            rsi=rsi_pts,
            obv=obv_pts,
            obv_ema=obv_ema_pts,
            channels=channels,
            levels=dict(
                support=key_support,
                resistance=key_resistance,
                support_touches=support_touches,
                resistance_touches=resistance_touches,
                fib=fib["levels"] if fib else {},
                invalidation=risk["invalidation"],
                target=risk["target"],
            ),
        ),
    )
