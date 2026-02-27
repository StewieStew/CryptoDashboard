"""
Crypto Technical Analysis Engine
Fetches OHLCV from Binance public API and computes all 7 framework sections.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

BINANCE_KLINES = "https://api.binance.us/api/v3/klines"


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

def confluence_score(regime, structure, vol, rsi_data, sweeps, fib) -> dict:
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

    # 2. Clear BOS on 4H (2 pts)
    bos = structure and (structure["bullish_bos"] or structure["bearish_bos"])
    if bos:
        score += 2
        d = "Bullish" if structure["bullish_bos"] else "Bearish"
        reasons.append({"pts": 2, "earned": True,
                        "text": f"{d} Break of Structure confirmed on 4H — price outside last swing"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "No confirmed Break of Structure on 4H — price inside structure"})

    # 3. Liquidity sweep (2 pts)
    if sweeps:
        score += 2
        s = sweeps[0]
        reasons.append({"pts": 2, "earned": True,
                        "text": f"Liquidity sweep: {s['type']} @ {s['level']} — {s['desc']}"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "No recent liquidity sweep detected on 4H"})

    # 4. Volume expansion (1 pt)
    if vol["expanding"]:
        score += 1
        reasons.append({"pts": 1, "earned": True,
                        "text": "Volume expanding on recent moves — conviction present"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "Volume not expanding — weak participation"})

    # 5. OBV divergence (1 pt)
    if vol["bullish_obv_div"] or vol["bearish_obv_div"]:
        score += 1
        d = "Bullish" if vol["bullish_obv_div"] else "Bearish"
        reasons.append({"pts": 1, "earned": True,
                        "text": f"{d} OBV divergence detected — volume not confirming price"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "No OBV divergence — price and volume in agreement"})

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

def risk_context(df: pd.DataFrame, structure, swing_highs, swing_lows) -> dict:
    cur = float(df["close"].iloc[-1])

    if structure and structure["bullish_bos"]:
        inval  = structure["last_swing_low"]  or cur * 0.95
        target = structure["last_swing_high"] or cur * 1.10
        bias   = "Long"
    elif structure and structure["bearish_bos"]:
        inval  = structure["last_swing_high"] or cur * 1.05
        target = structure["last_swing_low"]  or cur * 0.90
        bias   = "Short"
    else:
        inval  = swing_lows[-1][1]  if swing_lows  else cur * 0.95
        target = swing_highs[-1][1] if swing_highs else cur * 1.05
        bias   = "Neutral"

    risk_d   = abs(cur - inval)
    reward_d = abs(target - cur)
    rr       = round(reward_d / risk_d, 2) if risk_d > 0 else 0

    invalidation_note = (
        f"4H close {'below' if bias == 'Long' else 'above'} ${inval:,.2f} (last swing {'low' if bias == 'Long' else 'high'})"
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
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def full_analysis(symbol: str) -> dict:
    # Fetch data
    daily = fetch_ohlcv(symbol, "1d", 300)
    h4    = fetch_ohlcv(symbol, "4h", 200)

    # Section 1: Regime
    regime   = market_regime(daily)

    # Section 2: Structure
    sh4, sl4 = detect_swings(h4, n=5)
    structure = detect_structure(h4, sh4, sl4)
    sweeps    = detect_sweeps(h4, sh4, sl4)

    # Section 3: Volume & RSI
    vol      = volume_analysis(h4)
    rsi_data = rsi_analysis(h4)

    # Section 4: Fibonacci (only in trending regimes)
    fib = fib_analysis(h4, sh4, sl4) if "Trend" in regime["regime"] else None

    # Section 5: Volatility
    h4_atr = atr(h4)
    volatility = dict(
        current=round(float(h4_atr.iloc[-1]), 2),
        avg=round(float(h4_atr.iloc[-20:].mean()), 2),
        expanding=bool(h4_atr.iloc[-1] > h4_atr.iloc[-20:].mean()),
        compressing=bool(h4_atr.iloc[-1] < h4_atr.iloc[-20:].mean() * 0.7),
    )

    # Section 6: Confluence
    confluence = confluence_score(regime, structure, vol, rsi_data, sweeps, fib)

    # Section 7: Risk
    risk = risk_context(h4, structure, sh4, sl4)

    # Signal (score >= 7 + BOS)
    signal = generate_signal(confluence, structure, risk, h4)

    # Key levels
    key_support    = [round(sl[1], 2) for sl in sl4[-2:]] if len(sl4) >= 2 else ([round(sl4[-1][1], 2)] if sl4 else [])
    key_resistance = [round(sh[1], 2) for sh in sh4[-2:]] if len(sh4) >= 2 else ([round(sh4[-1][1], 2)] if sh4 else [])

    # Chart data
    ema50_s  = ema(h4["close"], 50)
    ema200_s = ema(h4["close"], 200)
    rsi_s    = rsi(h4["close"])

    candles = [{"time": int(r.Index.timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in h4.itertuples()]

    volumes = [{"time": int(r.Index.timestamp()), "value": r.volume,
                "color": "#26a69a" if r.close >= r.open else "#ef5350"}
               for r in h4.itertuples()]

    ema50_pts  = [{"time": int(ts.timestamp()), "value": round(v, 2)}
                  for ts, v in ema50_s.items()]
    ema200_pts = [{"time": int(ts.timestamp()), "value": round(v, 2)}
                  for ts, v in ema200_s.items()]
    rsi_pts    = [{"time": int(ts.timestamp()), "value": round(v, 2)}
                  for ts, v in rsi_s.items() if not np.isnan(v)]

    return dict(
        symbol=symbol,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        current_price=risk["current"],
        regime=regime,
        structure=structure,
        swings=dict(
            highs=[(str(ts), round(v, 2)) for ts, v in sh4[-5:]],
            lows= [(str(ts), round(v, 2)) for ts, v in sl4[-5:]],
        ),
        sweeps=sweeps,
        volume=vol,
        rsi=rsi_data,
        fibonacci=fib,
        volatility=volatility,
        confluence=confluence,
        risk=risk,
        signal=signal,
        key_support=key_support,
        key_resistance=key_resistance,
        chart=dict(
            candles=candles,
            volume=volumes,
            ema50=ema50_pts,
            ema200=ema200_pts,
            rsi=rsi_pts,
            levels=dict(
                support=key_support,
                resistance=key_resistance,
                fib=fib["levels"] if fib else {},
                invalidation=risk["invalidation"],
                target=risk["target"],
            ),
        ),
    )
