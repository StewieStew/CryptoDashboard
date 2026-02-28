"""
Crypto Technical Analysis Engine
Fetches OHLCV from Binance public API and computes all 7 framework sections.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

try:
    from learning import get_weights, get_threshold, get_stop_multiplier
except ImportError:
    def get_weights():        return {"regime":2.0,"bos":2.0,"sweep":2.0,"volume":1.0,"obv":1.0,"rsi":1.0,"fib":1.0}
    def get_threshold():     return 7.0
    def get_stop_multiplier(): return 0.5

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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_p: int = 9):
    """MACD line, signal line, and histogram as aligned Series."""
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal_p)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def obv_swing_levels(obv_s: pd.Series, n: int = 5) -> tuple:
    """Detect recent OBV swing highs/lows for S/R visualization on the OBV chart."""
    vals  = obv_s.values
    times = obv_s.index
    highs, lows = [], []
    for i in range(n, len(vals) - n):
        w = vals[i - n: i + n + 1]
        if vals[i] >= max(w) - 1e-9:
            highs.append([int(times[i].timestamp()), round(float(vals[i]), 2)])
        if vals[i] <= min(w) + 1e-9:
            lows.append([int(times[i].timestamp()), round(float(vals[i]), 2)])
    return highs[-4:], lows[-4:]


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


def macd_analysis(df: pd.DataFrame) -> dict:
    """MACD state used in confluence scoring."""
    ml, sl_s, hist = macd(df["close"])
    cur_macd  = float(ml.iloc[-1])
    cur_sig   = float(sl_s.iloc[-1])
    cur_hist  = float(hist.iloc[-1])
    prev_hist = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0
    return dict(
        macd=round(cur_macd, 6),
        signal=round(cur_sig, 6),
        hist=round(cur_hist, 6),
        bullish=bool(cur_macd > cur_sig),
        bearish=bool(cur_macd < cur_sig),
        bullish_cross=bool(prev_hist <= 0 and cur_hist > 0),
        bearish_cross=bool(prev_hist >= 0 and cur_hist < 0),
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

def confluence_score(regime, structure, vol, rsi_data, sweeps, macd_data,
                     interval: str = "4h", weights: dict | None = None) -> dict:
    """
    Score = sum of earned factor weights.
    Weights come from the adaptive learning engine (defaults to original point values).
    A factors_snapshot dict is returned for post-trade learning.
    """
    if weights is None:
        weights = get_weights()

    score   = 0.0
    reasons = []
    snap    = {}   # which factors were present (for adaptive learning)
    tf      = interval.upper()

    # 1. Trend regime aligned (default 2 pts)
    w = weights.get("regime", 2.0)
    trending = "Uptrend" in regime["regime"] or "Downtrend" in regime["regime"]
    snap["regime"] = trending
    if trending:
        score += w
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"Daily regime '{regime['regime']}' — directional trend active"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"Regime '{regime['regime']}' — no directional trend"})

    # 2. Break of Structure (default 2 pts)
    w = weights.get("bos", 2.0)
    bos = structure and (structure["bullish_bos"] or structure["bearish_bos"])
    snap["bos"] = bool(bos)
    if bos:
        score += w
        d = "Bullish" if structure["bullish_bos"] else "Bearish"
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"{d} Break of Structure confirmed on {tf} — price outside last swing"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"No BOS on {tf} — price still inside structure"})

    # 3. Liquidity sweep (default 2 pts)
    w = weights.get("sweep", 2.0)
    snap["sweep"] = bool(sweeps)
    if sweeps:
        score += w
        s = sweeps[0]
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"Liquidity sweep: {s['type']} @ {s['level']} — {s['desc']}"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"No recent liquidity sweep on {tf}"})

    # 4. Volume expansion or spike (default 1 pt)
    w = weights.get("volume", 1.0)
    vol_ok = vol["expanding"] or vol["recent_spike"]
    snap["volume"] = vol_ok
    if vol_ok:
        score += w
        vtxt = "Volume expanding" if vol["expanding"] else "Volume spike (2× avg)"
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"{vtxt} — real participation behind the move"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "Volume declining / no spike — weak participation"})

    # 5. OBV confirms bias or diverges (default 1 pt)
    w = weights.get("obv", 1.0)
    bias_up = "Uptrend"   in regime["regime"] or bool(structure and structure.get("bullish_bos"))
    bias_dn = "Downtrend" in regime["regime"] or bool(structure and structure.get("bearish_bos"))
    obv_ok  = (
        (bias_up and vol["obv_bullish"]) or
        (bias_dn and not vol["obv_bullish"]) or
        vol["bullish_obv_div"] or vol["bearish_obv_div"]
    )
    snap["obv"] = obv_ok
    if obv_ok:
        score += w
        if vol["bullish_obv_div"] or vol["bearish_obv_div"]:
            obv_dir = "Bullish" if vol["bullish_obv_div"] else "Bearish"
            obv_txt = f"{obv_dir} OBV divergence — smart money accumulating/distributing"
        elif bias_up:
            obv_txt = "OBV trending up — confirms bullish bias"
        else:
            obv_txt = "OBV trending down — confirms bearish bias"
        reasons.append({"pts": round(w, 1), "earned": True, "text": obv_txt})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": "OBV not confirming direction — volume flow opposing thesis"})

    # 6. RSI confirmation (default 1 pt)
    w = weights.get("rsi", 1.0)
    rsi_ok = (
        (rsi_data["range"] == "Bullish" and "Uptrend"   in regime["regime"]) or
        (rsi_data["range"] == "Bearish" and "Downtrend" in regime["regime"]) or
        rsi_data["reset_oversold"] or rsi_data["reset_overbought"]
    )
    snap["rsi"] = rsi_ok
    if rsi_ok:
        score += w
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"RSI {rsi_data['value']} — confirms direction or reset from extreme"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"RSI {rsi_data['value']} ({rsi_data['range']}) — no directional confirmation"})

    # 7. MACD direction confirmation (weight key kept as "fib" for learning-engine compat)
    w = weights.get("fib", 1.0)
    bias = "long"
    if structure:
        if structure.get("bearish_bos") or structure.get("lh_ll"):
            bias = "short"
    elif not regime.get("above_200", True):
        bias = "short"
    macd_ok = macd_data and (
        (bias == "long"  and macd_data.get("bullish", False)) or
        (bias == "short" and macd_data.get("bearish", False))
    )
    snap["fib"] = bool(macd_ok)
    if macd_ok:
        cross = (" · bullish crossover" if macd_data.get("bullish_cross")
                 else " · bearish crossover" if macd_data.get("bearish_cross")
                 else "")
        score += w
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"MACD confirms {bias.upper()} — line "
                                f"{'above' if bias == 'long' else 'below'} signal{cross}"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"MACD not confirming {bias.upper()} bias — no momentum alignment"})

    score     = round(score, 1)
    max_score = round(sum(weights.values()), 1)

    if   score >= max_score * 0.78: strength = "High Probability Swing Environment"
    elif score >= max_score * 0.56: strength = "Moderate Setup"
    else:                           strength = "Low Quality / Avoid"

    missing = [r["text"] for r in reasons if not r["earned"]]
    improve = missing[:3] if missing else ["All confluence factors are satisfied"]

    return dict(
        score=score, max=max_score,
        strength=strength, reasons=reasons, improve=improve,
        factors_snapshot=snap,
    )


# ─────────────────────────────────────────────
# RISK CONTEXT
# ─────────────────────────────────────────────

def risk_context(df: pd.DataFrame, structure, swing_highs, swing_lows,
                 interval: str = "4h", stop_multiplier: float = 0.5) -> dict:
    cur        = float(df["close"].iloc[-1])
    atr_val    = float(atr(df).iloc[-1])
    ema50_val  = float(ema(df["close"], 50).iloc[-1])
    ema200_val = float(ema(df["close"], 200).iloc[-1])
    tf         = interval.upper()

    if structure and structure["bullish_bos"]:
        # ── Stop: just below the broken resistance (now acting as support)
        #    stop_multiplier×ATR buffer absorbs noise/wicks — tighter than last swing low
        last_sh = structure["last_swing_high"] or cur
        inval   = round(last_sh - stop_multiplier * atr_val, 6)

        # ── Target: next meaningful resistance ABOVE current price
        #    Priority: prior swing high → EMA200 → EMA50 → 2.5×ATR
        prev_sh = structure.get("prev_swing_high")
        if prev_sh and prev_sh > cur:
            target       = prev_sh
            target_basis = f"Prior swing high"
        elif ema200_val > cur:
            target       = round(ema200_val, 6)
            target_basis = "EMA200 dynamic resistance"
        elif ema50_val > cur:
            target       = round(ema50_val, 6)
            target_basis = "EMA50 dynamic resistance"
        else:
            target       = round(cur + 2.5 * atr_val, 6)
            target_basis = "2.5×ATR projection"
        bias       = "Long"
        inval_note = (f"{tf} close below ${inval:,.4f} — reclaiming below broken resistance "
                      f"invalidates the bullish BOS")

    elif structure and structure["bearish_bos"]:
        # ── Stop: just above the broken support (now acting as resistance)
        #    stop_multiplier×ATR buffer absorbs noise/wicks
        last_sl = structure["last_swing_low"] or cur
        inval   = round(last_sl + stop_multiplier * atr_val, 6)

        # ── Target: next meaningful support BELOW current price
        #    Priority: prior swing low → EMA200 → EMA50 → 2.5×ATR
        prev_sl = structure.get("prev_swing_low")
        if prev_sl and prev_sl < cur:
            target       = prev_sl
            target_basis = "Prior swing low"
        elif ema200_val < cur:
            target       = round(ema200_val, 6)
            target_basis = "EMA200 dynamic support"
        elif ema50_val < cur:
            target       = round(ema50_val, 6)
            target_basis = "EMA50 dynamic support"
        else:
            target       = round(cur - 2.5 * atr_val, 6)
            target_basis = "2.5×ATR projection"
        bias       = "Short"
        inval_note = (f"{tf} close above ${inval:,.4f} — reclaiming above broken support "
                      f"invalidates the bearish BOS")

    else:
        inval        = swing_lows[-1][1]  if swing_lows  else cur * 0.95
        target       = swing_highs[-1][1] if swing_highs else cur * 1.05
        bias         = "Neutral"
        target_basis = "Nearest swing high"
        inval_note   = f"Last swing low ${inval:,.4f} structural floor"

    risk_d   = abs(cur - inval)
    reward_d = abs(target - cur)

    # ── Cap stop loss at 25% of entry price ──────────────────────────────────
    MAX_RISK_PCT = 0.25
    if risk_d > 0 and (risk_d / cur) > MAX_RISK_PCT and bias != "Neutral":
        risk_d = cur * MAX_RISK_PCT
        if bias == "Long":
            inval = round(cur - risk_d, 6)
            inval_note = (f"Stop tightened to 25% max risk at ${inval:,.4f} — "
                          f"structural stop exceeded maximum allowed risk")
        else:
            inval = round(cur + risk_d, 6)
            inval_note = (f"Stop tightened to 25% max risk at ${inval:,.4f} — "
                          f"structural stop exceeded maximum allowed risk")
        reward_d = abs(target - cur)

    rr = round(reward_d / risk_d, 2) if risk_d > 0 else 0

    # ── Enforce minimum 2:1 R:R — project target further if structural level is too close ──
    if risk_d > 0 and rr < 2.0 and bias != "Neutral":
        if bias == "Long":
            target = round(cur + 2.0 * risk_d, 6)
        else:
            target = round(cur - 2.0 * risk_d, 6)
        target_basis = "2:1 R:R projection (structural target too close)"
        reward_d = abs(target - cur)
        rr = round(reward_d / risk_d, 2)

    return dict(
        bias=bias,
        current=round(cur, 6),
        invalidation=round(inval, 6),
        target=round(target, 6),
        target_basis=target_basis,
        rr=rr,
        favorable=rr >= 2.0,
        invalidation_note=inval_note,
    )


# ─────────────────────────────────────────────
# CHART DATA FOR ANY TIMEFRAME
# ─────────────────────────────────────────────

def chart_for_timeframe(symbol: str, interval: str) -> dict:
    """Return OHLCV + indicators for any timeframe (chart display only)."""
    limit = 300 if interval in ["15m", "30m", "1h"] else 200
    df = fetch_ohlcv(symbol, interval, limit)

    ema50_s   = ema(df["close"], 50)
    ema200_s  = ema(df["close"], 200)
    rsi_s     = rsi(df["close"])
    obv_s     = obv(df)
    obv_ema_s = ema(obv_s, 20)
    ml, sl_s, hist = macd(df["close"])

    candles = [{"time": int(r.Index.timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in df.itertuples()]
    volumes = [{"time": int(r.Index.timestamp()), "value": r.volume,
                "color": "#26a69a" if r.close >= r.open else "#ef5350"}
               for r in df.itertuples()]
    ema50_pts   = [{"time": int(ts.timestamp()), "value": round(v, 6)}
                   for ts, v in ema50_s.items()]
    ema200_pts  = [{"time": int(ts.timestamp()), "value": round(v, 6)}
                   for ts, v in ema200_s.items()]
    rsi_pts     = [{"time": int(ts.timestamp()), "value": round(v, 2)}
                   for ts, v in rsi_s.items() if not np.isnan(v)]
    obv_pts     = [{"time": int(ts.timestamp()), "value": round(float(v), 2)}
                   for ts, v in obv_s.items() if not np.isnan(v)]
    obv_ema_pts = [{"time": int(ts.timestamp()), "value": round(float(v), 2)}
                   for ts, v in obv_ema_s.items() if not np.isnan(v)]
    macd_pts      = [{"time": int(ts.timestamp()), "value": round(float(v), 6)}
                     for ts, v in ml.items() if not np.isnan(v)]
    macd_sig_pts  = [{"time": int(ts.timestamp()), "value": round(float(v), 6)}
                     for ts, v in sl_s.items() if not np.isnan(v)]
    macd_hist_pts = [{"time": int(ts.timestamp()), "value": round(float(v), 6),
                      "color": "#3fb95088" if v >= 0 else "#f8514988"}
                     for ts, v in hist.items() if not np.isnan(v)]

    obv_sh, obv_sl = obv_swing_levels(obv_s)

    return dict(candles=candles, volume=volumes,
                ema50=ema50_pts, ema200=ema200_pts, rsi=rsi_pts,
                obv=obv_pts, obv_ema=obv_ema_pts,
                macd=macd_pts, macd_signal=macd_sig_pts, macd_hist=macd_hist_pts,
                obv_highs=obv_sh, obv_lows=obv_sl)


# ─────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signal(confluence: dict, structure, risk: dict, h4_df,
                    signal_threshold: float | None = None) -> dict | None:
    """Returns a signal dict if effective score >= adaptive threshold, BOS confirmed, and R:R >= 2:1."""
    threshold = signal_threshold if signal_threshold is not None else get_threshold()
    if confluence["score"] < threshold:
        return None
    if not structure:
        return None
    # Hard R:R gate — never log a trade with worse than 2:1
    if risk.get("rr", 0) < 2.0:
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

    return dict(
        direction=direction,
        score=confluence["score"],
        max_score=confluence["max"],
        entry=risk["current"],
        target=risk["target"],
        target_basis=risk.get("target_basis", ""),
        stop=risk["invalidation"],
        rr=risk["rr"],
        favorable=risk["favorable"],
        reason=reason,
        top_reasons=top_reasons,
        bar_time=int(h4_df.index[-1].timestamp()),
        factors_snapshot=confluence.get("factors_snapshot", {}),
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

    # Section 3: Volume & RSI & MACD
    vol       = volume_analysis(df)
    rsi_data  = rsi_analysis(df)
    macd_data = macd_analysis(df)

    # Section 5: Volatility
    df_atr = atr(df)
    volatility = dict(
        current=round(float(df_atr.iloc[-1]), 2),
        avg=round(float(df_atr.iloc[-20:].mean()), 2),
        expanding=bool(df_atr.iloc[-1] > df_atr.iloc[-20:].mean()),
        compressing=bool(df_atr.iloc[-1] < df_atr.iloc[-20:].mean() * 0.7),
    )

    # Load adaptive parameters from learning engine
    adapt_weights   = get_weights()
    adapt_threshold = get_threshold()
    adapt_stop_mult = get_stop_multiplier()

    # Section 6: Confluence (uses adaptive weights; MACD replaces Fibonacci slot)
    confluence = confluence_score(regime, structure, vol, rsi_data, sweeps, macd_data,
                                  interval, adapt_weights)

    # Section 7: Risk (uses adaptive stop multiplier)
    risk = risk_context(df, structure, sh, sl, interval, adapt_stop_mult)

    # Signal (score >= adaptive threshold + BOS)
    signal = generate_signal(confluence, structure, risk, df, adapt_threshold)

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
    ml, sl_s, hist = macd(df["close"])

    candles = [{"time": int(r.Index.timestamp()),
                "open": r.open, "high": r.high, "low": r.low, "close": r.close}
               for r in df.itertuples()]

    volumes = [{"time": int(r.Index.timestamp()), "value": r.volume,
                "color": "#26a69a" if r.close >= r.open else "#ef5350"}
               for r in df.itertuples()]

    ema50_pts     = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in ema50_s.items()]
    ema200_pts    = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in ema200_s.items()]
    rsi_pts       = [{"time": int(ts.timestamp()), "value": round(v, 2)} for ts, v in rsi_s.items() if not np.isnan(v)]
    obv_pts       = [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in obv_s.items() if not np.isnan(v)]
    obv_ema_pts   = [{"time": int(ts.timestamp()), "value": round(float(v), 2)} for ts, v in obv_ema_s.items() if not np.isnan(v)]
    macd_pts      = [{"time": int(ts.timestamp()), "value": round(float(v), 6)} for ts, v in ml.items() if not np.isnan(v)]
    macd_sig_pts  = [{"time": int(ts.timestamp()), "value": round(float(v), 6)} for ts, v in sl_s.items() if not np.isnan(v)]
    macd_hist_pts = [{"time": int(ts.timestamp()), "value": round(float(v), 6),
                      "color": "#3fb95088" if v >= 0 else "#f8514988"}
                     for ts, v in hist.items() if not np.isnan(v)]
    obv_sh, obv_sl = obv_swing_levels(obv_s)

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
        macd=macd_data,
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
            macd=macd_pts,
            macd_signal=macd_sig_pts,
            macd_hist=macd_hist_pts,
            obv_highs=obv_sh,
            obv_lows=obv_sl,
            channels=channels,
            levels=dict(
                support=key_support,
                resistance=key_resistance,
                support_touches=support_touches,
                resistance_touches=resistance_touches,
                invalidation=risk["invalidation"],
                target=risk["target"],
            ),
        ),
    )
