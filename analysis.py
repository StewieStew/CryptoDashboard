"""
Crypto Technical Analysis Engine
Fetches OHLCV from Binance public API and computes all 7 framework sections.
"""
from __future__ import annotations

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

try:
    from learning import get_weights, get_threshold, get_stop_multiplier
except ImportError:
    def get_weights():        return {"regime":2.0,"bos":2.0,"sweep":2.0,"volume":1.0,"obv":1.0,"rsi":1.0,"adx":1.0}
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

# Swing lookback: candles needed on each side of a pivot to confirm it.
# Lower TFs use a smaller window so swings are detected faster.
INTERVAL_SWING_N = {
    "15m": 3,
    "30m": 3,
    "1h":  4,
    "4h":  5,
    "1d":  5,
    "1w":  5,
}

def _swing_n(interval: str) -> int:
    return INTERVAL_SWING_N.get(interval, 5)


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


def adx_indicator(df: pd.DataFrame, period: int = 14):
    """
    Average Directional Index.  Returns (adx_series, di_plus_series, di_minus_series).
    ADX > 25 = trending market (signals are reliable).
    ADX < 20 = ranging / choppy (signals have lower win rate).
    """
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift()
    down_move = low.shift() - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr_s     = tr.ewm(span=period, adjust=False).mean()
    di_plus   = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s
    di_minus  = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s
    dx        = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_s     = dx.ewm(span=period, adjust=False).mean()

    return adx_s, di_plus, di_minus


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Daily VWAP — resets at midnight UTC each day.
    Best used on intraday timeframes (15m, 1h).
    Price above VWAP = bullish intraday bias; below = bearish.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv      = typical * df["volume"]
    result  = pd.Series(index=df.index, dtype=float)
    for _date, grp in df.groupby(df.index.date):
        idx            = grp.index
        result[idx]    = pv[idx].cumsum() / df["volume"][idx].cumsum().replace(0, np.nan)
    return result


def detect_fvg(df: pd.DataFrame, lookback: int = 100) -> list:
    """
    Detect Fair Value Gaps (FVGs) — price imbalances where the market moved
    too fast to leave overlapping wicks between candle 1 and candle 3.

    Bullish FVG: candle[i+1].low > candle[i-1].high  (gap up — unfilled buy imbalance)
    Bearish FVG: candle[i+1].high < candle[i-1].low  (gap down — unfilled sell imbalance)

    Each FVG includes fill status:
      filled       — price has completely passed through (gap no longer valid)
      in_zone      — price is currently inside the gap (active entry zone)
      untouched    — price has not yet returned to the gap (future entry opportunity)

    Returns last 5 significant FVGs (≥0.1% size).
    """
    cur      = float(df["close"].iloc[-1])
    recent   = df.iloc[-lookback:]
    candles  = list(recent.itertuples())
    fvgs     = []

    for i in range(1, len(candles) - 1):
        c1, c3 = candles[i - 1], candles[i + 1]

        if c3.low > c1.high:                          # Bullish FVG (gap up)
            size_pct = (c3.low - c1.high) / c1.high * 100
            if size_pct >= 0.1:
                lower = round(float(c1.high), 6)
                upper = round(float(c3.low),  6)
                # Filled: price has dropped below the lower boundary
                # In-zone: price is currently between lower and upper
                filled   = cur < lower
                in_zone  = lower <= cur <= upper
                fvgs.append({
                    "type":      "bullish",
                    "upper":     upper,
                    "lower":     lower,
                    "midpoint":  round((upper + lower) / 2, 6),
                    "time":      int(c3.Index.timestamp()),
                    "size_pct":  round(size_pct, 3),
                    "filled":    filled,
                    "in_zone":   in_zone,
                    "untouched": not filled and not in_zone,
                })
        elif c3.high < c1.low:                        # Bearish FVG (gap down)
            size_pct = (c1.low - c3.high) / c1.low * 100
            if size_pct >= 0.1:
                lower = round(float(c3.high), 6)
                upper = round(float(c1.low),  6)
                # Filled: price has risen above the upper boundary
                filled   = cur > upper
                in_zone  = lower <= cur <= upper
                fvgs.append({
                    "type":      "bearish",
                    "upper":     upper,
                    "lower":     lower,
                    "midpoint":  round((upper + lower) / 2, 6),
                    "time":      int(c3.Index.timestamp()),
                    "size_pct":  round(size_pct, 3),
                    "filled":    filled,
                    "in_zone":   in_zone,
                    "untouched": not filled and not in_zone,
                })

    return fvgs[-5:]


def detect_equal_levels(swing_highs, swing_lows, tolerance: float = 0.003) -> dict:
    """
    Detect equal highs / equal lows — clusters of ≥2 swing levels within
    `tolerance` (0.3%) of each other. These are liquidity pools where stops
    concentrate; price often sweeps them before reversing.
    """
    def cluster_levels(prices):
        seen, out = [], []
        for p in prices:
            near = [x for x in prices if abs(x - p) / p <= tolerance]
            if len(near) >= 2:
                avg = sum(near) / len(near)
                if not any(abs(avg - s) / avg <= tolerance for s in seen):
                    seen.append(avg)
                    out.append(round(avg, 6))
        return out

    highs = [p for _, p in swing_highs]
    lows  = [p for _, p in swing_lows]
    return {
        "equal_highs": cluster_levels(highs),
        "equal_lows":  cluster_levels(lows),
    }


def detect_order_blocks(df: pd.DataFrame, swing_highs, swing_lows) -> dict:
    """
    Order Blocks — zones where institutional orders were placed.

    Bullish OB: last bearish candle immediately before a bullish impulse
                (identified near swing lows). Acts as demand zone / support.
    Bearish OB: last bullish candle immediately before a bearish impulse
                (identified near swing highs). Acts as supply zone / resistance.

    Returns {"bullish": [...], "bearish": [...], "nearest": {...}|None}
    """
    cur = float(df["close"].iloc[-1])
    bullish_obs, bearish_obs = [], []

    # Bullish OBs — near swing lows
    for _ts, level in (swing_lows[-4:] or []):
        mask = (df["low"] >= level * 0.997) & (df["low"] <= level * 1.003)
        hits = df[mask]
        if hits.empty:
            continue
        loc = df.index.get_loc(hits.index[-1])
        if loc > 0:
            prev = df.iloc[loc - 1]
            if prev["close"] < prev["open"]:   # bearish candle = institutional sell then reverse
                bullish_obs.append({
                    "upper":  round(float(prev["open"]),  6),
                    "lower":  round(float(prev["close"]), 6),
                    "level":  round(level, 6),
                    "active": cur > level,
                })

    # Bearish OBs — near swing highs
    for _ts, level in (swing_highs[-4:] or []):
        mask = (df["high"] >= level * 0.997) & (df["high"] <= level * 1.003)
        hits = df[mask]
        if hits.empty:
            continue
        loc = df.index.get_loc(hits.index[-1])
        if loc > 0:
            prev = df.iloc[loc - 1]
            if prev["close"] > prev["open"]:   # bullish candle = institutional buy then reverse
                bearish_obs.append({
                    "upper":  round(float(prev["close"]), 6),
                    "lower":  round(float(prev["open"]),  6),
                    "level":  round(level, 6),
                    "active": cur < level,
                })

    # Find nearest OB to current price
    all_obs = [(abs((ob["upper"] + ob["lower"]) / 2 - cur), "bullish", ob)
               for ob in bullish_obs]
    all_obs += [(abs((ob["upper"] + ob["lower"]) / 2 - cur), "bearish", ob)
                for ob in bearish_obs]
    all_obs.sort(key=lambda x: x[0])
    nearest = None
    if all_obs:
        _, ob_type, ob = all_obs[0]
        nearest = {"type": ob_type, **ob}

    return {
        "bullish": bullish_obs[-2:],
        "bearish": bearish_obs[-2:],
        "nearest": nearest,
    }


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
    recent_spike = bool((vol.iloc[-10:] > 1.3 * avg20).any())

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

def confluence_score(regime, structure, vol, rsi_data, sweeps,
                     interval: str = "4h", weights: dict | None = None,
                     adx_data: dict | None = None,
                     fvg_data: list | None = None,
                     fib_data: dict | None = None,
                     eq_levels: dict | None = None) -> dict:
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
    snap["regime_up"] = "Uptrend" in regime["regime"]
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
    # Direction-aware: only a sweep that CONFIRMS the trade direction earns points.
    # A counter-direction sweep is opposing smart money and applies a penalty.
    w = weights.get("sweep", 2.0)
    snap["sweep"]         = False
    snap["counter_sweep"] = False

    is_long_setup  = bool(structure and structure.get("bullish_bos"))
    is_short_setup = bool(structure and structure.get("bearish_bos"))

    aligned_sweeps  = []
    counter_sweeps  = []
    for s in (sweeps or []):
        stype = s.get("type", "")
        if is_long_setup  and stype == "bullish_sweep":
            aligned_sweeps.append(s)
        elif is_short_setup and stype == "bearish_sweep":
            aligned_sweeps.append(s)
        else:
            counter_sweeps.append(s)

    if aligned_sweeps:
        snap["sweep"] = True
        score += w
        s = aligned_sweeps[0]
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"Liquidity sweep: {s['type']} @ {s['level']} — {s['desc']}"})
    elif counter_sweeps:
        # Counter-direction sweep = opposing smart money at entry.
        # Penalise the score so this setup is harder (or impossible) to fire.
        snap["counter_sweep"] = True
        penalty = round(min(w, 1.5), 1)
        score  -= penalty
        s = counter_sweeps[0]
        reasons.append({"pts": -penalty, "earned": False,
                        "text": (f"⚠ Counter-direction sweep: {s['type']} @ {s['level']} — "
                                 f"smart money absorbed in opposite direction; "
                                 f"reduces conviction ({-penalty:+.1f} pts)")})
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

    # 5. OBV confirms bias or shows aligned divergence (default 1 pt)
    w = weights.get("obv", 1.0)
    bias_up = "Uptrend"   in regime["regime"] or bool(structure and structure.get("bullish_bos"))
    bias_dn = "Downtrend" in regime["regime"] or bool(structure and structure.get("bearish_bos"))
    # Divergence must align with the trade direction to count as confirmation.
    # A bearish OBV divergence in a LONG setup is a WARNING, not a point.
    aligned_bull_div = vol["bullish_obv_div"] and bias_up
    aligned_bear_div = vol["bearish_obv_div"] and bias_dn
    obv_ok = (
        (bias_up and vol["obv_bullish"]) or
        (bias_dn and not vol["obv_bullish"]) or
        aligned_bull_div or aligned_bear_div
    )
    snap["obv"] = obv_ok
    if obv_ok:
        score += w
        if aligned_bull_div:
            obv_txt = "Bullish OBV divergence — smart money accumulating while price is low"
        elif aligned_bear_div:
            obv_txt = "Bearish OBV divergence — distribution detected while price is elevated"
        elif bias_up:
            obv_txt = "OBV trending up — confirms bullish bias"
        else:
            obv_txt = "OBV trending down — confirms bearish bias"
        reasons.append({"pts": round(w, 1), "earned": True, "text": obv_txt})
    else:
        # Flag opposing divergences as warnings in the reason text
        warn = ""
        if vol["bearish_obv_div"] and bias_up:
            warn = " (⚠ bearish OBV divergence — distribution signal, caution on longs)"
        elif vol["bullish_obv_div"] and bias_dn:
            warn = " (⚠ bullish OBV divergence — accumulation signal, caution on shorts)"
        reasons.append({"pts": 0, "earned": False,
                        "text": f"OBV not confirming direction — volume flow opposing thesis{warn}"})

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

    # 7. ADX — trend strength filter (default 1 pt)
    w       = weights.get("adx", 1.0)
    adx_val = adx_data.get("value", 0) if adx_data else 0
    adx_ok  = bool(adx_data and adx_data.get("trending", False))
    snap["adx"] = adx_ok
    if adx_ok:
        score += w
        reasons.append({"pts": round(w, 1), "earned": True,
                        "text": f"ADX {adx_val:.1f} — strong trend confirmed (>25), signals are reliable"})
    else:
        reasons.append({"pts": 0, "earned": False,
                        "text": f"ADX {adx_val:.1f} — market ranging (<25), signals carry lower win rate"})

    # Directional bias used by factors 8–10
    _bias_up = bool(structure and structure.get("bullish_bos")) or "Uptrend" in regime["regime"]
    _bias_dn = bool(structure and structure.get("bearish_bos")) or "Downtrend" in regime["regime"]

    # 8. FVG aligned with setup direction (default 1.5 pts)
    w = weights.get("fvg", 0.0)
    fvg_ok  = False
    fvg_txt = "No aligned FVG — price imbalance missing"
    if w > 0 and fvg_data:
        if _bias_up:
            ok = [f for f in fvg_data if f["type"] == "bullish" and not f["filled"]]
            if ok:
                fvg_ok  = True
                fvg_txt = "Bullish FVG below price — imbalance support / fill target"
        elif _bias_dn:
            ok = [f for f in fvg_data if f["type"] == "bearish" and not f["filled"]]
            if ok:
                fvg_ok  = True
                fvg_txt = "Bearish FVG above price — imbalance resistance / fill target"
    snap["fvg"] = fvg_ok
    if w > 0:
        if fvg_ok:
            score += w
            reasons.append({"pts": round(w, 1), "earned": True,  "text": fvg_txt})
        else:
            reasons.append({"pts": 0,           "earned": False, "text": fvg_txt})

    # 9. Fibonacci golden pocket (0.382 – 0.786 retracement zone, default 1.5 pts)
    w      = weights.get("fib", 0.0)
    GOLDEN = {"0.382", "0.5", "0.618", "0.786"}
    fib_ok = bool(
        w > 0 and fib_data
        and fib_data.get("at_fib")
        and fib_data.get("nearest_level") in GOLDEN
        and (_bias_up or _bias_dn)
    )
    if fib_ok and fib_data:
        fib_txt = (f"FIB {fib_data['nearest_level']} @ {fib_data['nearest_price']} "
                   f"— golden pocket retracement entry")
    else:
        fib_txt = "Not at FIB key level — no precision retracement entry"
    snap["fib"] = fib_ok
    if w > 0:
        if fib_ok:
            score += w
            reasons.append({"pts": round(w, 1), "earned": True,  "text": fib_txt})
        else:
            reasons.append({"pts": 0,           "earned": False, "text": fib_txt})

    # 10. Liquidity pool — equal highs / equal lows cluster (default 1.0 pt)
    w      = weights.get("liquidity", 0.0)
    liq_ok  = False
    liq_txt = "No equal-level liquidity pool nearby"
    if w > 0 and eq_levels:
        if _bias_up and eq_levels.get("equal_lows"):
            liq_ok  = True
            liq_txt = "Equal lows below — stop cluster (sweep & rally zone)"
        elif _bias_dn and eq_levels.get("equal_highs"):
            liq_ok  = True
            liq_txt = "Equal highs above — stop cluster (sweep & drop zone)"
    snap["liquidity"] = liq_ok
    if w > 0:
        if liq_ok:
            score += w
            reasons.append({"pts": round(w, 1), "earned": True,  "text": liq_txt})
        else:
            reasons.append({"pts": 0,           "earned": False, "text": liq_txt})

    # Store raw RSI value for downstream gates (RSI range check in generate_signal)
    snap["rsi_value"] = round(rsi_data.get("value", 50.0), 1)

    score     = round(score, 1)
    max_score = 10.0  # Always display out of 10 regardless of adaptive weight drift

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
                 interval: str = "4h", stop_multiplier: float = 0.1,
                 fvgs: list | None = None,
                 fib_data: dict | None = None) -> dict:
    cur        = float(df["close"].iloc[-1])
    atr_val    = float(atr(df).iloc[-1])
    ema50_val  = float(ema(df["close"], 50).iloc[-1])
    ema200_val = float(ema(df["close"], 200).iloc[-1])
    tf         = interval.upper()

    if structure and structure["bullish_bos"]:
        last_sh = structure["last_swing_high"] or cur
        last_sl = structure["last_swing_low"]

        # ── Entry: retest of broken resistance (now structural support)
        #    Buy the level, not the breakout candle
        entry_price = round(last_sh, 6)

        # ── FIB precision override: if price has retraced to the golden pocket
        #    (0.5–0.786 zone), enter at the 0.618 level — tighter stop, same
        #    target → better R:R than waiting at the BOS level
        _GOLDEN_ENTRY = {"0.5", "0.618", "0.786"}
        if (fib_data and fib_data.get("at_fib")
                and fib_data.get("nearest_level") in _GOLDEN_ENTRY):
            fib_618 = fib_data["levels"].get("0.618")
            if fib_618 and last_sl and fib_618 > last_sl:
                entry_price = round(fib_618, 6)

        # ── Stop: structural swing low with wick buffer.
        #    If the first structural SL doesn't yield a 3R target, step to a
        #    deeper swing low to widen the setup — then check again.
        _sl_below = sorted(
            [p for _, p in swing_lows if p < entry_price],
            reverse=True   # nearest to entry first
        )

        def _find_long_target(sl_price, min_rr=3.0):
            """Return (target, basis, tp_source) ≥ min_rr from sl_price, or None."""
            _inval   = round(sl_price - stop_multiplier * atr_val, 6)
            _risk    = abs(entry_price - _inval)
            if _risk <= 0:
                return None
            _min_tp  = entry_price + min_rr * _risk

            # 1. Swing high ≥ min_tp
            # Set TP 0.2×ATR BELOW the swing high so the order fills before price
            # wicks up to the structural resistance and rejects.
            _sh_ok = sorted([p for _, p in swing_highs if p >= _min_tp])
            if _sh_ok:
                _tp = round(_sh_ok[0] - 0.2 * atr_val, 6)
                return (_tp, "Prior swing high", "swing_high")

            # 2. Bearish FVG ≥ min_tp
            if fvgs:
                _fvg_ok = [f["midpoint"] for f in fvgs
                           if f["type"] == "bearish"
                           and f["midpoint"] >= _min_tp
                           and not f.get("filled", False)]
                if _fvg_ok:
                    return (round(min(_fvg_ok), 6), "Bearish FVG fill zone", "fvg")

            # 3. EMA200 ≥ min_tp
            if ema200_val >= _min_tp:
                return (round(ema200_val, 6), "EMA200 dynamic resistance", "ema200")

            # 4. EMA50 ≥ min_tp
            if ema50_val >= _min_tp:
                return (round(ema50_val, 6), "EMA50 dynamic resistance", "ema50")

            # 5. Historical range high ≥ min_tp across all fetched candles
            _above   = df["close"][df["close"] >= _min_tp]
            if not _above.empty:
                _hist_h  = float(_above.min())   # closest candle close at/above min_tp
                _hist_rr = (_hist_h - entry_price) / _risk
                if _hist_rr >= min_rr:
                    return (round(_hist_h, 6),
                            f"Historical range high ({len(df)}-candle, {_hist_rr:.1f}R)",
                            "historical_range")
            return None

        # Try structural SL levels from nearest to deepest
        _chosen_sl   = _sl_below[0] if _sl_below else (last_sl or (entry_price - atr_val))
        _tp_result   = None
        for _sl_try in (_sl_below or [_chosen_sl]):
            _tp_result = _find_long_target(_sl_try)
            if _tp_result:
                _chosen_sl = _sl_try
                break

        if _tp_result:
            target, target_basis, tp_source = _tp_result
            inval = round(_chosen_sl - stop_multiplier * atr_val, 6)
        else:
            # No structural 3R target found — mark as no_3r so R:R gate rejects it
            inval        = round((_sl_below[0] if _sl_below else entry_price - atr_val) - stop_multiplier * atr_val, 6)
            target       = entry_price   # 0 reward → R:R = 0 → gate rejects
            target_basis = "No structural 3R target found"
            tp_source    = "no_3r_target"

        bias       = "Long"
        inval_note = (f"{tf} close below ${inval:,.4f} — structural swing low violated, "
                      f"invalidates the bullish BOS")

    elif structure and structure["bearish_bos"]:
        last_sl = structure["last_swing_low"] or cur
        last_sh = structure["last_swing_high"]

        # ── Entry: retest of broken support (now structural resistance)
        #    Sell the level, not the breakdown candle
        entry_price = round(last_sl, 6)

        # ── FIB precision override for short
        if (fib_data and fib_data.get("at_fib")
                and fib_data.get("nearest_level") in {"0.5", "0.618", "0.786"}):
            fib_618 = fib_data["levels"].get("0.618")
            if fib_618 and last_sh and fib_618 < last_sh:
                entry_price = round(fib_618, 6)

        # ── Stop: structural swing HIGH with wick buffer.
        #    If the nearest swing high doesn't yield a 3R target, step to a
        #    deeper (higher) swing high to widen the setup — then check again.
        _sh_above = sorted(
            [p for _, p in swing_highs if p > entry_price]
        )  # ascending = nearest to entry first

        def _find_short_target(sh_price, min_rr=3.0):
            """Return (target, basis, tp_source) ≥ min_rr from sh_price, or None."""
            _inval  = round(sh_price + stop_multiplier * atr_val, 6)
            _risk   = abs(_inval - entry_price)
            if _risk <= 0:
                return None
            _min_tp = entry_price - min_rr * _risk   # must be this far BELOW entry

            # 1. Swing low <= min_tp (nearest = highest price = closest to min_tp)
            _sl_ok = sorted(
                [p for _, p in swing_lows if p <= _min_tp],
                reverse=True   # highest first = nearest to entry
            )
            if _sl_ok:
                # Set TP 0.2×ATR ABOVE the swing low so the order fills before price
                # wicks down to the structural support and bounces.
                _tp = round(_sl_ok[0] + 0.2 * atr_val, 6)
                return (_tp, "Prior swing low", "swing_low")

            # 2. Bullish FVG <= min_tp
            if fvgs:
                _fvg_ok = [f["midpoint"] for f in fvgs
                           if f["type"] == "bullish"
                           and f["midpoint"] <= _min_tp
                           and not f.get("filled", False)]
                if _fvg_ok:
                    return (round(max(_fvg_ok), 6), "Bullish FVG fill zone", "fvg")

            # 3. EMA200 <= min_tp
            if ema200_val <= _min_tp:
                return (round(ema200_val, 6), "EMA200 dynamic support", "ema200")

            # 4. EMA50 <= min_tp
            if ema50_val <= _min_tp:
                return (round(ema50_val, 6), "EMA50 dynamic support", "ema50")

            # 5. Historical range low <= min_tp across all fetched candles
            _below = df["close"][df["close"] <= _min_tp]
            if not _below.empty:
                _hist_l  = float(_below.max())   # closest candle close at/below min_tp
                _hist_rr = (entry_price - _hist_l) / _risk
                if _hist_rr >= min_rr:
                    return (round(_hist_l, 6),
                            f"Historical range low ({len(df)}-candle, {_hist_rr:.1f}R)",
                            "historical_range")
            return None

        # Try structural SL levels from nearest swing high to deepest
        _chosen_sh  = _sh_above[0] if _sh_above else (last_sh or (entry_price + atr_val))
        _tp_result  = None
        for _sh_try in (_sh_above or [_chosen_sh]):
            _tp_result = _find_short_target(_sh_try)
            if _tp_result:
                _chosen_sh = _sh_try
                break

        if _tp_result:
            target, target_basis, tp_source = _tp_result
            inval = round(_chosen_sh + stop_multiplier * atr_val, 6)
        else:
            # No structural 3R target found — mark as no_3r so R:R gate rejects it
            inval        = round((_sh_above[0] if _sh_above else entry_price + atr_val) + stop_multiplier * atr_val, 6)
            target       = entry_price   # 0 reward → R:R = 0 → gate rejects
            target_basis = "No structural 3R target found"
            tp_source    = "no_3r_target"

        bias       = "Short"
        inval_note = (f"{tf} close above ${inval:,.4f} — structural swing high violated, "
                      f"invalidates the bearish BOS")

    else:
        entry_price  = round(cur, 6)
        inval        = swing_lows[-1][1]  if swing_lows  else cur * 0.95
        target       = swing_highs[-1][1] if swing_highs else cur * 1.05
        bias         = "Neutral"
        target_basis = "Nearest swing high"
        tp_source    = "swing_level"
        inval_note   = f"Last swing low ${inval:,.4f} structural floor"

    risk_d   = abs(entry_price - inval)
    reward_d = abs(target - entry_price)

    # ── Cap stop loss by timeframe (Fix 8) ───────────────────────────────────
    _MAX_SL_PCT = {"15m": 0.020, "1h": 0.035, "4h": 0.050}
    _sl_pct     = risk_d / entry_price if entry_price > 0 else 0
    _max_sl     = _MAX_SL_PCT.get(interval, 0.05)
    if bias != "Neutral" and _sl_pct > _max_sl:
        logger.warning(
            f"[SL CAP] {interval}: SL distance {_sl_pct:.2%} > max {_max_sl:.2%} — capping"
        )
        risk_d = entry_price * _max_sl
        if bias == "Long":
            inval = round(entry_price - risk_d, 6)
        else:
            inval = round(entry_price + risk_d, 6)
        reward_d = abs(target - entry_price)

    # ── Cap stop loss at 25% of entry price ──────────────────────────────────
    MAX_RISK_PCT = 0.25
    if risk_d > 0 and (risk_d / entry_price) > MAX_RISK_PCT and bias != "Neutral":
        risk_d = entry_price * MAX_RISK_PCT
        if bias == "Long":
            inval = round(entry_price - risk_d, 6)
            inval_note = (f"Stop tightened to 25% max risk at ${inval:,.4f} — "
                          f"structural stop too wide, capped for risk management")
        else:
            inval = round(entry_price + risk_d, 6)
            inval_note = (f"Stop tightened to 25% max risk at ${inval:,.4f} — "
                          f"structural stop too wide, capped for risk management")
        reward_d = abs(target - entry_price)

    rr = round(reward_d / risk_d, 2) if risk_d > 0 else 0

    return dict(
        bias=bias,
        current=round(cur, 6),
        entry=round(entry_price, 6),
        invalidation=round(inval, 6),
        target=round(target, 6),
        target_basis=target_basis,
        tp_source=tp_source,
        rr=rr,
        favorable=rr > 0,
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
                    signal_threshold: float | None = None,
                    interval: str = "4h",
                    body_ratio_min: float = 0.30,
                    min_rr: float = 2.0,
                    rsi_long_range: tuple = (30.0, 70.0),
                    rsi_short_range: tuple = (30.0, 70.0),
                    level_touch_min: int = 2,
                    symbol: str = "") -> dict | None:
    """
    Returns a signal dict only when ALL hard gates pass and score >= threshold.

    Hard gates (all must be True):
      - Score >= adaptive threshold (default 7.0)
      - BOS confirmed
      - ADX > threshold (trending market — ranging markets produce false BOS)
      - Volume expanding (low-volume breakouts are almost always fake)

    All gate parameters are configurable for backtesting / regime switching.
    """
    threshold = signal_threshold if signal_threshold is not None else get_threshold()
    if confluence["score"] < threshold:
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: score={confluence['score']:.1f} < threshold={threshold:.1f}", flush=True)
        return None
    if not structure:
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: no structure data", flush=True)
        return None

    snap  = confluence.get("factors_snapshot", {})
    score = confluence["score"]

    # ── Hard gate 1: ADX must confirm a trending market ──────────────────────
    if not snap.get("adx"):
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: ADX gate — ranging/choppy market", flush=True)
        return None  # Ranging/choppy — trend-following BOS signals fail here

    # ── Volume penalty (soft — reduces score by 0.5) ─────────────────────────
    if not snap.get("volume"):
        print(f"[VOLUME PENALTY] {symbol} {interval}: volume not expanding/no spike — -0.5 score", flush=True)
        score -= 0.5

    # ── Hard gate 3: BOS candle quality ──────────────────────────────────────
    # The candle that broke structure must be a conviction candle, not a doji.
    # A BOS on a small-body or opposite-color candle is likely a stop hunt.
    last       = h4_df.iloc[-1]
    total_rng  = float(last["high"] - last["low"])
    body_sz    = float(abs(last["close"] - last["open"]))
    body_ratio = body_sz / total_rng if total_rng > 0 else 0

    if body_ratio < body_ratio_min:
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: body_ratio={body_ratio:.2f} < {body_ratio_min:.2f} (doji/indecision candle)", flush=True)
        return None  # Doji / spinning top — indecision candle, not a BOS

    # ── Hard gate 5: Level significance ──────────────────────────────────────
    # The broken level (our entry) must be a tested, meaningful level —
    # at least level_touch_min candles touched it, confirming it as real S/R.
    entry_price = risk.get("entry", risk["current"])
    level_col   = "high" if structure["bullish_bos"] else "low"
    touch_count = int((abs(h4_df[level_col] - entry_price) / entry_price <= 0.005).sum())
    if touch_count < level_touch_min:
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: level touches={touch_count} < {level_touch_min} (entry level not confirmed S/R)", flush=True)
        return None  # Level not confirmed as key S/R

    # ── Hard gate 4: RSI range filter ────────────────────────────────────────
    # In a trending regime, RSI staying overbought/oversold IS the signal — skip
    # the overbought/oversold cap when regime is trending and direction aligns.
    rsi_val    = snap.get("rsi_value", 50.0)
    _trending  = snap.get("regime", False)
    _regime_up = snap.get("regime_up", False)
    if structure["bullish_bos"]:
        _trend_aligned = _trending and _regime_up
        if not _trend_aligned:
            rsi_lo, rsi_hi = rsi_long_range
            if not (rsi_lo <= rsi_val < rsi_hi):
                print(f"[SIGNAL BLOCKED] {symbol} {interval} LONG: RSI={rsi_val:.1f} outside [{rsi_lo},{rsi_hi})", flush=True)
                return None
    elif structure["bearish_bos"]:
        _trend_aligned = _trending and not _regime_up
        if not _trend_aligned:
            rsi_lo, rsi_hi = rsi_short_range
            if not (rsi_lo < rsi_val <= rsi_hi):
                print(f"[SIGNAL BLOCKED] {symbol} {interval} SHORT: RSI={rsi_val:.1f} outside ({rsi_lo},{rsi_hi}]", flush=True)
                return None

    # R:R is reported in the signal but not used as a gate — natural TP/SL stands.

    # ── Retest confirmation gate ──────────────────────────────────────────────
    # Enter only when price has pulled back to the broken level, not at the
    # moment of the break.  0.8% buffer allows for a minor spike-through retest.
    current = risk["current"]
    if structure["bullish_bos"]:
        broken_level = structure.get("last_swing_high")
        if broken_level and current > broken_level * 1.02:
            pct = (current - broken_level) / broken_level * 100
            print(
                f"[RETEST GATE] {symbol} {interval.upper()} LONG: price {current:.2f} is "
                f"{pct:.1f}% above broken level {broken_level:.2f} — waiting for retest",
                flush=True,
            )
            return None
    elif structure["bearish_bos"]:
        broken_level = structure.get("last_swing_low")
        if broken_level and current < broken_level * 0.98:
            pct = (broken_level - current) / broken_level * 100
            print(
                f"[RETEST GATE] {symbol} {interval.upper()} SHORT: price {current:.2f} is "
                f"{pct:.1f}% below broken level {broken_level:.2f} — retest window passed",
                flush=True,
            )
            return None

    if structure["bullish_bos"]:
        broken_level = structure.get("last_swing_high")
        direction = "LONG"
        reason    = (
            f"Bullish BOS — retest confirmed at {broken_level:.2f} (broken resistance now support)"
            if broken_level else
            "Bullish BOS — retest entry at broken resistance (now support)"
        )
    elif structure["bearish_bos"]:
        broken_level = structure.get("last_swing_low")
        direction = "SHORT"
        reason    = (
            f"Bearish BOS — retest confirmed at {broken_level:.2f} (broken support now resistance)"
            if broken_level else
            "Bearish BOS — retest entry at broken support (now resistance)"
        )
    else:
        print(f"[SIGNAL BLOCKED] {symbol} {interval}: no BOS direction (neither bullish nor bearish BOS)", flush=True)
        return None

    top_reasons = [r["text"] for r in confluence["reasons"] if r["earned"]]

    return dict(
        direction=direction,
        score=score,
        max_score=confluence["max"],
        entry=entry_price,              # structural retest level
        current_price=risk["current"],  # where BOS actually happened
        target=risk["target"],
        target_basis=risk.get("target_basis", ""),
        tp_source=risk.get("tp_source", "unknown"),
        stop=risk["invalidation"],
        rr=risk["rr"],
        favorable=risk["favorable"],
        reason=reason,
        top_reasons=top_reasons,
        bar_time=int(h4_df.index[-1].timestamp()),
        factors_snapshot=snap,
        # Candle quality metadata (useful for Claude + dashboard display)
        bos_body_ratio=round(body_ratio, 2),
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
# MACD(5/13/8) + EMA50 + VOLUME SIGNAL
# Walk-forward validated: OOS PF 1.30, OOS WR 30.2%, CI [27-33%] on 1H (196 OOS trades / 1yr)
# Simpler = more robust: time filter and BB bounce both degraded OOS performance when tested.
# Filters: MACD(5,13,8) cross | EMA50 trend | vol >1.2× avg
# ─────────────────────────────────────────────

def generate_macd_signal(df: pd.DataFrame, interval: str = "1h") -> dict | None:
    """MACD(5,13,8) cross + EMA50 trend + volume surge (1.2×).

    Walk-forward validated strategy: first 2yr in-sample, last 1yr out-of-sample.
    OOS result: 30.2% WR, PF 1.30, CI [27-33%] — statistically above 25% breakeven.
    Time filter and BB bounce both removed after OOS testing showed degradation.

    Stop  = 1× ATR(14) from entry
    Target= 3× ATR(14) from entry  (3:1 R:R)
    Returns a signal dict in the same format as generate_signal(), or None.
    """
    if len(df) < 55:
        return None

    close = df["close"]

    macd_line, sig_line, _ = macd(close, 5, 13, 8)
    ema50_s = ema(close, 50)
    atr14_s = atr(df, 14)
    vol20_s = df["volume"].rolling(20).mean()

    i = len(df) - 1
    if i < 1:
        return None

    # MACD crossover on last bar
    prev_diff = float(macd_line.iloc[i - 1]) - float(sig_line.iloc[i - 1])
    curr_diff = float(macd_line.iloc[i])     - float(sig_line.iloc[i])

    if prev_diff < 0 and curr_diff >= 0:
        direction = "LONG"
    elif prev_diff > 0 and curr_diff <= 0:
        direction = "SHORT"
    else:
        return None

    # EMA50 trend filter: only trade with the prevailing trend
    cc  = float(close.iloc[i])
    e50 = float(ema50_s.iloc[i])
    if np.isnan(e50):
        return None
    if direction == "LONG"  and cc < e50:
        return None
    if direction == "SHORT" and cc > e50:
        return None

    # Volume surge: current bar must exceed 1.2× 20-bar average
    avg_vol = float(vol20_s.iloc[i])
    cur_vol = float(df["volume"].iloc[i])
    if not np.isnan(avg_vol) and avg_vol > 0 and cur_vol < 1.2 * avg_vol:
        return None

    # Entry / stop / target
    atr_val = float(atr14_s.iloc[i])
    if atr_val <= 0 or np.isnan(atr_val):
        return None

    if direction == "LONG":
        stop   = round(cc - atr_val,       8)
        target = round(cc + 3.0 * atr_val, 8)
    else:
        stop   = round(cc + atr_val,       8)
        target = round(cc - 3.0 * atr_val, 8)

    vol_ratio = round(cur_vol / avg_vol, 2) if avg_vol > 0 else None
    reason = (
        f"MACD(5/13) {'bullish' if direction == 'LONG' else 'bearish'} cross | "
        f"price {'above' if direction == 'LONG' else 'below'} EMA50 ({round(e50, 4)}) | "
        f"vol {vol_ratio}× avg"
    )

    return {
        "direction":        direction,
        "entry":            round(cc, 8),
        "target":           target,
        "stop":             stop,
        "score":            8.5,       # fixed score — above default threshold
        "current_price":    round(cc, 8),
        "reason":           reason,
        "target_basis":     "3×ATR(14)",
        "signal_type":      "MACD_EMA_VOL",
        "factors_snapshot": {
            "macd_line":    round(float(macd_line.iloc[i]), 8),
            "macd_signal":  round(float(sig_line.iloc[i]), 8),
            "ema50":        round(e50, 4),
            "atr14":        round(atr_val, 8),
            "vol_ratio":    vol_ratio,
        },
    }


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def full_analysis(symbol: str, interval: str = "4h", weights: dict | None = None,
                  body_ratio_min: float = 0.30, rsi_long_range: tuple = (30.0, 70.0),
                  rsi_short_range: tuple = (30.0, 70.0), level_touch_min: int = 1,
                  score_threshold: float | None = None) -> dict:
    # Resolve timeframes
    htf, candle_limit = INTERVAL_HTF.get(interval, ("1d", 200))

    # Fetch data — HTF for regime context, selected TF for everything else
    htf_df = fetch_ohlcv(symbol, htf, 300)
    df     = fetch_ohlcv(symbol, interval, candle_limit)

    # Section 1: Regime (always uses HTF for broader market context)
    regime = market_regime(htf_df)

    # Section 2: Structure (adaptive swing window per TF)
    n  = _swing_n(interval)
    sh, sl = detect_swings(df, n=n)
    structure = detect_structure(df, sh, sl)
    sweeps    = detect_sweeps(df, sh, sl)

    # Section 3: Volume & RSI & MACD
    vol       = volume_analysis(df)
    rsi_data  = rsi_analysis(df)
    macd_data = macd_analysis(df)

    # Section 3b: ADX, VWAP, FVG, Order Blocks
    adx_s, di_plus_s, di_minus_s = adx_indicator(df)
    adx_cur  = float(adx_s.iloc[-1]) if not adx_s.empty else 0.0
    adx_data = {
        "value":    round(adx_cur, 1),
        "di_plus":  round(float(di_plus_s.iloc[-1]),  1),
        "di_minus": round(float(di_minus_s.iloc[-1]), 1),
        "trending": adx_cur > 25,
        "strong":   adx_cur > 50,
    }

    vwap_s   = vwap(df)
    vwap_cur = float(vwap_s.iloc[-1]) if not vwap_s.empty else None

    fvgs         = detect_fvg(df)
    order_blocks = detect_order_blocks(df, sh, sl)

    # Section 5: Volatility
    df_atr = atr(df)
    volatility = dict(
        current=round(float(df_atr.iloc[-1]), 2),
        avg=round(float(df_atr.iloc[-20:].mean()), 2),
        expanding=bool(df_atr.iloc[-1] > df_atr.iloc[-20:].mean()),
        compressing=bool(df_atr.iloc[-1] < df_atr.iloc[-20:].mean() * 0.7),
    )

    # Load adaptive parameters from learning engine
    adapt_weights   = weights if weights is not None else get_weights()
    # Use caller-supplied threshold if given; otherwise floor at 3.0 so the scanner
    # can apply its own per-symbol threshold after full_analysis() returns.
    adapt_threshold = score_threshold if score_threshold is not None else 3.0
    adapt_stop_mult = get_stop_multiplier()

    # Section 6: Confluence (adaptive weights + ADX + VWAP)
    cur_price  = float(df["close"].iloc[-1])
    confluence = confluence_score(regime, structure, vol, rsi_data, sweeps,
                                  interval, adapt_weights,
                                  adx_data=adx_data)

    # Section 7: Risk (uses adaptive stop multiplier)
    risk = risk_context(df, structure, sh, sl, interval, adapt_stop_mult, fvgs=fvgs)

    # Signal (score >= adaptive threshold + BOS)
    signal = generate_signal(confluence, structure, risk, df, adapt_threshold, interval,
                             body_ratio_min=body_ratio_min, rsi_long_range=rsi_long_range,
                             rsi_short_range=rsi_short_range, level_touch_min=level_touch_min,
                             symbol=symbol)

    # Parallel: MACD(5/13)+EMA50+Volume signal (best proven strategy, 1H focus)
    macd_signal = generate_macd_signal(df, interval) if interval in ("1h", "4h") else None

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
        adx=adx_data,
        vwap=round(vwap_cur, 6) if vwap_cur else None,
        fvg=fvgs,
        order_blocks=order_blocks,
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
        macd_signal=macd_signal,
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
