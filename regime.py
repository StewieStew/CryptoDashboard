"""
Market Regime Detector — determines per-symbol whether the market is trending
or ranging, and returns adjusted signal parameters accordingly.

Trending:      ADX > 25 AND clear HH/HL or LH/LL structure → standard params
Transitioning: ADX 20–25 or mixed structure → raise threshold +0.5
Ranging:       ADX < 20 AND no clear structure → raise threshold +1.0,
               widen RSI zones, reduce min R:R
"""

import pandas as pd
import analysis


def detect_regime(df: pd.DataFrame) -> dict:
    """
    Detect the current market regime for a symbol using ADX + price structure.

    Looks at:
      - Latest ADX value (strength of trend)
      - HH/HL count (bullish structure) vs LH/LL count (bearish structure)
        in the last ~60 bars

    Returns:
        {
            "regime":    "trending" | "ranging" | "transitioning",
            "direction": "bullish" | "bearish" | "neutral",
            "adx":       float,
            "hh_hl":     int,   # bullish structure event count
            "lh_ll":     int,   # bearish structure event count
        }
    """
    if len(df) < 50:
        return {
            "regime":    "transitioning",
            "direction": "neutral",
            "adx":       0.0,
            "hh_hl":     0,
            "lh_ll":     0,
        }

    # ADX value
    adx_s, _, _ = analysis.adx_indicator(df)
    adx_val = float(adx_s.iloc[-1]) if not adx_s.empty else 0.0

    # Swing structure in a 60-bar window
    window = min(60, len(df))
    df_win = df.iloc[-window:]
    sh, sl = analysis.detect_swings(df_win, n=3)

    hh_hl_count = 0
    lh_ll_count = 0

    if len(sh) >= 2:
        for i in range(1, len(sh)):
            if sh[i][1] > sh[i - 1][1]:
                hh_hl_count += 1
            else:
                lh_ll_count += 1

    if len(sl) >= 2:
        for i in range(1, len(sl)):
            if sl[i][1] > sl[i - 1][1]:
                hh_hl_count += 1
            else:
                lh_ll_count += 1

    # Direction from structure
    if hh_hl_count > lh_ll_count + 1:
        direction = "bullish"
    elif lh_ll_count > hh_hl_count + 1:
        direction = "bearish"
    else:
        direction = "neutral"

    # Regime classification
    if adx_val > 25 and (hh_hl_count >= 3 or lh_ll_count >= 3):
        regime = "trending"
    elif adx_val < 20 or (hh_hl_count < 2 and lh_ll_count < 2):
        regime = "ranging"
    else:
        regime = "transitioning"

    return {
        "regime":    regime,
        "direction": direction,
        "adx":       round(adx_val, 1),
        "hh_hl":     hh_hl_count,
        "lh_ll":     lh_ll_count,
    }


def detect_regime_from_data(analysis_data: dict) -> dict:
    """
    Detect market regime from an already-computed full_analysis() dict.
    Avoids an extra Binance API call during the background scanner.
    """
    adx_val = float((analysis_data.get("adx") or {}).get("value", 0))
    swings  = analysis_data.get("swings", {})

    sh_prices = [float(p) for _, p in swings.get("highs", [])]
    sl_prices = [float(p) for _, p in swings.get("lows",  [])]

    hh_hl_count = 0
    lh_ll_count = 0

    for i in range(1, len(sh_prices)):
        if sh_prices[i] > sh_prices[i - 1]:
            hh_hl_count += 1
        else:
            lh_ll_count += 1

    for i in range(1, len(sl_prices)):
        if sl_prices[i] > sl_prices[i - 1]:
            hh_hl_count += 1
        else:
            lh_ll_count += 1

    if hh_hl_count > lh_ll_count + 1:
        direction = "bullish"
    elif lh_ll_count > hh_hl_count + 1:
        direction = "bearish"
    else:
        direction = "neutral"

    if adx_val > 25 and (hh_hl_count >= 3 or lh_ll_count >= 3):
        regime_type = "trending"
    elif adx_val < 20 or (hh_hl_count < 2 and lh_ll_count < 2):
        regime_type = "ranging"
    else:
        regime_type = "transitioning"

    return {
        "regime":    regime_type,
        "direction": direction,
        "adx":       round(adx_val, 1),
        "hh_hl":     hh_hl_count,
        "lh_ll":     lh_ll_count,
    }


def get_regime_params(regime: dict, base_params: dict) -> dict:
    """
    Return a copy of base_params adjusted for the detected regime.

    Trending:
        No changes — standard parameters apply.

    Transitioning:
        score_threshold  +0.5 (be more selective)
        min_rr           capped at 2.5 (smaller moves in transitions)

    Ranging:
        score_threshold  +1.0 (much more selective — BOS unreliable)
        min_rr           capped at 2.5
        RSI zones widened (mean-reversion more relevant than momentum)
        sweep_weight_multiplier = 0 (sweeps misleading in consolidation)
    """
    params      = dict(base_params)
    regime_type = regime.get("regime", "transitioning")

    if regime_type == "ranging":
        params["score_threshold"] = min(
            params.get("score_threshold", 7.0) + 1.0, 9.5
        )
        params["min_rr"]          = min(params.get("min_rr", 3.0), 2.5)
        # Widen RSI zones for mean-reversion opportunities
        params["rsi_long_min"]    = 35.0
        params["rsi_long_max"]    = 70.0
        params["rsi_short_min"]   = 30.0
        params["rsi_short_max"]   = 65.0
        # Signal to the scanner that sweep weight should be zeroed
        params["sweep_weight_override"] = 0.0

    elif regime_type == "transitioning":
        params["score_threshold"] = min(
            params.get("score_threshold", 7.0) + 0.5, 9.5
        )
        params["min_rr"] = min(params.get("min_rr", 3.0), 2.5)

    # trending: no changes, standard params apply
    return params
