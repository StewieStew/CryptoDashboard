"""
Claude AI signal evaluation — uses Haiku for low latency on every signal,
and Sonnet for periodic deeper performance reviews.

Set ANTHROPIC_API_KEY in your Render environment variables.
If unset, all functions return empty dicts gracefully.
"""

import os
import json
import anthropic

def _get_client() -> anthropic.Anthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        return anthropic.Anthropic(api_key=key)
    except Exception:
        return None


def _tier(interval: str) -> str:
    return "Day" if interval in ("15m", "30m", "1h") else "Swing"


def _fmt(v) -> str:
    if v is None:
        return "—"
    n = float(v)
    if n >= 10000:  return f"{n:,.2f}"
    if n >= 100:    return f"{n:.3f}"
    if n >= 1:      return f"{n:.4f}"
    if n >= 0.01:   return f"{n:.5f}"
    return f"{n:.6f}"


# ── Signal evaluation ─────────────────────────────────────────────────────────

def analyze_signal(signal: dict, trade_history: list,
                   market_context: dict,
                   ai_accuracy: dict | None = None) -> dict:
    """
    Ask Claude Haiku to evaluate a trade signal before it's logged.

    Returns:
        {confidence, recommendation, reasoning, risks, positives}
        or {} if API key not set or call fails.

    `recommendation` is one of: "strong_take" | "take" | "skip"
    """
    client = _get_client()
    if not client:
        return {}

    sym       = signal.get("symbol", "")
    direction = signal.get("direction", "")
    interval  = signal.get("interval", "")
    entry     = float(signal.get("entry") or 0)
    tp        = float(signal.get("tp") or signal.get("target") or 0)
    sl        = float(signal.get("sl") or signal.get("stop") or 0)
    score     = signal.get("score", 0)
    reason    = signal.get("reason", "")
    factors   = signal.get("factors_snapshot", {})
    adx       = signal.get("adx_value", 0)
    vwap_side = signal.get("vwap_side", "")   # "above" | "below" | ""

    tp_pct = abs((tp - entry) / entry * 100) if entry else 0
    sl_pct = abs((sl - entry) / entry * 100) if entry else 0
    rr     = round(tp_pct / sl_pct, 2) if sl_pct else 0

    # Recent symbol performance
    sym_closed = [t for t in trade_history
                  if t.get("symbol") == sym and t.get("status") in ("win", "loss")][-10:]
    sym_wins  = sum(1 for t in sym_closed if t.get("status") == "win")

    # Overall system
    all_closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
    total      = len(all_closed)
    win_rate   = (sum(1 for t in all_closed if t.get("status") == "win") / total * 100) if total else 0
    avg_roi    = (sum(t.get("roi_pct") or 0 for t in all_closed) / total) if total else 0

    # Market context lines
    fg      = market_context.get("fear_greed", {})
    btc_dom = market_context.get("btc_dominance", 0)
    funding = market_context.get("funding_rate", 0)
    oi      = market_context.get("open_interest", 0)

    ctx_parts = []
    if fg:
        ctx_parts.append(f"Fear & Greed: {fg.get('value','?')} — {fg.get('label','?')}")
    if btc_dom:
        ctx_parts.append(f"BTC Dominance: {btc_dom:.1f}%")
    if funding:
        direction_note = "longs paying shorts (overheated)" if funding > 0.05 else \
                         "shorts paying longs (squeeze risk)" if funding < -0.05 else "neutral"
        ctx_parts.append(f"Funding Rate: {funding:+.4f}% ({direction_note})")
    if oi:
        ctx_parts.append(f"Open Interest: {oi:,.0f} contracts")
    if adx:
        ctx_parts.append(f"ADX: {adx:.1f} ({'trending' if adx > 25 else 'ranging'})")
    if vwap_side:
        ctx_parts.append(f"Price vs VWAP: {vwap_side}")

    ctx_str = "\n".join(ctx_parts) if ctx_parts else "No extended market context available."

    # ── Build Claude's own accuracy block ─────────────────────────────────────
    acc_parts = []
    if ai_accuracy:
        for rec in ("strong_take", "take"):
            s = ai_accuracy.get(rec)
            if s and s["total"] >= 1:
                acc_parts.append(
                    f"  {rec}: {s['total']} trades logged → "
                    f"{s['wins']}W / {s['losses']}L "
                    f"({s['win_rate_pct']:.0f}% win rate, avg ROI {s['avg_roi_pct']:+.2f}%)"
                )
    acc_str = (
        "\n".join(acc_parts)
        if acc_parts
        else "  No closed trades yet — calibrate conservatively until track record builds."
    )

    system_prompt = (
        "You are a professional crypto quant trader evaluating algorithmic signals. "
        "Be direct and concise. Respond ONLY with valid JSON — no markdown, no explanation outside the JSON."
    )

    user_prompt = f"""Evaluate this trade signal:

SIGNAL:
- Symbol: {sym} | Direction: {direction} | Timeframe: {interval} ({_tier(interval)} trade)
- Entry: ${_fmt(entry)} | Target: ${_fmt(tp)} (+{tp_pct:.1f}%) | Stop: ${_fmt(sl)} (-{sl_pct:.1f}%)
- Technical Score: {score}/10 | R:R: {rr}:1
- Factors: {reason}
- Factor snapshot: {json.dumps(factors)}

MARKET CONTEXT:
{ctx_str}

SYSTEM TRACK RECORD:
- Win rate: {win_rate:.0f}% over {total} closed trades | Avg ROI: {avg_roi:.2f}%
- {sym} recent: {sym_wins}/{len(sym_closed)} wins

YOUR OWN PREDICTION ACCURACY (use this to self-calibrate):
{acc_str}
- If your strong_take win rate is below 55%, be more selective — raise the bar for strong_take.
- If your take win rate is below 45%, tighten your criteria — skip more borderline setups.
- If both are performing well (>65%), you can trust your conviction signals more.

Respond with this exact JSON:
{{"confidence": <0-100>, "recommendation": "<strong_take|take|skip>", "reasoning": "<2 sentences max>", "risks": ["<specific risk>"], "positives": ["<specific positive>"]}}

Rules: skip if R:R < 2 or score < 7. strong_take only if 3+ major factors confirm AND market context aligns. Be critical — false signals hurt the system."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return json.loads(msg.content[0].text.strip())
    except Exception:
        return {}


# ── On-demand chart analysis ─────────────────────────────────────────────────

def analyze_chart(symbol: str, interval: str, data: dict) -> dict:
    """
    Ask Claude Haiku for a TA read on the currently viewed chart.
    Called by /api/ai-chart when the user opens the AI card.

    Returns {bias, confidence, summary, watch_for, key_levels, risks} or {}
    """
    client = _get_client()
    if not client:
        return {}

    price     = data.get("current_price", 0)
    regime    = data.get("regime", {})
    structure = data.get("structure", {}) or {}
    rsi_d     = data.get("rsi", {}) or {}
    macd_d    = data.get("macd", {}) or {}
    vol_d     = data.get("volume", {}) or {}
    conf_d    = data.get("confluence", {}) or {}
    risk_d    = data.get("risk", {}) or {}
    adx_d     = data.get("adx", {}) or {}
    vwap      = data.get("vwap")

    above_200 = regime.get("above_200", False)
    bos = ("Bullish BOS" if structure.get("bullish_bos") else
           "Bearish BOS" if structure.get("bearish_bos") else
           "HH/HL uptrend" if structure.get("hh_hl") else
           "LH/LL downtrend" if structure.get("lh_ll") else "No confirmed BOS")

    tier = _tier(interval)

    vwap_str = (f"${_fmt(vwap)} — price {'above' if price > vwap else 'below'}"
                if vwap else "N/A")

    prompt = f"""Analyze this {tier} crypto chart and provide a trading bias.

CHART: {symbol} | {interval} timeframe | Current price: ${_fmt(price)}

TECHNICAL DATA:
- Macro regime: {'Bullish' if above_200 else 'Bearish'} (200 EMA: ${_fmt(regime.get('ema200', 0))}, price {'above' if above_200 else 'below'})
- Market structure: {bos}
- RSI(14): {rsi_d.get('value', 0):.1f} ({rsi_d.get('range', '?')})
- MACD: {'above signal line' if macd_d.get('above_signal') else 'below signal line'}, histogram {'positive' if macd_d.get('histogram_positive') else 'negative'}
- Volume: {'expanding' if vol_d.get('expanding') else 'contracting'} ({vol_d.get('trend', '?')})
- ADX: {adx_d.get('value', 0):.1f} ({'trending' if adx_d.get('trending') else 'ranging/weak trend'})
- VWAP: {vwap_str}
- Confluence score: {conf_d.get('score', 0)}/{conf_d.get('max', 12)}
- R:R setup: {risk_d.get('rr', 0)}:1 ({'favorable' if risk_d.get('favorable') else 'unfavorable'})
- Entry: ${_fmt(risk_d.get('current', 0))} | Target: ${_fmt(risk_d.get('target', 0))} | Stop: ${_fmt(risk_d.get('invalidation', 0))}

Respond with ONLY this JSON (no markdown):
{{"bias":"<LONG|SHORT|NEUTRAL>","confidence":<0-100>,"summary":"<2 sentences max — current TA read>","watch_for":"<1 sentence — key trigger to confirm or invalidate>","key_levels":["<price: what it means>"],"risks":["<specific risk>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            system="You are a professional crypto technical analyst. Be direct and concise. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences if Claude wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[-2] if "```" in raw[3:] else raw
            raw = raw.lstrip("json").strip()
        return json.loads(raw)
    except Exception as e:
        return {"_error": str(e)}


# ── Performance review ────────────────────────────────────────────────────────

def get_performance_insights(trade_history: list,
                             ai_accuracy: dict | None = None) -> dict:
    """
    Ask Claude Sonnet to analyze closed trade history and surface patterns.
    Only runs when called explicitly (e.g. via /api/ai-insights).
    Requires at least 10 closed trades.

    Returns: {overall_assessment, patterns, recommendations, watch_out} or {}
    """
    client = _get_client()
    if not client:
        return {}

    closed = [t for t in trade_history if t.get("status") in ("win", "loss")]
    if len(closed) < 10:
        return {"error": f"Need at least 10 closed trades (have {len(closed)})"}

    wins   = [t for t in closed if t.get("status") == "win"]
    losses = [t for t in closed if t.get("status") == "loss"]

    # Per-symbol breakdown
    by_sym = {}
    for t in closed:
        s = t.get("symbol", "?")
        if s not in by_sym:
            by_sym[s] = {"wins": 0, "losses": 0, "total_roi": 0.0}
        if t.get("status") == "win":
            by_sym[s]["wins"] += 1
        else:
            by_sym[s]["losses"] += 1
        by_sym[s]["total_roi"] += float(t.get("roi_pct") or 0)

    # Factor win rates
    factor_wins = {}
    for t in closed:
        snap = t.get("factors_snapshot") or {}
        if isinstance(snap, str):
            try:
                snap = json.loads(snap)
            except Exception:
                snap = {}
        for f, present in snap.items():
            if f not in factor_wins:
                factor_wins[f] = {"wins": 0, "losses": 0}
            if present:
                if t.get("status") == "win":
                    factor_wins[f]["wins"] += 1
                else:
                    factor_wins[f]["losses"] += 1

    summary = {
        "total_trades":  len(closed),
        "win_rate_pct":  round(len(wins) / len(closed) * 100, 1),
        "avg_roi_pct":   round(sum(t.get("roi_pct") or 0 for t in closed) / len(closed), 2),
        "best_roi":      round(max((t.get("roi_pct") or 0) for t in closed), 2),
        "worst_roi":     round(min((t.get("roi_pct") or 0) for t in closed), 2),
        "per_symbol":    by_sym,
        "factor_stats":  factor_wins,
    }

    recent_fields = ["symbol", "interval", "direction", "score", "roi_pct", "status", "reason"]
    recent = [{k: t.get(k) for k in recent_fields} for t in closed[-20:]]

    # ── Claude's own prediction accuracy ──────────────────────────────────────
    acc_parts = []
    if ai_accuracy:
        for rec in ("strong_take", "take", "skip"):
            s = ai_accuracy.get(rec)
            if s and s["total"] >= 1:
                acc_parts.append(
                    f"  {rec}: {s['total']} evaluated → "
                    f"{s['wins']}W / {s['losses']}L "
                    f"({s['win_rate_pct']:.0f}% win rate, avg ROI {s['avg_roi_pct']:+.2f}%)"
                )
    acc_str = (
        "\n".join(acc_parts)
        if acc_parts
        else "  No AI-evaluated closed trades yet."
    )

    prompt = f"""Analyze this crypto algorithmic trading system's performance. Be specific and actionable.

PERFORMANCE SUMMARY:
{json.dumps(summary, indent=2)}

RECENT TRADES (last 20):
{json.dumps(recent, indent=2)}

CLAUDE AI PREDICTION ACCURACY (your own past recommendations on these trades):
{acc_str}

Respond with this exact JSON:
{{"overall_assessment": "<1-2 sentences on system health>", "patterns": ["<observed pattern with evidence>"], "recommendations": ["<specific actionable change>"], "watch_out": ["<specific warning or risk>"]}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(msg.content[0].text.strip())
    except Exception:
        return {}
