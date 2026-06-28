"""
CEO AGENT — runs once per analyst cycle, after analyst and before risk/trade_manager.
Job: Read all sub-agent reports and make the final call on which trades to execute.
     Approves, rejects, or holds analyst signals.
     Sends a human-readable decision to Discord.

Uses Claude Haiku to keep costs low (called once per cycle, not per coin).
MIN R:R = 1.5:1 (stricter than analyst's 1.2 floor).
"""
from __future__ import annotations
import json, os
from datetime import datetime, timezone

import anthropic
import requests

from agents.state import get_state, post_to_render, log_ceo_decision

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_URL   = os.environ.get("DISCORD_WEBHOOK_URL", "")
RENDER_URL    = os.environ.get("RENDER_URL", "http://localhost:8080")
MIN_RR        = 1.5  # CEO's R:R floor — stricter than analyst's 1.2


def _discord(content: str, color: int = 0x00b4d8) -> None:
    if not DISCORD_URL:
        return
    try:
        requests.post(DISCORD_URL, json={
            "embeds": [{
                "title":       "🧠 CEO Decision",
                "description": content[:3900],
                "color":       color,
                "footer":      {"text": f"CEO Agent · {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
            }]
        }, timeout=8)
    except Exception:
        pass


def _manual_filter(signals: list, open_trades: list, forced: bool) -> tuple[list, list]:
    """Rule-based fallback when Claude is unavailable."""
    approved, rejected = [], []
    for s in signals:
        rr   = float(s.get("rr_ratio", 0))
        sym  = s.get("symbol", "")
        dirn = s.get("direction", "")
        already_open = any(
            t.get("symbol") == sym and t.get("direction") == dirn
            for t in open_trades
        )
        if already_open:
            rejected.append({"symbol": sym, "direction": dirn, "rr_ratio": rr,
                              "reason": "duplicate — same coin+direction already open"})
        elif rr < MIN_RR:
            rejected.append({"symbol": sym, "direction": dirn, "rr_ratio": rr,
                              "reason": f"R:R {rr:.1f}:1 below CEO floor of {MIN_RR}:1"})
        else:
            approved.append(s)

    # Forced mode: if nothing approved, take the best available regardless of R:R
    if not approved and forced and signals:
        best = max(signals, key=lambda s: float(s.get("rr_ratio", 0)))
        sym  = best.get("symbol", "")
        approved = [best]
        rejected = [r for r in rejected if r.get("symbol") != sym]

    return approved, rejected


def run(forced: bool = False, hours_since_trade: float = 0.0) -> dict:
    """Run CEO agent: review sub-agent reports, decide, report to Discord."""
    analyst_output = get_state("analyst_ratings", {})
    macro          = get_state("macro_regime", {})
    risk_status    = get_state("risk_status", {})

    # Fetch live trades from Render
    open_trades, all_trades = [], []
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
        if r.status_code == 200:
            all_trades  = r.json()
            open_trades = [t for t in all_trades if t.get("status") in ("open", "pending")]
    except Exception:
        pass

    closed    = [t for t in all_trades if t.get("status") in ("win", "loss")]
    wins      = sum(1 for t in closed if t.get("status") == "win")
    recent_pnl = sum(float(t.get("roi_pct") or 0) for t in closed[-10:])
    win_rate   = wins / len(closed) * 100 if closed else 0
    account    = {
        "total_trades":   len(closed),
        "win_rate_pct":   round(win_rate, 1),
        "recent_pnl_pct": round(recent_pnl, 2),
        "portfolio_health": risk_status.get("portfolio_health", "unknown"),
    }

    signals = analyst_output.get("all_signals", [])

    if not signals and not forced:
        print("  [CEO] No analyst signals this cycle — skipping.", flush=True)
        return {"approved_trades": [], "rejected_trades": [], "message_to_user": ""}

    # ── Build briefing ────────────────────────────────────────────────────────
    macro_regime  = macro.get("regime_type", "uncertain")
    macro_risk    = macro.get("risk_level", "medium")
    fear_greed    = macro.get("fear_greed", {}).get("current", {})
    mkt_summary   = analyst_output.get("market_summary", "")

    sig_lines = []
    for s in signals:
        rr = float(s.get("rr_ratio", 0))
        sig_lines.append(
            f"  {s.get('symbol','').replace('USDT','')} {s.get('direction','')} "
            f"[{s.get('timeframe','')}]  entry={s.get('entry')}  TP={s.get('tp')}  SL={s.get('sl')}"
            f"  R:R={rr:.1f}:1  conf={s.get('confidence',0)}/10  quality={s.get('setup_quality','')}"
            f"  type={s.get('entry_type','limit')}"
            f"\n    reason: {s.get('reason','')[:120]}"
        )

    open_lines = [
        f"  {t.get('symbol','').replace('USDT','')} {t.get('direction','')} "
        f"@ {t.get('entry')}  (TP={t.get('tp')}  SL={t.get('sl')})"
        for t in open_trades
    ]

    forced_note = ""
    if forced and hours_since_trade >= 4:
        forced_note = (
            f"\n⚠️  FORCED ENTRY — {hours_since_trade:.1f}h without a trade. "
            "You MUST approve at least one trade. If all signals are below the R:R floor, "
            "approve the single best one anyway and note it's a forced entry."
        )

    prompt = f"""You are the CEO of a paper crypto trading desk. Sub-agents have filed their reports. Make the final call.

MACRO REPORT:
  Regime: {macro_regime}  |  Risk level: {macro_risk}
  Fear & Greed: {fear_greed.get('value','?')} ({fear_greed.get('label','?')})

ANALYST REPORT ({len(signals)} signal(s)):
  Market summary: {mkt_summary or 'Not provided'}
  Signals:
{chr(10).join(sig_lines) if sig_lines else '  None this cycle'}

OPEN POSITIONS ({len(open_trades)}):
{chr(10).join(open_lines) if open_lines else '  None'}

ACCOUNT STATUS:
  Total closed trades: {account['total_trades']}  |  Win rate: {account['win_rate_pct']:.0f}%
  Recent P&L (last 10 trades): {account['recent_pnl_pct']:+.1f}%
  Portfolio health: {account['portfolio_health']}{forced_note}

YOUR DECISION RULES — apply strictly:
1. R:R must be ≥ {MIN_RR}:1 — reject anything below (analyst's floor was 1.2, yours is stricter)
2. No duplicate coin+direction: if same coin+direction is already open, reject
3. If macro is strongly bearish and analyst signals a LONG (or vice versa), be skeptical — require R:R ≥ 2.0 and confidence ≥ 7
4. If portfolio_health is "loss_streak", require confidence ≥ 7 and R:R ≥ 2.0
5. If forced=True (noted above), you MUST approve the single best available signal even if R:R < floor

For approved trades: copy the COMPLETE trade object from the analyst's signal (all fields).
For rejected trades: include symbol, direction, rr_ratio, and a brief reason.

message_to_user must be direct and informative, like a trading desk head briefing the prop firm owner:
  • What was approved: symbol, direction, entry type, price, R:R
  • What was rejected and why (one line)
  • If forced, say "Forced entry: 4h+ idle"
Examples:
  "Opened ETH LONG limit at $3,420, target $3,520 (+2.9%). BTC macro bullish, clean demand zone. R:R 2.4:1."
  "No trade this cycle — BTC ranging, no clean setups. ETH signal had R:R 1.3:1, below our 1.5 floor."
  "Forced entry (4.2h idle): BTC LONG market at $67,400. Not ideal macro but we need activity. R:R 2.1:1."

Respond with ONLY this JSON (no markdown, no code block):
{{
  "approved_trades": [<full trade objects from analyst signals>],
  "rejected_trades": [{{"symbol": "XYZUSDT", "direction": "LONG", "rr_ratio": 1.2, "reason": "..."}}],
  "message_to_user": "<direct natural message>"
}}"""

    approved_trades: list = []
    rejected_trades: list = []
    message_to_user: str  = ""

    if ANTHROPIC_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try to close any open brackets
                raw += "]" * max(raw.count("[") - raw.count("]"), 0)
                raw += "}" * max(raw.count("{") - raw.count("}"), 0)
                parsed = json.loads(raw)

            approved_trades = parsed.get("approved_trades", [])
            rejected_trades = parsed.get("rejected_trades", [])
            message_to_user = parsed.get("message_to_user", "")

        except Exception as e:
            print(f"  [CEO] Claude error: {e} — applying manual rules", flush=True)
            approved_trades, rejected_trades = _manual_filter(signals, open_trades, forced)
            message_to_user = (
                f"CEO fallback (Claude error): {len(approved_trades)} approved, "
                f"{len(rejected_trades)} rejected."
            )
    else:
        # No API key — apply rule-based filter
        approved_trades, rejected_trades = _manual_filter(signals, open_trades, forced)
        if approved_trades:
            t = approved_trades[0]
            message_to_user = (
                f"{'Forced entry: ' if forced else ''}"
                f"{t.get('symbol','').replace('USDT','')} {t.get('direction','')} "
                f"{t.get('entry_type','limit')} @ ${t.get('entry')}, R:R {t.get('rr_ratio',0):.1f}:1."
            )
        else:
            message_to_user = "No trade this cycle — no signals passed CEO R:R filter."

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  [CEO] Approved: {len(approved_trades)}  |  Rejected: {len(rejected_trades)}", flush=True)
    for t in approved_trades:
        sym = str(t.get("symbol", "")).replace("USDT", "")
        rr  = float(t.get("rr_ratio", 0))
        print(f"    ✓  {sym} {t.get('direction','')}  R:R={rr:.1f}:1  {t.get('entry_type','limit')}", flush=True)
    for t in rejected_trades:
        sym = str(t.get("symbol", "")).replace("USDT", "")
        print(f"    ✗  {sym} {t.get('direction','')} — {t.get('reason','')}", flush=True)
    if message_to_user:
        print(f"  [CEO] → {message_to_user}", flush=True)

    # ── Post approved signals to Render so the executor can pick them up ──────
    if approved_trades:
        primary = max(approved_trades, key=lambda t: float(t.get("confidence", 0)))
        post_to_render("/api/agent/insight", {
            "type":          "analyst_signal",  # executor reads this type
            "agent":         "ceo",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "trade_signal":  primary,
            "all_signals":   approved_trades,
            "market_summary": analyst_output.get("market_summary"),
            "coins_to_avoid": analyst_output.get("coins_to_avoid", []),
            "ceo_approved":  True,
            "forced":        forced,
        })
        print(f"  [CEO] {len(approved_trades)} signal(s) forwarded to Render executor.", flush=True)

    # ── Log to ceo_decisions table ────────────────────────────────────────────
    for t in approved_trades:
        log_ceo_decision(
            symbol=str(t.get("symbol", "")),
            decision="approved",
            reason=str(t.get("reason", ""))[:200],
            approved_count=len(approved_trades),
            rejected_count=len(rejected_trades),
            message=message_to_user[:400],
        )
    for t in rejected_trades:
        log_ceo_decision(
            symbol=str(t.get("symbol", "")),
            decision="rejected",
            reason=str(t.get("reason", ""))[:200],
            approved_count=len(approved_trades),
            rejected_count=len(rejected_trades),
            message=message_to_user[:400],
        )
    if not approved_trades and not rejected_trades:
        log_ceo_decision(
            symbol="",
            decision="hold",
            reason="no signals this cycle",
            approved_count=0,
            rejected_count=0,
            message=message_to_user[:400],
        )

    # ── Send Discord ──────────────────────────────────────────────────────────
    if message_to_user:
        if forced and approved_trades:
            color = 0xff9800   # orange for forced entries
        elif approved_trades:
            color = 0x57f287   # green for normal approvals
        else:
            color = 0x5865F2   # blue for hold/no-trade cycles
        _discord(message_to_user, color=color)

    return {
        "approved_trades":  approved_trades,
        "rejected_trades":  rejected_trades,
        "message_to_user":  message_to_user,
        "approved_count":   len(approved_trades),
        "rejected_count":   len(rejected_trades),
    }
