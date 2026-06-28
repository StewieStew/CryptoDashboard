"""
CEO AGENT — runs once per analyst cycle, after analyst, before risk/trade_manager.

Acts as the actual desk head: not just filtering signals but managing the book.
Uses Anthropic native tool use so Claude explicitly calls each action.

Tools available to CEO:
  approve_trade   — forward signal to Render executor
  reject_trade    — block signal, log reason
  cancel_trade    — cancel open/pending trade in DB (POST /api/trades/<id>/cancel)
  update_config   — adjust signal_threshold or stop_multiplier (POST /api/admin/set_weights)
  hold_cycle      — suppress trade_manager this cycle (e.g. flash crash, news risk)
  reanalyze       — force analyst re-scan immediately (once per outer cycle)
  report_to_desk  — send message to Discord (always called last)

Return value used by orchestrator:
  {
    "approved_trades": [...],
    "hold_cycle":      bool,   # suppresses trade_manager
    "reanalyze":       bool,   # triggers immediate analyst re-run
    "message_to_user": str,
  }
"""
from __future__ import annotations
import os
from datetime import datetime, timezone

import anthropic
import requests

from agents.state import get_state, post_to_render, log_ceo_decision

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_URL   = os.environ.get("DISCORD_WEBHOOK_URL", "")
RENDER_URL    = os.environ.get("RENDER_URL", "http://localhost:8080")
MIN_RR        = 1.5  # CEO's R:R floor — stricter than analyst's 1.2

# ── Tool definitions ──────────────────────────────────────────────────────────

CEO_TOOLS = [
    {
        "name": "approve_trade",
        "description": (
            "Forward an analyst signal to the trade executor. "
            "Use when R:R ≥ 1.5, macro aligned or neutral, no duplicate position open. "
            "Copy the full signal object exactly as received from the analyst."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "signal": {
                    "type": "object",
                    "description": "Full trade signal from analyst — include all fields"
                },
                "reason": {
                    "type": "string",
                    "description": "Why you're approving this — 1-2 sentences"
                }
            },
            "required": ["signal", "reason"]
        }
    },
    {
        "name": "reject_trade",
        "description": (
            "Block an analyst signal. "
            "Use when R:R < 1.5, macro strongly opposed, duplicate already open, "
            "or setup_quality is 'marginal' with no forced-entry obligation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol":    {"type": "string",  "description": "e.g. ETHUSDT"},
                "direction": {"type": "string",  "enum": ["LONG", "SHORT"]},
                "rr_ratio":  {"type": "number",  "description": "Analyst's R:R"},
                "reason":    {"type": "string",  "description": "Why rejected — 1 sentence"}
            },
            "required": ["symbol", "direction", "reason"]
        }
    },
    {
        "name": "cancel_trade",
        "description": (
            "Cancel an OPEN or PENDING trade in the database right now. "
            "Use when: macro regime flipped against the trade, news risk materialised, "
            "entry level was invalidated before fill, or the setup thesis is broken. "
            "This is irreversible — the trade is gone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trade_id":  {"type": "string", "description": "Trade ID from open_trades list"},
                "symbol":    {"type": "string"},
                "direction": {"type": "string"},
                "reason":    {"type": "string", "description": "Why cancelling — 1-2 sentences"}
            },
            "required": ["trade_id", "symbol", "reason"]
        }
    },
    {
        "name": "update_config",
        "description": (
            "Adjust a bot parameter for the current session. "
            "signal_threshold (float 5–9): score gate — higher = fewer, higher-quality trades. "
            "stop_multiplier (float 0.5–2.5): stop width — higher = wider stops, less noise. "
            "Use conservatively: only when market conditions clearly warrant a change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key":   {
                    "type": "string",
                    "enum": ["signal_threshold", "stop_multiplier"],
                    "description": "Parameter name"
                },
                "value":  {"type": "number", "description": "New value"},
                "reason": {"type": "string", "description": "Market condition justifying this"}
            },
            "required": ["key", "value", "reason"]
        }
    },
    {
        "name": "hold_cycle",
        "description": (
            "Suppress trade_manager execution this cycle — no new entries, position management paused. "
            "Risk agent still runs to watch open trades. "
            "Use when: BTC just moved 5%+ in one candle (spreads wide, liquidity thin), "
            "known news event in next 30 min, or market is clearly in a stop-hunt with no directional edge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Specific condition — e.g. 'BTC -4.8% in 15 min, spreads too wide'"
                },
                "resume_in_minutes": {
                    "type": "integer",
                    "description": "Optional estimate of when conditions may normalise"
                }
            },
            "required": ["reason"]
        }
    },
    {
        "name": "reanalyze",
        "description": (
            "Request an immediate analyst re-scan with fresh price data. "
            "Use ONLY when the analyst found borderline setups and 1-2 more 15m candles "
            "would materially clarify the picture. Adds ~2 min and one more Haiku call. "
            "Do NOT use as a fishing expedition — only if a specific condition was borderline. "
            "Happens once per cycle maximum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why fresh data would help"},
                "focus":  {
                    "type": "string",
                    "description": "Optional — what to watch (e.g. 'ETH RSI divergence at 1566 support')"
                }
            },
            "required": ["reason"]
        }
    },
    {
        "name": "report_to_desk",
        "description": (
            "Send your decision summary to the Discord trading desk. "
            "ALWAYS call this as your final action. Be direct and specific."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "Direct message to the desk owner. Examples: "
                        "'Opened ETH LONG limit $3,420 → $3,520 (+2.9%), R:R 2.4:1. BTC macro bullish.' | "
                        "'Cancelled BTC SHORT — regime just flipped bullish on macro reread.' | "
                        "'Held cycle: BTC -4.8% in 15 min, spreads too wide. Waiting for stabilisation.' | "
                        "'No trade: 2 signals but both under 1.5 R:R. Raised threshold to 8.0 — bear regime.'"
                    )
                }
            },
            "required": ["message"]
        }
    }
]


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _cancel_trade_api(trade_id: str) -> bool:
    try:
        r = requests.post(f"{RENDER_URL}/api/trades/{trade_id}/cancel", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _update_config_api(key: str, value: float) -> bool:
    try:
        r = requests.post(f"{RENDER_URL}/api/admin/set_weights",
                          json={key: value}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _manual_fallback(signals: list, open_trades: list, forced: bool
                     ) -> tuple[list, list]:
    """Rule-based filter used when Claude is unavailable."""
    approved, rejected = [], []
    for s in signals:
        rr   = float(s.get("rr_ratio", 0))
        sym  = s.get("symbol", "")
        dirn = s.get("direction", "")
        dupe = any(t.get("symbol") == sym and t.get("direction") == dirn
                   for t in open_trades)
        if dupe:
            rejected.append({"symbol": sym, "direction": dirn, "rr_ratio": rr,
                              "reason": "duplicate — same coin+direction already open"})
        elif rr < MIN_RR:
            rejected.append({"symbol": sym, "direction": dirn, "rr_ratio": rr,
                              "reason": f"R:R {rr:.1f}:1 below CEO floor {MIN_RR}:1"})
        else:
            approved.append(s)
    if not approved and forced and signals:
        best = max(signals, key=lambda s: float(s.get("rr_ratio", 0)))
        approved = [best]
        rejected  = [r for r in rejected if r.get("symbol") != best.get("symbol")]
    return approved, rejected


# ── Main entry point ─────────────────────────────────────────────────────────

def run(forced: bool = False, hours_since_trade: float = 0.0,
        is_reanalysis: bool = False) -> dict:
    """
    Run CEO agent. Returns:
      approved_trades — signals forwarded to Render executor
      hold_cycle      — True → orchestrator suppresses trade_manager
      reanalyze       — True → orchestrator forces analyst re-run immediately
      message_to_user — text posted to Discord
    """
    analyst_output = get_state("analyst_ratings", {})
    macro          = get_state("macro_regime",    {})
    risk_status    = get_state("risk_status",     {})

    open_trades, all_trades = [], []
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
        if r.status_code == 200:
            all_trades  = r.json()
            open_trades = [t for t in all_trades
                           if t.get("status") in ("open", "pending")]
    except Exception:
        pass

    closed     = [t for t in all_trades if t.get("status") in ("win", "loss")]
    wins       = sum(1 for t in closed if t.get("status") == "win")
    recent_pnl = sum(float(t.get("roi_pct") or 0) for t in closed[-10:])
    win_rate   = wins / len(closed) * 100 if closed else 0
    account    = {
        "total_trades":     len(closed),
        "win_rate_pct":     round(win_rate, 1),
        "recent_pnl_pct":   round(recent_pnl, 2),
        "portfolio_health": risk_status.get("portfolio_health", "unknown"),
    }

    signals = analyst_output.get("all_signals", [])

    if not signals and not forced:
        print("  [CEO] No analyst signals — skipping.", flush=True)
        return {"approved_trades": [], "hold_cycle": False,
                "reanalyze": False, "message_to_user": ""}

    # ── Build briefing ────────────────────────────────────────────────────────
    macro_regime = macro.get("regime_type", "uncertain")
    macro_risk   = macro.get("risk_level",  "medium")
    fear_greed   = macro.get("fear_greed", {}).get("current", {})
    mkt_summary  = analyst_output.get("market_summary", "")

    sig_lines = []
    for s in signals:
        rr = float(s.get("rr_ratio", 0))
        sig_lines.append(
            f"  {s.get('symbol','').replace('USDT','')} {s.get('direction','')} "
            f"[{s.get('timeframe','')}]  entry={s.get('entry')}  "
            f"TP={s.get('tp')}  SL={s.get('sl')}  R:R={rr:.1f}:1  "
            f"conf={s.get('confidence',0)}/10  quality={s.get('setup_quality','')}  "
            f"type={s.get('entry_type','limit')}\n"
            f"    reason: {s.get('reason','')[:120]}"
        )

    open_lines = [
        f"  id={t.get('id','?')}  "
        f"{t.get('symbol','').replace('USDT','')} {t.get('direction','')} "
        f"@ {t.get('entry')}  (TP={t.get('tp')}  SL={t.get('sl')}  "
        f"status={t.get('status','')})"
        for t in open_trades
    ]

    forced_note = ""
    if forced and hours_since_trade >= 4:
        forced_note = (
            f"\n\n⚠️  FORCED ENTRY — {hours_since_trade:.1f}h without a trade. "
            "You MUST call approve_trade for at least one signal. "
            "If all are below R:R floor, approve the best one and note it's a forced entry."
        )
    if is_reanalysis:
        forced_note += "\n\nNOTE: This is a re-analysis cycle requested by you. Fresh data is in the signals above."

    briefing = f"""You are the CEO of a paper crypto trading desk. Reports are in. Make your calls.

MACRO:
  Regime: {macro_regime}  |  Risk: {macro_risk}
  Fear & Greed: {fear_greed.get('value','?')} ({fear_greed.get('label','?')})

ANALYST ({len(signals)} signal(s)):
  Market: {mkt_summary or 'not provided'}
{chr(10).join(sig_lines) if sig_lines else '  (none)'}

OPEN BOOK ({len(open_trades)} position(s)):
{chr(10).join(open_lines) if open_lines else '  (none)'}

ACCOUNT:
  {account['total_trades']} closed  |  {account['win_rate_pct']:.0f}% win rate  |  Recent 10-trade P&L: {account['recent_pnl_pct']:+.1f}%
  Portfolio health: {account['portfolio_health']}{forced_note}

YOUR RULES:
1. R:R ≥ 1.5:1 to approve (analyst floor was 1.2 — you're stricter)
2. No duplicate coin+direction in open book
3. Strong macro opposition → require R:R ≥ 2.0 and confidence ≥ 7
4. Loss streak (health=loss_streak) → require R:R ≥ 2.0 and confidence ≥ 7
5. Forced mode: MUST approve at least one signal

Use your tools to act. Call report_to_desk last with your full cycle summary."""

    # ── Claude tool-use call ──────────────────────────────────────────────────
    approved_trades: list = []
    rejected_trades: list = []
    hold_cycle:  bool = False
    reanalyze:   bool = False
    message_to_user: str = ""

    if not ANTHROPIC_KEY:
        approved_trades, rejected_trades = _manual_fallback(signals, open_trades, forced)
        if approved_trades:
            t = approved_trades[0]
            message_to_user = (
                f"{'Forced entry: ' if forced else ''}"
                f"{t.get('symbol','').replace('USDT','')} {t.get('direction','')} "
                f"@ ${t.get('entry')}  R:R {t.get('rr_ratio',0):.1f}:1  (no API key — manual rules)"
            )
        else:
            message_to_user = "No trade — signals below CEO floor. (no API key)"
    else:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                tools=CEO_TOOLS,
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": briefing}],
            )

            for block in response.content:
                if block.type != "tool_use":
                    continue
                name  = block.name
                inp   = block.input

                # ── approve_trade ────────────────────────────────────────────
                if name == "approve_trade":
                    signal = inp.get("signal", {})
                    reason = inp.get("reason", "")
                    if signal and isinstance(signal, dict):
                        approved_trades.append(signal)
                        sym = signal.get("symbol", "").replace("USDT", "")
                        rr  = float(signal.get("rr_ratio", 0))
                        print(f"  [CEO] ✓ approve  {sym} {signal.get('direction','')} "
                              f"R:R={rr:.1f}:1", flush=True)
                        log_ceo_decision(
                            symbol=signal.get("symbol", ""),
                            decision="approve_trade",
                            reason=reason[:200],
                            approved_count=0, rejected_count=0,
                            message="",
                            action_data=f"rr={rr:.1f} entry={signal.get('entry')} "
                                        f"tp={signal.get('tp')} sl={signal.get('sl')}",
                        )

                # ── reject_trade ─────────────────────────────────────────────
                elif name == "reject_trade":
                    sym    = inp.get("symbol", "")
                    dirn   = inp.get("direction", "")
                    rr     = float(inp.get("rr_ratio", 0))
                    reason = inp.get("reason", "")
                    rejected_trades.append({"symbol": sym, "direction": dirn,
                                            "rr_ratio": rr, "reason": reason})
                    print(f"  [CEO] ✗ reject   {sym.replace('USDT','')} {dirn} "
                          f"— {reason}", flush=True)
                    log_ceo_decision(
                        symbol=sym, decision="reject_trade",
                        reason=reason[:200],
                        approved_count=0, rejected_count=0, message="",
                        action_data=f"rr={rr:.1f}",
                    )

                # ── cancel_trade ─────────────────────────────────────────────
                elif name == "cancel_trade":
                    trade_id = str(inp.get("trade_id", ""))
                    sym      = inp.get("symbol", "")
                    dirn     = inp.get("direction", "")
                    reason   = inp.get("reason", "")
                    ok = _cancel_trade_api(trade_id)
                    status = "cancelled" if ok else "CANCEL FAILED"
                    print(f"  [CEO] ⊗ cancel   {sym.replace('USDT','')} {dirn} "
                          f"id={trade_id} — {status}: {reason}", flush=True)
                    log_ceo_decision(
                        symbol=sym, decision="cancel_trade",
                        reason=reason[:200],
                        approved_count=0, rejected_count=0, message="",
                        action_data=f"trade_id={trade_id} status={status}",
                    )

                # ── update_config ────────────────────────────────────────────
                elif name == "update_config":
                    key    = inp.get("key", "")
                    value  = float(inp.get("value", 0))
                    reason = inp.get("reason", "")
                    ok = _update_config_api(key, value)
                    status = "updated" if ok else "UPDATE FAILED"
                    print(f"  [CEO] ⚙ config   {key}={value} — {status}: {reason}",
                          flush=True)
                    log_ceo_decision(
                        symbol="", decision="update_config",
                        reason=reason[:200],
                        approved_count=0, rejected_count=0, message="",
                        action_data=f"key={key} value={value} status={status}",
                    )

                # ── hold_cycle ───────────────────────────────────────────────
                elif name == "hold_cycle":
                    hold_cycle = True
                    reason     = inp.get("reason", "")
                    resume     = inp.get("resume_in_minutes")
                    note = f" (~{resume} min)" if resume else ""
                    print(f"  [CEO] ⏸ HOLD CYCLE{note} — {reason}", flush=True)
                    log_ceo_decision(
                        symbol="", decision="hold_cycle",
                        reason=reason[:200],
                        approved_count=0, rejected_count=0, message="",
                        action_data=f"resume_in={resume}",
                    )

                # ── reanalyze ────────────────────────────────────────────────
                elif name == "reanalyze":
                    if not is_reanalysis:  # prevent recursion
                        reanalyze = True
                        reason    = inp.get("reason", "")
                        focus     = inp.get("focus", "")
                        print(f"  [CEO] ↺ REANALYZE — {reason}"
                              f"{f' | focus: {focus}' if focus else ''}", flush=True)
                        log_ceo_decision(
                            symbol="", decision="reanalyze",
                            reason=reason[:200],
                            approved_count=0, rejected_count=0, message="",
                            action_data=f"focus={focus[:100]}",
                        )
                    else:
                        print("  [CEO] reanalyze suppressed (already a re-analysis cycle)",
                              flush=True)

                # ── report_to_desk ───────────────────────────────────────────
                elif name == "report_to_desk":
                    message_to_user = inp.get("message", "")

        except Exception as e:
            print(f"  [CEO] Claude error: {e} — applying manual rules", flush=True)
            approved_trades, rejected_trades = _manual_fallback(signals, open_trades, forced)
            message_to_user = (
                f"CEO fallback (error): {len(approved_trades)} approved, "
                f"{len(rejected_trades)} rejected."
            )

    # ── Synthesise message if report_to_desk wasn't called ───────────────────
    if not message_to_user:
        parts = []
        for t in approved_trades:
            sym = t.get("symbol", "").replace("USDT", "")
            rr  = t.get("rr_ratio", 0)
            et  = t.get("entry_type", "limit")
            parts.append(f"{sym} {t.get('direction','')} {et} @ ${t.get('entry')} R:R {rr:.1f}:1")
        if hold_cycle:
            parts.append("[cycle held]")
        if reanalyze:
            parts.append("[re-scan requested]")
        if not parts and rejected_trades:
            top = rejected_trades[0]
            sym = str(top.get("symbol", "")).replace("USDT", "")
            parts.append(f"No trade — {sym}: {top.get('reason','')}")
        message_to_user = "CEO: " + " | ".join(parts) if parts else "CEO: no action this cycle."

    # ── Update counts on logged decisions ────────────────────────────────────
    n_approved = len(approved_trades)
    n_rejected = len(rejected_trades)

    # ── Post approved signals to Render executor ──────────────────────────────
    if approved_trades:
        primary = max(approved_trades, key=lambda t: float(t.get("confidence", 0)))
        post_to_render("/api/agent/insight", {
            "type":          "analyst_signal",
            "agent":         "ceo",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "trade_signal":  primary,
            "all_signals":   approved_trades,
            "market_summary": analyst_output.get("market_summary"),
            "coins_to_avoid": analyst_output.get("coins_to_avoid", []),
            "ceo_approved":  True,
            "forced":        forced,
        })
        print(f"  [CEO] {n_approved} signal(s) forwarded to Render executor.", flush=True)

    # ── Final log entry (cycle-level summary) ────────────────────────────────
    log_ceo_decision(
        symbol="",
        decision="cycle_summary",
        reason=f"approved={n_approved} rejected={n_rejected} hold={hold_cycle} reanalyze={reanalyze}",
        approved_count=n_approved,
        rejected_count=n_rejected,
        message=message_to_user[:400],
        action_data="",
    )

    # ── Discord ───────────────────────────────────────────────────────────────
    if message_to_user:
        if hold_cycle:
            color = 0xffa500   # orange — hold
        elif forced and approved_trades:
            color = 0xff9800   # amber  — forced entry
        elif approved_trades:
            color = 0x57f287   # green  — normal approval
        else:
            color = 0x5865F2   # indigo — hold / no trade
        _discord(message_to_user, color=color)

    print(f"  [CEO] Done — approved={n_approved} rejected={n_rejected} "
          f"hold={hold_cycle} reanalyze={reanalyze}", flush=True)

    return {
        "approved_trades":  approved_trades,
        "rejected_trades":  rejected_trades,
        "hold_cycle":       hold_cycle,
        "reanalyze":        reanalyze,
        "message_to_user":  message_to_user,
        "approved_count":   n_approved,
        "rejected_count":   n_rejected,
    }
