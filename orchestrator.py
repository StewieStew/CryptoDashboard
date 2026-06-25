#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  TRADING DESK ORCHESTRATOR — runs on Mac Mini 24/7
  Coordinates all agents like a professional trading desk.
═══════════════════════════════════════════════════════════════════════════════

THE TRADING DESK:

  MACRO AGENT    (every 30 min) — big picture: news, sentiment, regime
  ANALYST AGENT  (every 15 min) — price action, levels, liquidity, coin ratings
  RISK AGENT     (every 5 min)  — monitors open trades, flags problems
  LEARNING AGENT (every 60 min) — post-mortems, patterns, improvements

All agents share state via SQLite. Each posts findings to Render.
The Render bot reads agent intelligence before every trade entry.

Set environment variables:
  ANTHROPIC_API_KEY   — required
  RENDER_URL          — default: https://cryptodashboard-nuf5.onrender.com
  DISCORD_WEBHOOK_URL — optional, for desk briefings

Run:         python3 orchestrator.py
Auto-start:  double-click install_agent_autostart.command
"""
from __future__ import annotations

import os, time, traceback, json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ── Load .env file if present (picks up TG credentials etc.) ──────────────────
_env_file = Path.home() / "Desktop" / "CryptoDashboard" / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Import all agents ─────────────────────────────────────────────────────────
from agents import state as S
from agents import macro_agent, analyst_agent, risk_agent, learning_agent, trade_manager_agent

# Telegram agent — optional, only active if TG_API_ID env var is set
try:
    from agents import telegram_agent as _tg_agent
    _TG_ENABLED = bool(os.environ.get("TG_API_ID"))
except ImportError:
    _tg_agent   = None
    _TG_ENABLED = False

RENDER_URL    = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")
DISCORD_URL   = os.environ.get("DISCORD_WEBHOOK_URL", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

LOG_DIR = Path.home() / "CryptoDashboard" / "agent_kb"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Schedule config ───────────────────────────────────────────────────────────
MACRO_INTERVAL     = 60 * 60   # 60 minutes
ANALYST_INTERVAL   = 30 * 60   # 30 minutes — quality setups, not frequency
RISK_INTERVAL      =  5 * 60   #  5 minutes (Haiku — cheap, keep fast)
TRADE_MGR_INTERVAL =  5 * 60   #  5 minutes — BE stops, partials, trails
LEARNING_INTERVAL  = 60 * 60   # 60 minutes
TELEGRAM_INTERVAL  = 30 * 60   # 30 minutes

_last_run = {
    "macro":       0,
    "analyst":     0,
    "risk":        0,
    "trade_mgr":   0,
    "learning":    0,
    "telegram":    0,
}


def log(msg: str, level: str = "INFO") -> None:
    ts   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    # Clean readable format — no JSON noise, just what matters
    if level == "ERROR":
        line = f"  !! {ts}  {msg}"
    elif level == "WARN":
        line = f"  ?? {ts}  {msg}"
    else:
        line = f"  {ts}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG_DIR / "orchestrator.log", "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_divider(label: str = "") -> None:
    """Print a clean section divider."""
    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    if label:
        pad = "─" * max(0, 52 - len(label))
        print(f"\n  ┌─ {label} {pad}", flush=True)
    else:
        print(f"  └{'─'*55}", flush=True)


def discord(content: str, title: str = "", color: int = 0x5865F2) -> None:
    if not DISCORD_URL:
        return
    try:
        requests.post(DISCORD_URL, json={
            "embeds": [{
                "title":       title or "Trading Desk",
                "description": content[:3900],
                "color":       color,
                "footer":      {"text": f"Mac Mini Trading Desk • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"},
            }]
        }, timeout=8)
    except Exception:
        pass


def should_run(agent: str, interval: int) -> bool:
    return time.time() - _last_run.get(agent, 0) >= interval


def run_agent_safe(name: str, fn, interval: int) -> None:
    if not should_run(name, interval):
        return
    labels = {
        "macro":     "MACRO    — reading market regime & sentiment",
        "analyst":   "ANALYST  — scanning charts for setups",
        "risk":      "RISK     — checking open trades",
        "trade_mgr": "TRADE MGR — managing open positions (BE/trail/partial)",
        "learning":  "LEARNING — reviewing closed trades",
        "telegram":  "TELEGRAM — listener status",
    }
    log_divider(labels.get(name, name.upper()))
    try:
        result = fn()
        _last_run[name] = time.time()
        log_divider()
        return result
    except Exception as e:
        log(f"{name.upper()} crashed: {e}", "ERROR")
        log_divider()


def post_desk_briefing() -> None:
    """Post a consolidated desk briefing every hour."""
    macro    = S.get_state("macro_regime", {})
    analyst  = S.get_state("analyst_ratings", {})
    risk     = S.get_state("risk_status", {})
    learning = S.get_state("learning_status", {})

    if not macro:
        return

    ratings  = analyst.get("ratings", {})
    best     = analyst.get("best_setup", "none")
    best_dir = analyst.get("best_direction", "")
    regime   = macro.get("regime_type", "?")
    risk_lvl = macro.get("risk_level", "?")
    alerts   = risk.get("alerts", [])
    open_c   = risk.get("open_count", 0)
    health   = risk.get("portfolio_health", "?")

    rating_str = " | ".join(f"{s.replace('USDT','')}: {v}/10"
                            for s, v in sorted(ratings.items(),
                                               key=lambda x: x[1], reverse=True))

    msg = (
        f"**Regime:** {regime} | Risk: {risk_lvl}\n"
        f"**Portfolio:** {open_c} open trades | Health: {health}\n"
        f"**Coin ratings:** {rating_str}\n"
        f"**Best setup:** {best} {best_dir}\n"
        f"**Alerts:** {len(alerts)} flagged trades\n"
    )

    if analyst.get("best_reason"):
        msg += f"**Why:** {analyst.get('best_reason','')[:150]}\n"

    # Also post to Render
    S.post_to_render("/api/agent/insight", {
        "type":       "desk_briefing",
        "agent":      "orchestrator",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "regime":     regime,
        "risk_level": risk_lvl,
        "ratings":    ratings,
        "best_setup": best,
        "best_direction": best_dir,
        "open_trades": open_c,
        "alerts":     len(alerts),
        "health":     health,
    })

    discord(msg, "📊 Hourly Desk Briefing", color=0x00b4d8)


def startup_checks() -> bool:
    """Verify everything is set up correctly."""
    log("Running startup checks...")

    if not ANTHROPIC_KEY:
        log("ERROR: ANTHROPIC_API_KEY not set!", "ERROR")
        return False

    # Test Render connection
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
        if r.status_code == 200:
            trade_count = len(r.json())
            log(f"Render connected — {trade_count} trades in DB")
        else:
            log(f"Render returned {r.status_code}", "WARN")
    except Exception as e:
        log(f"Render connection failed: {e}", "WARN")

    S.init_db()
    log("State DB initialized")
    return True


def main():
    log("═" * 65)
    log("  CRYPTO TRADING DESK — MULTI-AGENT SYSTEM STARTING")
    log(f"  Render: {RENDER_URL}")
    log(f"  Claude: {'configured ✓' if ANTHROPIC_KEY else 'NOT SET ✗'}")
    log(f"  Discord: {'configured ✓' if DISCORD_URL else 'not configured'}")
    log("═" * 65)
    log("  AGENTS:")
    log(f"  • MACRO     — runs every {MACRO_INTERVAL//60}min  — news, sentiment, regime")
    log(f"  • ANALYST   — runs every {ANALYST_INTERVAL//60}min  — price action, levels, charts, ratings")
    log(f"  • TELEGRAM  — REAL-TIME listener — {'ENABLED' if _TG_ENABLED else 'DISABLED (run setup_telegram.command)'}")
    log(f"  • RISK      — runs every {RISK_INTERVAL//60}min   — monitors open trades, flags problems")
    log(f"  • TRADE MGR — runs every {TRADE_MGR_INTERVAL//60}min   — BE stops, partial profits, trailing stops")
    log(f"  • LEARNING  — runs every {LEARNING_INTERVAL//60}min — post-mortems, improvements")
    log("═" * 65)

    if not startup_checks():
        log("Startup checks failed. Exiting.", "ERROR")
        return

    # Start Telegram listener in background thread (real-time, not scheduled)
    if _TG_ENABLED and _tg_agent:
        try:
            started = _tg_agent.start_listener()
            if started:
                log("  Telegram listener started — watching groups in real-time")
            else:
                log("  Telegram listener failed to start — check setup_telegram.command", "WARN")
        except Exception as _tg_err:
            log(f"  Telegram listener error: {_tg_err}", "WARN")

    discord(
        "Trading desk is online. All agents starting.\n"
        f"Monitoring: BTC, ETH, XRP, DOGE, SOL\n"
        f"Telegram: {'real-time' if _TG_ENABLED else 'not configured'}\n"
        f"Render: {RENDER_URL}",
        "🚀 Trading Desk Online",
        color=0x57f287,
    )

    briefing_last = 0

    # Force immediate first run of all agents on startup
    for agent in _last_run:
        _last_run[agent] = 0

    loop_count = 0
    while True:
        loop_count += 1
        try:
            now = time.time()

            # Run agents in order (macro → analyst → risk → learning)
            # Telegram runs in its own background thread — not scheduled here
            run_agent_safe("macro",     macro_agent.run,         MACRO_INTERVAL)
            run_agent_safe("analyst",   analyst_agent.run,       ANALYST_INTERVAL)
            run_agent_safe("risk",      risk_agent.run,          RISK_INTERVAL)
            run_agent_safe("trade_mgr", trade_manager_agent.run, TRADE_MGR_INTERVAL)
            run_agent_safe("learning",  learning_agent.run,      LEARNING_INTERVAL)

            # Hourly desk briefing
            if now - briefing_last >= 3600:
                try:
                    post_desk_briefing()
                    briefing_last = now
                except Exception:
                    pass

        except Exception as e:
            log(f"Orchestrator loop error: {e}", "ERROR")
            log(traceback.format_exc(), "ERROR")

        # Keep Render awake — free tier sleeps after 15 min inactivity.
        # A lightweight GET every 60s ensures the executor thread stays live.
        try:
            requests.get(f"{RENDER_URL}/api/trades", timeout=10)
        except Exception:
            pass

        # Sleep 60 seconds between checks
        time.sleep(60)


if __name__ == "__main__":
    main()
