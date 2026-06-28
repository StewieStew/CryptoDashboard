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
  RENDER_URL          — default: http://localhost:8080
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
# Also check ~/CryptoDashboard/.env for when running via LaunchAgent (no Desktop TCC)
for _env_candidate in [
    Path.home() / "Desktop" / "CryptoDashboard" / ".env",
    Path.home() / "CryptoDashboard" / ".env",
]:
    try:
        if _env_candidate.exists():
            for _line in _env_candidate.read_text().splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
            break
    except (PermissionError, OSError):
        pass  # Desktop not readable from LaunchAgent context; env vars set via plist

# ── Import all agents ─────────────────────────────────────────────────────────
from agents import state as S
from agents import macro_agent, analyst_agent, risk_agent, learning_agent, trade_manager_agent


RENDER_URL    = os.environ.get("RENDER_URL", "http://localhost:8080")
DISCORD_URL   = os.environ.get("DISCORD_WEBHOOK_URL", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

LOG_DIR = Path.home() / "CryptoDashboard" / "agent_kb"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Schedule config ───────────────────────────────────────────────────────────
MACRO_INTERVAL     = 60 * 60   # 60 minutes
ANALYST_INTERVAL   = 15 * 60   # 15 minutes — catches 15m setups quickly
HTF_INTERVAL       = 30 * 60   # 30 minutes — 1h/4h setups change more slowly
RISK_INTERVAL      =  5 * 60   #  5 minutes (Haiku — cheap, keep fast)
TRADE_MGR_INTERVAL =  5 * 60   #  5 minutes — BE stops, partials, trails
LEARNING_INTERVAL  = 60 * 60   # 60 minutes

_last_run = {
    "macro":        0,
    "analyst":      0,
    "analyst_htf":  0,  # last time 1h/4h setups were checked
    "risk":         0,
    "trade_mgr":    0,
    "learning":     0,
}

# Forced 15m entry: tracks when we last saw at least one open trade (any timeframe)
_last_15m_check_time = 0.0
_had_open_since      = time.time()  # start clock from now; resets when open/pending trade exists


def _check_15m_forced() -> bool:
    """Return True if 4+ hours have passed with zero open trades (any timeframe)."""
    global _last_15m_check_time, _had_open_since
    now = time.time()
    if now - _last_15m_check_time < 300:  # cache for 5 minutes
        return now - _had_open_since >= 4 * 3600
    try:
        r = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
        if r.status_code == 200:
            has_open = any(
                isinstance(t, dict) and t.get("status") in ("open", "pending")
                for t in r.json()
            )
            if has_open:
                _had_open_since = now  # reset: we have at least one open or pending trade
    except Exception:
        pass
    _last_15m_check_time = now
    return now - _had_open_since >= 4 * 3600


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
    global _had_open_since
    log("═" * 65)
    log("  CRYPTO TRADING DESK — MULTI-AGENT SYSTEM STARTING")
    log(f"  Render: {RENDER_URL}")
    log(f"  Claude: {'configured ✓' if ANTHROPIC_KEY else 'NOT SET ✗'}")
    log(f"  Discord: {'configured ✓' if DISCORD_URL else 'not configured'}")
    log("═" * 65)
    log("  AGENTS:")
    log(f"  • MACRO     — runs every {MACRO_INTERVAL//60}min  — news, sentiment, regime")
    log(f"  • ANALYST   — runs every {ANALYST_INTERVAL//60}min  — pre-scan (Python) → Claude only when setup found (forced every 4h)")
    log(f"  • RISK      — runs every {RISK_INTERVAL//60}min   — monitors open trades, flags problems")
    log(f"  • TRADE MGR — runs every {TRADE_MGR_INTERVAL//60}min   — BE stops, partial profits, trailing stops")
    log(f"  • LEARNING  — runs every {LEARNING_INTERVAL//60}min — post-mortems, improvements")
    log("═" * 65)

    if not startup_checks():
        log("Startup checks failed. Exiting.", "ERROR")
        return

    discord(
        "Trading desk is online. All agents starting.\n"
        f"Monitoring: BTC, ETH, XRP, DOGE, SOL\n"
        f"Forced 15m entry: every 4 hours guaranteed\n"
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
            run_agent_safe("macro", macro_agent.run, MACRO_INTERVAL)

            # Check 4h forced 15m entry timer — if expired, force an early analyst run
            _forced_15m = _check_15m_forced()
            if _forced_15m:
                elapsed_h = (now - _had_open_since) / 3600
                log(f"[FORCED ENTRY] No open trades for {elapsed_h:.1f}h — analyst MUST take a 15m trade", "WARN")
                if not should_run("analyst", ANALYST_INTERVAL):
                    _last_run["analyst"] = 0  # trigger early analyst run

            # Snapshot open/pending trade IDs before analyst runs so we can detect
            # whether a new trade actually appeared after a forced entry cycle.
            _open_trade_ids_before: set = set()
            if _forced_15m:
                try:
                    _snap = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
                    if _snap.status_code == 200:
                        _open_trade_ids_before = {
                            t["id"] for t in _snap.json()
                            if isinstance(t, dict) and t.get("status") in ("open", "pending")
                        }
                except Exception:
                    pass

            # Include 1h/4h setup checks only every 30 min (15m checks run every cycle)
            _analyst_ran_before = _last_run["analyst"]
            _include_htf = (now - _last_run["analyst_htf"] >= HTF_INTERVAL)
            run_agent_safe("analyst",
                           lambda: analyst_agent.run(forced=_forced_15m,
                                                     include_htf=_include_htf),
                           ANALYST_INTERVAL)
            if _last_run["analyst"] != _analyst_ran_before and _include_htf:
                _last_run["analyst_htf"] = _last_run["analyst"]
            # After a forced analyst run, only restart the 4h clock if a new trade
            # actually appeared in the DB. If no trade was placed (e.g. the analyst
            # returned a limit order that was rejected or Claude returned nothing),
            # keep the clock running so the next cycle tries again.
            if _forced_15m and _last_run["analyst"] != _analyst_ran_before:
                try:
                    _snap2 = requests.get(f"{RENDER_URL}/api/trades", timeout=10)
                    if _snap2.status_code == 200:
                        _open_trade_ids_after = {
                            t["id"] for t in _snap2.json()
                            if isinstance(t, dict) and t.get("status") in ("open", "pending")
                        }
                        if _open_trade_ids_after - _open_trade_ids_before:
                            _had_open_since = time.time()
                            log("Forced entry placed — new trade confirmed, 4h timer restarted", "INFO")
                        else:
                            log("Forced analyst ran but no new trade appeared — 4h timer continues", "WARN")
                    else:
                        _had_open_since = time.time()
                        log("Forced entry attempted — 4h timer restarted (API unavailable)", "INFO")
                except Exception:
                    _had_open_since = time.time()
                    log("Forced entry attempted — 4h timer restarted (could not verify trade)", "INFO")
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
