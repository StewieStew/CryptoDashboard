"""
Shared state manager for the trading desk agent system.
All agents read and write through this module.
SQLite-backed so state persists across restarts.
"""
from __future__ import annotations
import sqlite3, json, os
from datetime import datetime, timezone
from pathlib import Path


# ── Trading Session Clock ──────────────────────────────────────────────────────
def get_session() -> dict:
    """
    Return the current trading session and its trading characteristics.
    All times in UTC. Crypto trades 24/7 but session timing matters enormously —
    Asian = choppy range, London/NY = real directional moves with volume.
    """
    hour = datetime.now(timezone.utc).hour  # 0-23 UTC

    # Session boundaries (UTC)
    # Asian:   00:00 – 08:00 UTC  (Tokyo/Singapore — low volume, range-bound)
    # London:  07:00 – 16:00 UTC  (London open at 08:00 = big moves start)
    # NY:      13:00 – 22:00 UTC  (NY open at 13:30 = highest volume overlap)
    # Dead:    22:00 – 00:00 UTC  (post-NY, pre-Asia — avoid)

    if 0 <= hour < 7:
        session = "ASIAN"
        quality = "low"
        note    = ("Asian session — low volume, choppy ranges, frequent fakeouts. "
                   "Avoid breakout trades. Range setups only if very clean structure.")
        caution = True
    elif 7 <= hour < 8:
        session = "LONDON_PRE"
        quality = "medium"
        note    = ("London pre-market — volatility picking up. Watch for London open "
                   "sweep (spike to grab stops) in the first 30 min after 08:00 UTC.")
        caution = False
    elif 8 <= hour < 13:
        session = "LONDON"
        quality = "high"
        note    = ("London session — strong directional moves, real volume. "
                   "Best time for breakout and trend-following setups.")
        caution = False
    elif 13 <= hour < 14:
        session = "LONDON_NY_OVERLAP"
        quality = "very_high"
        note    = ("London/NY overlap — peak volume of the day. "
                   "Highest probability for sustained directional moves. "
                   "Watch for NY open sweep at 13:30 UTC, then follow the real direction.")
        caution = False
    elif 14 <= hour < 21:
        session = "NEW_YORK"
        quality = "high"
        note    = ("New York session — high volume, trending moves. "
                   "Good for continuation setups with the established intraday trend.")
        caution = False
    elif 21 <= hour < 22:
        session = "NY_CLOSE"
        quality = "medium"
        note    = ("Approaching NY close — institutions reducing exposure. "
                   "Avoid new entries. Manage open trades.")
        caution = True
    else:  # 22-23
        session = "DEAD_ZONE"
        quality = "very_low"
        note    = ("Dead zone — post-NY, pre-Asia. Thinnest liquidity of the day. "
                   "Do not enter new trades. Any moves here are not representative.")
        caution = True

    return {
        "session":     session,
        "quality":     quality,
        "note":        note,
        "caution":     caution,
        "hour_utc":    hour,
        "is_london":   8  <= hour < 16,
        "is_ny":       13 <= hour < 22,
        "is_asian":    hour < 7 or hour >= 22,
        "is_overlap":  13 <= hour < 16,
    }

DB_PATH = Path.home() / "CryptoDashboard" / "agent_kb" / "agent_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

RENDER_URL = os.environ.get("RENDER_URL", "http://localhost:8080")


def _conn() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    db.row_factory = sqlite3.Row
    return db


def init_db() -> None:
    db = _conn()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS agent_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS agent_reports (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            agent      TEXT,
            report_type TEXT,
            content    TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS postmortems (
            trade_id   TEXT PRIMARY KEY,
            symbol     TEXT,
            direction  TEXT,
            outcome    TEXT,
            roi        REAL,
            analysis   TEXT,
            lessons    TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            category   TEXT,
            entry      TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS ceo_decisions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT,
            decision       TEXT,
            reason         TEXT,
            approved_count INTEGER,
            rejected_count INTEGER,
            message        TEXT,
            timestamp      TEXT
        );
    """)
    db.commit()
    db.close()


def set_state(key: str, value: dict | list | str) -> None:
    db = _conn()
    db.execute(
        "INSERT OR REPLACE INTO agent_state (key, value, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value, default=str), datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()


def get_state(key: str, default=None):
    db = _conn()
    row = db.execute("SELECT value FROM agent_state WHERE key=?", (key,)).fetchone()
    db.close()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]
    return default


def add_report(agent: str, report_type: str, content: dict) -> None:
    db = _conn()
    db.execute(
        "INSERT INTO agent_reports (agent, report_type, content, created_at) VALUES (?,?,?,?)",
        (agent, report_type, json.dumps(content, default=str),
         datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()


def get_recent_reports(agent: str = None, limit: int = 20) -> list:
    db = _conn()
    if agent:
        rows = db.execute(
            "SELECT * FROM agent_reports WHERE agent=? ORDER BY id DESC LIMIT ?",
            (agent, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM agent_reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    db.close()
    result = []
    for r in rows:
        try:
            result.append({
                "id": r["id"], "agent": r["agent"], "type": r["report_type"],
                "content": json.loads(r["content"]), "created_at": r["created_at"]
            })
        except Exception:
            pass
    return result


def save_postmortem(trade_id: str, symbol: str, direction: str,
                    outcome: str, roi: float, analysis: str, lessons: str) -> None:
    db = _conn()
    db.execute(
        """INSERT OR REPLACE INTO postmortems
           (trade_id, symbol, direction, outcome, roi, analysis, lessons, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (trade_id, symbol, direction, outcome, roi, analysis, lessons,
         datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()


def get_postmortems(limit: int = 20) -> list:
    db = _conn()
    rows = db.execute(
        "SELECT * FROM postmortems ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


def add_knowledge(category: str, entry: dict) -> None:
    db = _conn()
    db.execute(
        "INSERT INTO knowledge_base (category, entry, created_at) VALUES (?,?,?)",
        (category, json.dumps(entry, default=str),
         datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()


def log_ceo_decision(symbol: str, decision: str, reason: str,
                     approved_count: int, rejected_count: int, message: str) -> None:
    db = _conn()
    db.execute(
        """INSERT INTO ceo_decisions
           (symbol, decision, reason, approved_count, rejected_count, message, timestamp)
           VALUES (?,?,?,?,?,?,?)""",
        (symbol, decision, reason, approved_count, rejected_count, message,
         datetime.now(timezone.utc).isoformat())
    )
    db.commit()
    db.close()


def get_knowledge(category: str, limit: int = 10) -> list:
    db = _conn()
    rows = db.execute(
        "SELECT entry FROM knowledge_base WHERE category=? ORDER BY id DESC LIMIT ?",
        (category, limit)
    ).fetchall()
    db.close()
    result = []
    for r in rows:
        try:
            result.append(json.loads(r[0]))
        except Exception:
            pass
    return result


def post_to_render(endpoint: str, data: dict) -> bool:
    """Post data to Render dashboard.

    Render free tier cold-starts in 30-60s. We retry up to 3 times with a
    90-second timeout so a sleeping instance doesn't silently drop signals.
    """
    import requests, time as _time
    url = f"{RENDER_URL}{endpoint}"
    for attempt in range(3):
        try:
            r = requests.post(url, json=data, timeout=90)
            body = r.text[:120]
            if r.status_code not in (200,201): print(f"  !! Render {endpoint} → {r.status_code} (attempt {attempt+1})", flush=True)
            if r.status_code in (200, 201):
                return True
        except Exception as _e:
            print(f"  !! Render {endpoint} failed (attempt {attempt+1}): {_e}", flush=True)
        if attempt < 2:
            _time.sleep(5)  # brief pause before retry
    return False


def get_from_render(endpoint: str) -> dict | list:
    """Fetch data from Render."""
    import requests
    try:
        r = requests.get(f"{RENDER_URL}{endpoint}", timeout=90)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}
