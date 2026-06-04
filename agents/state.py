"""
Shared state manager for the trading desk agent system.
All agents read and write through this module.
SQLite-backed so state persists across restarts.
"""
from __future__ import annotations
import sqlite3, json, os
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / "CryptoDashboard" / "agent_kb" / "agent_state.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

RENDER_URL = os.environ.get("RENDER_URL", "https://cryptodashboard-nuf5.onrender.com")


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
    """Post data to Render dashboard."""
    import requests
    try:
        r = requests.post(f"{RENDER_URL}{endpoint}", json=data, timeout=12)
        return r.status_code in (200, 201)
    except Exception:
        return False


def get_from_render(endpoint: str) -> dict | list:
    """Fetch data from Render."""
    import requests
    try:
        r = requests.get(f"{RENDER_URL}{endpoint}", timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}
