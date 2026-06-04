"""
SQLite-backed debate persistence + shareable IDs.

Schema: debates(id TEXT PK, topic TEXT, created_at REAL, data TEXT[json])
DB path: data/debates.db (gitignored).
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_DB_PATH = Path(os.environ.get("DEBATE_DB", "data/debates.db"))
_lock = threading.Lock()


def _conn():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH))
    c.execute(
        "CREATE TABLE IF NOT EXISTS debates "
        "(id TEXT PRIMARY KEY, topic TEXT, created_at REAL, data TEXT)"
    )
    return c


def _short_id() -> str:
    # 8 hex chars from os.urandom — no Math.random/uuid needed.
    return os.urandom(4).hex()


def save_debate(topic: str, rounds, summary, created_at: Optional[float] = None) -> str:
    """Persist a debate and return its share id."""
    created_at = created_at if created_at is not None else time.time()
    payload = json.dumps({"topic": topic, "rounds": rounds, "summary": summary},
                         ensure_ascii=False)
    with _lock:
        c = _conn()
        try:
            did = _short_id()
            for _ in range(5):  # avoid the rare id collision
                exists = c.execute("SELECT 1 FROM debates WHERE id=?", (did,)).fetchone()
                if not exists:
                    break
                did = _short_id()
            c.execute("INSERT INTO debates (id, topic, created_at, data) VALUES (?,?,?,?)",
                      (did, topic, created_at, payload))
            c.commit()
            return did
        finally:
            c.close()


def get_debate(did: str):
    with _lock:
        c = _conn()
        try:
            row = c.execute("SELECT id, topic, created_at, data FROM debates WHERE id=?",
                            (did,)).fetchone()
        finally:
            c.close()
    if not row:
        return None
    data = json.loads(row[3])
    data["id"] = row[0]
    data["created_at"] = row[2]
    return data


def list_debates(limit: int = 20):
    with _lock:
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, topic, created_at FROM debates ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        finally:
            c.close()
    return [{"id": r[0], "topic": r[1], "created_at": r[2]} for r in rows]
