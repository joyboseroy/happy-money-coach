"""
feedback_store.py
Records user corrections to AI-assigned Honda tags and Arigato scores.
This dataset is the long-term hidden asset of the project:
  transaction + category + honda_tag + arigato_score + user_correction

Over time, this becomes training data for a fine-tuned Happy Money classifier.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "feedback.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            tx_description TEXT NOT NULL,
            tx_amount REAL NOT NULL,
            tx_date TEXT,
            category TEXT,
            honda_tag TEXT,
            arigato_score INTEGER,
            happy_money INTEGER,
            user_vote TEXT,           -- 'up' or 'down'
            user_corrected_tag TEXT,  -- if user corrects the tag
            user_corrected_score INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            tx_count INTEGER,
            wellbeing_score INTEGER,
            metadata TEXT   -- JSON blob for future use
        )
    """)
    conn.commit()
    conn.close()


def record_feedback(
    tx_description: str,
    tx_amount: float,
    tx_date: str,
    category: str,
    honda_tag: str,
    arigato_score: int,
    happy_money: bool,
    user_vote: str,           # 'up' or 'down'
    user_corrected_tag: str = None,
    user_corrected_score: int = None
):
    """Record a single user feedback entry."""
    init_db()
    conn = _get_conn()
    conn.execute("""
        INSERT INTO feedback
        (created_at, tx_description, tx_amount, tx_date, category,
         honda_tag, arigato_score, happy_money, user_vote,
         user_corrected_tag, user_corrected_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        tx_description,
        tx_amount,
        tx_date,
        category,
        honda_tag,
        arigato_score,
        int(happy_money),
        user_vote,
        user_corrected_tag,
        user_corrected_score
    ))
    conn.commit()
    conn.close()


def record_session(tx_count: int, wellbeing_score: int, metadata: dict = None):
    """Record an analysis session summary."""
    init_db()
    conn = _get_conn()
    conn.execute("""
        INSERT INTO sessions (created_at, tx_count, wellbeing_score, metadata)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        tx_count,
        wellbeing_score,
        json.dumps(metadata or {})
    ))
    conn.commit()
    conn.close()


def get_feedback_stats() -> dict:
    """Summary stats for the feedback dataset."""
    init_db()
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    upvotes = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_vote='up'").fetchone()[0]
    downvotes = conn.execute("SELECT COUNT(*) FROM feedback WHERE user_vote='down'").fetchone()[0]
    corrections = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE user_corrected_tag IS NOT NULL"
    ).fetchone()[0]
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    conn.close()

    accuracy = round(upvotes / total * 100, 1) if total > 0 else 0.0
    return {
        "total_feedback": total,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "corrections": corrections,
        "sessions": sessions,
        "ai_accuracy_pct": accuracy
    }


def export_training_data() -> list[dict]:
    """
    Export corrected feedback as training-ready data.
    Only rows where user made an explicit correction are included.
    """
    init_db()
    conn = _get_conn()
    rows = conn.execute("""
        SELECT tx_description, tx_amount, category,
               honda_tag AS predicted_tag,
               user_corrected_tag AS true_tag,
               arigato_score AS predicted_score,
               user_corrected_score AS true_score
        FROM feedback
        WHERE user_corrected_tag IS NOT NULL
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
