from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import os

app = FastAPI(title="MemGPT Memory Log Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

DB_PATH = None

def get_db_path():
    global DB_PATH
    if DB_PATH:
        return DB_PATH
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from memgpt.config import MemGPTConfig
        config = MemGPTConfig.load()
        DB_PATH = os.path.join(config.recall_storage_path, "sqlite.db")
    except Exception:
        DB_PATH = os.path.expanduser("~/.memgpt/sqlite.db")
    return DB_PATH

def query_logs(sql, params=()):
    db_path = get_db_path()
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        results = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in results]
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        return []


@app.get("/")
def root():
    return FileResponse(str(frontend_path / "index.html"))


@app.get("/api/sessions")
def sessions():
    return query_logs("""
        SELECT
            session_id,
            agent_id,
            MIN(timestamp) as started_at,
            MAX(timestamp) as last_event,
            COUNT(*) as total_ops,
            SUM(CASE WHEN operation LIKE 'core%'     AND operation NOT LIKE '%read%' THEN 1 ELSE 0 END) as core_ops,
            SUM(CASE WHEN operation LIKE 'archival%'                                 THEN 1 ELSE 0 END) as archival_ops,
            SUM(CASE WHEN operation LIKE 'recall%'                                   THEN 1 ELSE 0 END) as recall_ops,
            SUM(CASE WHEN operation LIKE '%read%'                                    THEN 1 ELSE 0 END) as read_ops,
            SUM(CASE WHEN ground_truth_label = 1                                     THEN 1 ELSE 0 END) as attack_ops,
            MAX(attack_scenario) as attack_scenario
        FROM memory_logs
        WHERE session_id IS NOT NULL
        GROUP BY session_id, agent_id
        ORDER BY started_at DESC
    """)


@app.get("/api/timeline/{agent_id}")
def timeline(agent_id: str):
    return query_logs("""
        SELECT * FROM memory_logs
        WHERE agent_id = ?
        ORDER BY sequence_num ASC
    """, (agent_id,))


@app.get("/api/before/{sequence_num}")
def before_response(sequence_num: int, agent_id: str = Query(...), window: int = 10):
    return query_logs("""
        SELECT * FROM memory_logs
        WHERE sequence_num < ? AND agent_id = ?
        ORDER BY sequence_num DESC
        LIMIT ?
    """, (sequence_num, agent_id, window))


@app.get("/api/recent-writes")
def recent_writes(hours: int = 1, agent_id: str = None):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    ops = (
        'core_memory_write_persona',
        'core_memory_write_human',
        'core_memory_append',
        'core_memory_replace',
        'core_write',
        'core_append',
        'core_replace',
    )
    placeholders = ','.join('?' * len(ops))
    if agent_id:
        return query_logs(f"""
            SELECT * FROM memory_logs
            WHERE operation IN ({placeholders})
            AND timestamp > ? AND agent_id = ?
            ORDER BY sequence_num ASC
        """, (*ops, cutoff, agent_id))
    return query_logs(f"""
        SELECT * FROM memory_logs
        WHERE operation IN ({placeholders})
        AND timestamp > ?
        ORDER BY sequence_num ASC
    """, (*ops, cutoff))


@app.get("/api/recent-reads")
def recent_reads(hours: int = 1, agent_id: str = None):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    if agent_id:
        return query_logs("""
            SELECT * FROM memory_logs
            WHERE operation LIKE '%read%'
            AND timestamp > ? AND agent_id = ?
            ORDER BY sequence_num ASC
        """, (cutoff, agent_id))
    return query_logs("""
        SELECT * FROM memory_logs
        WHERE operation LIKE '%read%'
        AND timestamp > ?
        ORDER BY sequence_num ASC
    """, (cutoff,))


@app.get("/api/diff/{log_id}")
def diff(log_id: int):
    rows = query_logs("SELECT * FROM memory_logs WHERE id = ?", (log_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="Log entry not found")
    row = rows[0]
    return {
        "id": row["id"],
        "operation": row["operation"],
        "timestamp": row["timestamp"],
        "previous": row["previous_value"],
        "new": row["content"],
        "token_offset": row["token_offset"],
        "context_window_pct": row["context_window_pct"],
        "agent_id": row["agent_id"],
        "session_id": row.get("session_id"),
        "sequence_num": row["sequence_num"],
    }


@app.get("/api/stats")
def stats():
    rows = query_logs("""
        SELECT
            COUNT(*) as total_ops,
            COUNT(DISTINCT session_id) as total_sessions,
            COUNT(DISTINCT agent_id) as total_agents,
            SUM(CASE WHEN ground_truth_label = 1 THEN 1 ELSE 0 END) as total_attacks,
            SUM(CASE WHEN operation LIKE 'core%'     AND operation NOT LIKE '%read%' THEN 1 ELSE 0 END) as core_ops,
            SUM(CASE WHEN operation LIKE 'archival%'                                 THEN 1 ELSE 0 END) as archival_ops,
            SUM(CASE WHEN operation LIKE 'recall%'                                   THEN 1 ELSE 0 END) as recall_ops,
            SUM(CASE WHEN operation LIKE '%read%'                                    THEN 1 ELSE 0 END) as read_ops
        FROM memory_logs
    """)
    return rows[0] if rows else {}


@app.get("/api/search")
def search(q: str = Query(...), agent_id: str = None):
    base = "SELECT * FROM memory_logs WHERE content LIKE ?"
    params = [f"%{q}%"]
    if agent_id:
        base += " AND agent_id = ?"
        params.append(agent_id)
    base += " ORDER BY sequence_num ASC LIMIT 100"
    return query_logs(base, params)


@app.post("/api/label/{session_id}")
def label_session(session_id: str, attack_scenario: str, ground_truth_label: int):
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        UPDATE memory_logs
        SET attack_scenario = ?, ground_truth_label = ?
        WHERE session_id = ?
    """, (attack_scenario, ground_truth_label, session_id))
    conn.commit()
    conn.close()
    return {"status": "ok"}
