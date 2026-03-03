import json
import sqlite3
import os
from datetime import datetime
from typing import List, Optional
from memgpt.memory import CoreMemory, BaseRecallMemory, EmbeddingArchivalMemory
from memgpt.data_types import Message

DB_PATH = None

_sequence_num = 0

def _next_seq():
    global _sequence_num
    _sequence_num += 1
    return _sequence_num

def _get_db_path():
    global DB_PATH
    if DB_PATH is not None:
        return DB_PATH
    # lazy import to avoid circular imports
    from memgpt.config import MemGPTConfig
    config = MemGPTConfig.load()
    DB_PATH = os.path.join(config.recall_storage_path, "sqlite.db")
    return DB_PATH

def _init_db():
    conn = sqlite3.connect(_get_db_path())
    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS memory_logs (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         timestamp TEXT NOT NULL,
    #         operation TEXT NOT NULL,
    #         content TEXT NOT NULL,
    #         user_id TEXT,
    #         agent_id TEXT,
    #         token_offset INTEGER,
    #         context_window_pct REAL,
    #         model TEXT,
    #         sequence_num INTEGER NOT NULL,
    #         context_window INTEGER,
    #         session_id TEXT,
    #         attack_scenario TEXT,
    #         ground_truth_label INTEGER,
    #         previous_value TEXT
    #     )
                 
    #     CREATE INDEX IF NOT EXISTS idx_session ON memory_logs(session_id);
    #     CREATE INDEX IF NOT EXISTS idx_agent ON memory_logs(agent_id);
    #     CREATE INDEX IF NOT EXISTS idx_operation ON memory_logs(operation);
    #     CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_logs(timestamp);
    # """)

    # Create table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            operation TEXT NOT NULL,
            content TEXT NOT NULL,
            user_id TEXT,
            agent_id TEXT,
            token_offset INTEGER,
            context_window_pct REAL,
            model TEXT,
            sequence_num INTEGER NOT NULL,
            context_window INTEGER,
            session_id TEXT,
            attack_scenario TEXT,
            ground_truth_label INTEGER,
            previous_value TEXT
        )
    """)

    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON memory_logs(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON memory_logs(agent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operation ON memory_logs(operation)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_logs(timestamp)")

    conn.commit()
    conn.close()

# DO NOT call _init_db() at module level - call it lazily
_db_initialized = False

def _ensure_db():
    global _db_initialized
    if not _db_initialized:
        _init_db()
        _db_initialized = True

def _log(operation, content, session_id=None, previous_value=None, user_id=None, agent_id=None, model=None, context_window=None):
    _ensure_db()
    conn = sqlite3.connect(_get_db_path())
    # conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """INSERT INTO memory_logs 
        (timestamp, sequence_num, operation, content, token_offset, context_window_pct, model, context_window, session_id, user_id, agent_id, attack_scenario, ground_truth_label, previous_value) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)""",
        (
            datetime.utcnow().isoformat(),
            _next_seq(),
            operation,
            json.dumps(content),
            CURRENT_TOKEN_OFFSET,
            CURRENT_CONTEXT_WINDOW_PCT,
            model,
            context_window,
            session_id,
            str(user_id) if user_id else None,
            str(agent_id) if agent_id else None,
            previous_value
        )
    )
    conn.commit()
    conn.close()

CURRENT_SESSION_ID = None
CURRENT_TOKEN_OFFSET = None
CURRENT_CONTEXT_WINDOW_PCT = None

def set_session(session_id: str):
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id

def update_token_context(total_tokens: int, context_window: int):
    global CURRENT_TOKEN_OFFSET, CURRENT_CONTEXT_WINDOW_PCT
    CURRENT_TOKEN_OFFSET = total_tokens
    CURRENT_CONTEXT_WINDOW_PCT = round(total_tokens / context_window, 4) if context_window else None


class LoggedCoreMemory(CoreMemory):
    def edit_persona(self, new_persona):
        print(f"[LoggedCoreMemory] edit_persona called")
        _log("core_memory_write_persona", {"field": "persona", "new_value": new_persona},
             session_id=CURRENT_SESSION_ID, previous_value=self.persona)
        return super().edit_persona(new_persona)

    def edit_human(self, new_human):
        print(f"[LoggedCoreMemory] edit_human called")
        _log("core_memory_write_human", {"field": "human", "new_value": new_human},
             session_id=CURRENT_SESSION_ID, previous_value=self.human)
        return super().edit_human(new_human)

    def edit_append(self, field, content, sep="\n"):
        print(f"[LoggedCoreMemory] edit_append called")
        prev = self.persona if field == "persona" else self.human
        _log("core_memory_append", {"field": field, "content": content},
             session_id=CURRENT_SESSION_ID, previous_value=prev)
        return super().edit_append(field, content, sep)

    def edit_replace(self, field, old_content, new_content):
        print(f"[LoggedCoreMemory] edit_replace called")
        prev = self.persona if field == "persona" else self.human
        _log("core_memory_replace", {"field": field, "old": old_content, "new": new_content},
             session_id=CURRENT_SESSION_ID, previous_value=prev)
        return super().edit_replace(field, old_content, new_content)


class LoggedRecallMemory(BaseRecallMemory):
    def text_search(self, query_string, count=None, start=None):
        results, total = super().text_search(query_string, count, start)
        _log("recall_text_search", {"query": query_string, "total_results": total},
             session_id=CURRENT_SESSION_ID,
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return results, total

    def date_search(self, start_date, end_date, count=None, start=None):
        results, total = super().date_search(start_date, end_date, count, start)
        _log("recall_date_search", {"start_date": start_date, "end_date": end_date, "total_results": total},
             session_id=CURRENT_SESSION_ID,
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return results, total

    def insert(self, message: Message):
        _log("recall_insert", {"role": message.role, "text": message.text},
             session_id=CURRENT_SESSION_ID,
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return super().insert(message)

    def insert_many(self, messages: List[Message]):
        for m in messages:
            _log("recall_insert_many", {"role": m.role, "text": m.text},
                 session_id=CURRENT_SESSION_ID,
                 user_id=self.agent_state.user_id,
                 context_window=self.agent_state.llm_config.context_window,
                 model=self.agent_state.llm_config.model,
                 agent_id=self.agent_state.id)
        return super().insert_many(messages)


class LoggedArchivalMemory(EmbeddingArchivalMemory):
    def insert(self, memory_string, return_ids=False):
        _log("archival_insert", {"content": memory_string},
             session_id=CURRENT_SESSION_ID,
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return super().insert(memory_string, return_ids)

    def search(self, query_string, count=None, start=None):
        results, total = super().search(query_string, count, start)
        _log("archival_search", {
            "query": query_string,
            "total_results": total,
            "results": [r["content"] for r in results]
        }, session_id=CURRENT_SESSION_ID,
           user_id=self.agent_state.user_id,
           context_window=self.agent_state.llm_config.context_window,
            model=self.agent_state.llm_config.model,
           agent_id=self.agent_state.id)
        return results, total

    def delete(self, filters=None):
        _log("archival_delete", {"filters": str(filters)},
             session_id=CURRENT_SESSION_ID,
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return super().delete(filters)
