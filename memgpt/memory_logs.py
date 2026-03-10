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
    conn.executescript("""
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
            attack_scenario TEXT,
            ground_truth_label INTEGER,
            previous_value TEXT
        );
                 
        CREATE INDEX IF NOT EXISTS idx_session ON memory_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_agent ON memory_logs(agent_id);
        CREATE INDEX IF NOT EXISTS idx_operation ON memory_logs(operation);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_logs(timestamp);
    """)

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

def _log(operation, content, previous_value=None, agent_id=None, user_id=None, model=None, context_window=None):
    # print(f"Logging operation: {operation}, content: {content}, session_id: {session_id}, previous_value: {previous_value}, agent_id: {agent_id}, user_id: {user_id}, model: {model}, context_window: {context_window}")
    _ensure_db()
    conn = sqlite3.connect(_get_db_path())
    if CURRENT_AGENT_ID:
        agent_id = CURRENT_AGENT_ID
    # conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """INSERT INTO memory_logs 
        (timestamp, sequence_num, operation, content, token_offset, context_window_pct, model, context_window, user_id, agent_id, attack_scenario, ground_truth_label, previous_value) 
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
            str(user_id) if user_id else None,
            str(agent_id) if agent_id else None,
            previous_value
        )
    )
    conn.commit()
    conn.close()

CURRENT_TOKEN_OFFSET = None
CURRENT_CONTEXT_WINDOW_PCT = None

def update_token_context(total_tokens: int, context_window: int):
    global CURRENT_TOKEN_OFFSET, CURRENT_CONTEXT_WINDOW_PCT
    CURRENT_TOKEN_OFFSET = total_tokens
    CURRENT_CONTEXT_WINDOW_PCT = round(total_tokens / context_window, 4) if context_window else None


class LoggedCoreMemory(CoreMemory):

    def set_agent_id(self, agent_id):
        # print(f"Setting CURRENT_AGENT_ID to {agent_id} for logging purposes")
        # This is a fallback in case agent_id is not passed in directly to the logging calls. Prefer explicit agent_id when available.
        global CURRENT_AGENT_ID
        CURRENT_AGENT_ID = agent_id

    # print("LoggedCoreMemory class defined")  # Debug print to confirm class definition
    # print(CoreMemory.__dict__)  # Debug print to see all attributes of CoreMemory
    # print("\n\n\n")
    
    # def _get_safe_id(self):
    #     # This checks three different places where the ID might be hiding
    #     print("SELF\n\n")
    #     print(self.__dict__)  # Debug print to see all attributes
    #     print("\n\n\n")
    #     try:
    #         return getattr(self, "id", None) or \
    #                getattr(self.agent_id, "id", None) or \
    #                getattr(self.agent, "id", None) or \
    #                getattr(self.agent_state, "id", None) or \
    #                "INITIALIZING" # If all else fails, mark it as initializing
    #     except:
    #         return "UNKNOWN"
            
    def edit_persona(self, new_persona):
        print(f"[LoggedCoreMemory] edit_persona called")
        _log("core_memory_write_persona", {"field": "persona", "new_value": new_persona},
            previous_value=self.persona)
        # base class ignores agent_id, but include for signature consistency
        return super().edit_persona(new_persona)

    def edit_human(self, new_human, agent_id=None):
        print(f"[LoggedCoreMemory] edit_human called")
        # prefer explicit agent_id over whatever the safe fetcher returns
        # aid = agent_id or self._get_safe_id()
        _log("core_memory_write_human", {"field": "human", "new_value": new_human},
            previous_value=self.human)
        # base class ignores agent_id, but keep signature for consistency
        return super().edit_human(new_human)

    def edit_append(self, field, content, sep="\n"):
        print(f"[LoggedCoreMemory] edit_append called")
        # aid = self._get_safe_id() # Use the safe ID fetcher
        prev = self.persona if field == "persona" else self.human
        _log("core_memory_append", {"field": field, "content": content},
             previous_value=prev)
        return super().edit_append(field, content, sep)

    def edit_replace(self, field, old_content, new_content):
        print(f"[LoggedCoreMemory] edit_replace called")
        # aid = self._get_safe_id() # Use the safe ID fetcher
        prev = self.persona if field == "persona" else self.human
        _log("core_memory_replace", {"field": field, "old": old_content, "new": new_content},
            previous_value=prev)
        return super().edit_replace(field, old_content, new_content)


class LoggedRecallMemory(BaseRecallMemory):
    # print("LoggedRecallMemory class defined")  # Debug print to confirm class definition
    # print(BaseRecallMemory.__dict__)  # Debug print to see all attributes of BaseRecallMemory
    # print("\n\n\n")

    def text_search(self, query_string, count=None, start=None):
        results, total = super().text_search(query_string, count, start)
        _log("recall_text_search", {"query": query_string, "total_results": total},
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return results, total

    def date_search(self, start_date, end_date, count=None, start=None):
        results, total = super().date_search(start_date, end_date, count, start)
        _log("recall_date_search", {"start_date": start_date, "end_date": end_date, "total_results": total},
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return results, total

    def insert(self, message: Message):
        # print(f"[LoggedRecallMemory] insert called with message: {message._dict__}")  # Debug print to see message content
        _log("recall_insert", {"role": message.role, "text": message.text},
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return super().insert(message)

    def insert_many(self, messages: List[Message]):
        for m in messages:
            _log("recall_insert_many", {"role": m.role, "text": m.text},
                 user_id=self.agent_state.user_id,
                 context_window=self.agent_state.llm_config.context_window,
                 model=self.agent_state.llm_config.model,
                 agent_id=self.agent_state.id)
        return super().insert_many(messages)


class LoggedArchivalMemory(EmbeddingArchivalMemory):
    # print("LoggedArchivalMemory class defined")  # Debug print to confirm class definition
    # print(EmbeddingArchivalMemory.__dict__)  # Debug print to see all attributes of EmbeddingArchivalMemory
    # print("\n\n\n")
    
    def insert(self, memory_string, return_ids=False):
        _log("archival_insert", {"content": memory_string},
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
        },
           user_id=self.agent_state.user_id,
           context_window=self.agent_state.llm_config.context_window,
            model=self.agent_state.llm_config.model,
           agent_id=self.agent_state.id)
        return results, total

    def delete(self, filters=None):
        _log("archival_delete", {"filters": str(filters)},
             user_id=self.agent_state.user_id,
             context_window=self.agent_state.llm_config.context_window,
             model=self.agent_state.llm_config.model,
             agent_id=self.agent_state.id)
        return super().delete(filters)
