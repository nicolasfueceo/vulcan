import sqlite3
from typing import Optional

class AgenticDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._ensure_table()

    def _ensure_table(self):
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def set(self, key: str, value: str) -> None:
        self.conn.execute("REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()
