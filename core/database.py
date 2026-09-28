"""SQLite persistence for fruit sorting results."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SorterDatabase:
    """Stores classification events in a small local SQLite database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sorting_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    label TEXT NOT NULL,
                    fruit TEXT NOT NULL,
                    quality TEXT NOT NULL,
                    confidence REAL,
                    source TEXT NOT NULL DEFAULT 'camera',
                    frame_path TEXT
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(sorting_events)")}
            if "frame_path" not in columns:
                connection.execute("ALTER TABLE sorting_events ADD COLUMN frame_path TEXT")
            # Normalize records written by the previous dashboard version so
            # the report uses an explicit health status and a real item name.
            connection.execute(
                "UPDATE sorting_events SET quality = 'Healthy' WHERE quality = 'Identified'"
            )
            connection.execute(
                "UPDATE sorting_events SET fruit = UPPER(SUBSTR(label, 1, 1)) || SUBSTR(label, 2) "
                "WHERE fruit = 'Fruit'"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sorting_events_created_at "
                "ON sorting_events(created_at DESC)"
            )

    def record(self, result: dict, source: str = "camera", frame_path: str | None = None) -> dict:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sorting_events (label, fruit, quality, confidence, source, frame_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (result["label"], result["fruit"], result["quality"], result.get("confidence"), source, frame_path),
            )
            row = connection.execute(
                "SELECT id, created_at, label, fruit, quality, confidence, source, frame_path "
                "FROM sorting_events WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return dict(row)

    def total_events(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM sorting_events").fetchone()[0])

    def latest_event(self) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, created_at, label, fruit, quality, confidence, source, frame_path "
                "FROM sorting_events ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, created_at, label, fruit, quality, confidence, source, frame_path "
                "FROM sorting_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def report(self) -> dict:
        """Return aggregate counts and newest persisted findings."""
        with self._connection() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM sorting_events").fetchone()[0])
            status_rows = connection.execute(
                "SELECT quality, COUNT(*) AS count FROM sorting_events GROUP BY quality"
            ).fetchall()
            fruit_rows = connection.execute(
                "SELECT fruit, COUNT(*) AS count FROM sorting_events GROUP BY fruit ORDER BY count DESC, fruit"
            ).fetchall()
        return {
            "total_fruits": total,
            "health_statuses": {row["quality"]: row["count"] for row in status_rows},
            "fruit_counts": {row["fruit"]: row["count"] for row in fruit_rows},
            "recent_events": self.recent_events(12),
        }
