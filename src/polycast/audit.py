"""Audit trail and persistence.

Every run writes an append-only JSONL trace: which documents were retrieved,
what each analyst concluded, what the red team objected to, and how the panel
was aggregated. When a number looks wrong three weeks later this is the
difference between "re-run and hope" and actually debugging the decision.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .schemas import Forecast, utcnow


class AuditTrail:
    def __init__(self, trace_id: str, directory: str | Path = "runs") -> None:
        self.trace_id = trace_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{trace_id}.jsonl"
        self._seq = 0

    def record(self, step: str, payload: dict[str, Any]) -> None:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "ts": utcnow().isoformat(),
            "trace_id": self.trace_id,
            "step": step,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


class ForecastStore:
    def __init__(self, path: str | Path = "polycast.db") -> None:
        self.path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    probability REAL NOT NULL,
                    lower REAL NOT NULL,
                    upper REAL NOT NULL,
                    disagreement REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(self, forecast: Forecast) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecasts
                    (forecast_id, question_text, probability, lower, upper,
                     disagreement, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    forecast.forecast_id,
                    forecast.question.text,
                    forecast.probability,
                    forecast.lower,
                    forecast.upper,
                    forecast.disagreement,
                    forecast.model_dump_json(),
                    forecast.created_at.isoformat(),
                ),
            )

    def get(self, forecast_id: str) -> Forecast | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM forecasts WHERE forecast_id = ?", (forecast_id,)
            ).fetchone()
        return Forecast.model_validate_json(row["payload"]) if row else None

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT forecast_id, question_text, probability, lower, upper,
                       disagreement, created_at
                FROM forecasts ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS c FROM forecasts").fetchone()["c"]
