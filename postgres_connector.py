from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg import Connection
from psycopg.rows import tuple_row


_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"MERGE|REPLACE|CALL|EXECUTE|COPY)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PostgresConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    enabled: bool = True
    preview_rows: int = 10
    max_rows: int = 5000
    connect_timeout: int = 10
    statement_timeout_ms: Optional[int] = 120_000


@dataclass
class QueryResult:
    columns: List[str]
    preview_rows: List[List[Any]]
    row_count: int
    truncated: bool
    csv_content: str

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.preview_rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "csv": self.csv_content,
        }


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def load_postgres_config_from_env() -> PostgresConfig:
    timeout_raw = _env("POSTGRES_STATEMENT_TIMEOUT_MS", "120000")
    statement_timeout_ms = int(timeout_raw) if timeout_raw else None
    return PostgresConfig(
        host=_env("POSTGRES_HOST", "localhost"),
        port=int(_env("POSTGRES_PORT", "5432")),
        database=_env("POSTGRES_DATABASE", "ecommerce_analytics"),
        user=_env("POSTGRES_USER", "postgres"),
        password=_env("POSTGRES_PASSWORD", ""),
        enabled=_env("POSTGRES_ENABLED", "true").lower() in {"1", "true", "yes"},
        preview_rows=int(_env("POSTGRES_PREVIEW_ROWS", "10")),
        max_rows=int(_env("POSTGRES_MAX_ROWS", "5000")),
        connect_timeout=int(_env("POSTGRES_CONNECT_TIMEOUT", "10")),
        statement_timeout_ms=statement_timeout_ms,
    )


def is_read_only_query(sql: str) -> bool:
    """Allow single SELECT / WITH statements only."""
    text = sql.strip()
    if not text:
        return False
    normalized = text.rstrip(";").strip()
    if ";" in normalized:
        return False
    upper = normalized.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False
    if _FORBIDDEN_SQL.search(normalized):
        return False
    return True


def _serialize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def rows_to_csv(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_serialize_cell(cell) for cell in row])
    return buf.getvalue()


class PostgresConnector:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config

    def _conninfo(self) -> str:
        cfg = self.config
        return (
            f"host={cfg.host} port={cfg.port} dbname={cfg.database} "
            f"user={cfg.user} password={cfg.password}"
        )

    def connect(self) -> Connection:
        return psycopg.connect(
            self._conninfo(),
            connect_timeout=self.config.connect_timeout,
            row_factory=tuple_row,
        )

    def test_connection(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")

    def run_query(self, sql: str) -> QueryResult:
        if not self.config.enabled:
            raise RuntimeError("PostgreSQL execution is disabled (POSTGRES_ENABLED=false).")
        if not is_read_only_query(sql):
            raise ValueError("Only read-only SELECT queries are allowed.")

        cfg = self.config
        columns: List[str] = []
        fetched: List[List[Any]] = []
        truncated = False

        with self.connect() as conn:
            if cfg.statement_timeout_ms:
                timeout_ms = int(cfg.statement_timeout_ms)
                with conn.cursor() as setup:
                    setup.execute(f"SET statement_timeout = {timeout_ms}")
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    raise ValueError("Query did not return a result set.")
                columns = [desc.name for desc in cur.description]
                while len(fetched) < cfg.max_rows:
                    batch = cur.fetchmany(min(500, cfg.max_rows - len(fetched)))
                    if not batch:
                        break
                    for row in batch:
                        fetched.append([_serialize_cell(v) for v in row])
                        if len(fetched) >= cfg.max_rows:
                            truncated = True
                            break
                    if truncated:
                        break
                if len(fetched) >= cfg.max_rows:
                    extra = cur.fetchone()
                    if extra is not None:
                        truncated = True

        preview = fetched[: cfg.preview_rows]
        return QueryResult(
            columns=columns,
            preview_rows=preview,
            row_count=len(fetched),
            truncated=truncated,
            csv_content=rows_to_csv(columns, fetched),
        )