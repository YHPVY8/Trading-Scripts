from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from decimal import Decimal
import os
from typing import Any, Iterable, Sequence
from uuid import UUID


@dataclass
class QueryResult:
    data: list[dict[str, Any]]
    count: int | None = None


@dataclass
class Filter:
    column: str
    op: str
    value: Any


def get_database():
    return PostgresClient(_database_url())


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")
    if url:
        return url

    try:
        import streamlit as st

        return st.secrets["DATABASE_URL"]
    except Exception as exc:
        raise RuntimeError("Missing DATABASE_URL for Trading-Scripts database access") from exc


def quote_identifier(identifier: str) -> str:
    if not identifier:
        raise ValueError("identifier cannot be empty")
    return ".".join(
        f'"{part.replace(chr(34), chr(34) + chr(34)).replace("%", "%%")}"'
        for part in identifier.split(".")
    )


def quote_table_name(table_name: str) -> str:
    if "." in table_name:
        return quote_identifier(table_name)
    return quote_identifier(f"public.{table_name}")


def split_select(select_clause: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_quotes = False
    depth = 0
    for ch in select_clause:
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue
        if not in_quotes:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def normalize_column(column: str) -> str:
    column = column.strip()
    if len(column) >= 2 and column[0] == '"' and column[-1] == '"':
        return column[1:-1].replace('""', '"')
    return column


def clean_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    return value


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: clean_value(v) for k, v in row.items()}


def db_param(value: Any) -> Any:
    if isinstance(value, dict):
        from psycopg.types.json import Jsonb

        return Jsonb(value)
    return value


def batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


class PostgresClient:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            import psycopg

            self._conn = psycopg.connect(self.database_url)
            self._conn.autocommit = True
        return self._conn

    def table(self, table_name: str) -> "TableQuery":
        return TableQuery(self, table_name)


class TableQuery:
    def __init__(self, client: PostgresClient, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self._select = "*"
        self._filters: list[Filter] = []
        self._order_by: str | None = None
        self._desc = False
        self._limit: int | None = None
        self._offset: int | None = None
        self._action = "select"
        self._payload: Any = None
        self._count: str | None = None
        self._on_conflict: list[str] = []

    def select(self, columns: str = "*", count: str | None = None) -> "TableQuery":
        self._select = columns
        self._count = count
        return self

    def eq(self, column: str, value: Any) -> "TableQuery":
        self._filters.append(Filter(column, "eq", value))
        return self

    def in_(self, column: str, values: Sequence[Any]) -> "TableQuery":
        self._filters.append(Filter(column, "in", list(values)))
        return self

    def gte(self, column: str, value: Any) -> "TableQuery":
        self._filters.append(Filter(column, "gte", value))
        return self

    def lte(self, column: str, value: Any) -> "TableQuery":
        self._filters.append(Filter(column, "lte", value))
        return self

    def order(self, column: str, desc: bool = False) -> "TableQuery":
        self._order_by = column
        self._desc = desc
        return self

    def limit(self, limit: int) -> "TableQuery":
        self._limit = int(limit)
        return self

    def range(self, start: int, end: int) -> "TableQuery":
        self._offset = int(start)
        self._limit = max(0, int(end) - int(start) + 1)
        return self

    def insert(self, payload: dict[str, Any] | list[dict[str, Any]]) -> "TableQuery":
        self._action = "insert"
        self._payload = payload
        return self

    def update(self, payload: dict[str, Any]) -> "TableQuery":
        self._action = "update"
        self._payload = payload
        return self

    def delete(self) -> "TableQuery":
        self._action = "delete"
        return self

    def upsert(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        on_conflict: str | Sequence[str] | None = None,
    ) -> "TableQuery":
        self._action = "upsert"
        self._payload = payload
        if isinstance(on_conflict, str):
            self._on_conflict = [part.strip() for part in on_conflict.split(",") if part.strip()]
        elif on_conflict:
            self._on_conflict = [str(part).strip() for part in on_conflict if str(part).strip()]
        return self

    def execute(self) -> QueryResult:
        if self._action == "select":
            return self._execute_select()
        if self._action == "insert":
            return self._execute_insert()
        if self._action == "update":
            return self._execute_update()
        if self._action == "delete":
            return self._execute_delete()
        if self._action == "upsert":
            return self._execute_upsert()
        raise RuntimeError(f"Unsupported query action: {self._action}")

    def _execute_select(self) -> QueryResult:
        if self.table_name == "tj_trade_group_members" and "tj_trades(*)" in self._select:
            return self._execute_group_members_with_trades()

        from psycopg.rows import dict_row

        params: list[Any] = []
        columns = self._select_sql()
        where_sql = self._where_sql(params)
        sql = f"SELECT {columns} FROM {quote_table_name(self.table_name)}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if self._order_by:
            direction = "DESC" if self._desc else "ASC"
            sql += f" ORDER BY {quote_identifier(self._order_by)} {direction}"
        if self._limit is not None:
            sql += " LIMIT %s"
            params.append(self._limit)
        if self._offset is not None:
            sql += " OFFSET %s"
            params.append(self._offset)

        with self.client.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = [clean_row(dict(row)) for row in cur.fetchall()]
        return QueryResult(rows, count=len(rows) if self._count else None)

    def _execute_group_members_with_trades(self) -> QueryResult:
        from psycopg.rows import dict_row

        sql = """
            SELECT
                gm.group_id,
                gm.trade_id,
                to_jsonb(t.*) AS tj_trades
            FROM public.tj_trade_group_members gm
            LEFT JOIN public.tj_trades t ON t.id = gm.trade_id
        """
        with self.client.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            rows = []
            for row in cur.fetchall():
                item = dict(row)
                item["tj_trades"] = clean_value(item.get("tj_trades") or {})
                rows.append(clean_row(item))
        return QueryResult(rows)

    def _execute_insert(self) -> QueryResult:
        rows = self._rows_payload()
        if not rows:
            return QueryResult([])
        sql, params = self._insert_sql(rows)
        sql += " RETURNING *"
        return self._execute_returning(sql, params)

    def _execute_update(self) -> QueryResult:
        if not self._payload:
            return QueryResult([])
        if not self._filters:
            raise ValueError("update requires at least one filter")
        params = [db_param(value) for value in self._payload.values()]
        assignments = ", ".join(f"{quote_identifier(col)} = %s" for col in self._payload)
        where_sql = self._where_sql(params)
        sql = f"UPDATE {quote_table_name(self.table_name)} SET {assignments} WHERE {where_sql} RETURNING *"
        return self._execute_returning(sql, params)

    def _execute_delete(self) -> QueryResult:
        if not self._filters:
            raise ValueError("delete requires at least one filter")
        params: list[Any] = []
        where_sql = self._where_sql(params)
        sql = f"DELETE FROM {quote_table_name(self.table_name)} WHERE {where_sql} RETURNING *"
        return self._execute_returning(sql, params)

    def _execute_upsert(self) -> QueryResult:
        rows = self._rows_payload()
        if not rows:
            return QueryResult([])
        if not self._on_conflict:
            raise ValueError("upsert requires explicit on_conflict")
        partial_results = self._try_partial_upserts(rows)
        if partial_results is not None:
            return QueryResult(partial_results, count=len(partial_results) if self._count else None)
        sql, params = self._insert_sql(rows)
        columns = list(rows[0].keys())
        conflict = ", ".join(quote_identifier(col) for col in self._on_conflict)
        update_cols = [col for col in columns if col not in set(self._on_conflict)]
        if update_cols:
            updates = ", ".join(
                f"{quote_identifier(col)} = EXCLUDED.{quote_identifier(col)}"
                for col in update_cols
            )
            sql += f" ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        else:
            sql += f" ON CONFLICT ({conflict}) DO NOTHING"
        sql += " RETURNING *"
        return self._execute_returning(sql, params)

    def _try_partial_upserts(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        all_columns = set().union(*(row.keys() for row in rows))
        if not set(self._on_conflict).issubset(all_columns):
            return None
        update_cols = [col for col in rows[0].keys() if col not in set(self._on_conflict)]
        if not update_cols:
            return None

        results: list[dict[str, Any]] = []
        from psycopg.rows import dict_row

        with self.client.conn.cursor(row_factory=dict_row) as cur:
            for row in rows:
                if not set(self._on_conflict).issubset(row.keys()):
                    return None
                row_update_cols = [col for col in row.keys() if col not in set(self._on_conflict)]
                if not row_update_cols:
                    continue
                params = [db_param(row[col]) for col in row_update_cols]
                assignments = ", ".join(f"{quote_identifier(col)} = %s" for col in row_update_cols)
                where_parts = []
                for col in self._on_conflict:
                    where_parts.append(f"{quote_identifier(col)} = %s")
                    params.append(row[col])
                sql = (
                    f"UPDATE {quote_table_name(self.table_name)} SET {assignments} "
                    f"WHERE {' AND '.join(where_parts)} RETURNING *"
                )
                cur.execute(sql, params)
                updated = cur.fetchone()
                if updated:
                    results.append(clean_row(dict(updated)))
                    continue

                sql, params = self._insert_sql([row])
                sql += " RETURNING *"
                cur.execute(sql, params)
                inserted = cur.fetchone()
                if inserted:
                    results.append(clean_row(dict(inserted)))
        return results

    def _execute_returning(self, sql: str, params: list[Any]) -> QueryResult:
        from psycopg.rows import dict_row

        with self.client.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = [clean_row(dict(row)) for row in cur.fetchall()]
        return QueryResult(rows, count=len(rows) if self._count else None)

    def _insert_sql(self, rows: list[dict[str, Any]]) -> tuple[str, list[Any]]:
        columns = list(rows[0].keys())
        for row in rows:
            if list(row.keys()) != columns:
                raise ValueError("All rows must have the same columns in the same order")
        placeholders = ", ".join(["(" + ", ".join(["%s"] * len(columns)) + ")"] * len(rows))
        sql = (
            f"INSERT INTO {quote_table_name(self.table_name)} "
            f"({', '.join(quote_identifier(col) for col in columns)}) VALUES {placeholders}"
        )
        return sql, [db_param(row[col]) for row in rows for col in columns]

    def _rows_payload(self) -> list[dict[str, Any]]:
        if self._payload is None:
            return []
        if isinstance(self._payload, dict):
            return [self._payload]
        return list(self._payload)

    def _select_sql(self) -> str:
        if self._select.strip() == "*":
            return "*"
        columns = [normalize_column(part) for part in split_select(self._select)]
        return ", ".join(quote_identifier(col) for col in columns)

    def _where_sql(self, params: list[Any]) -> str:
        clauses: list[str] = []
        for f in self._filters:
            if f.op == "eq":
                clauses.append(f"{quote_identifier(f.column)} = %s")
                params.append(f.value)
            elif f.op == "in":
                clauses.append(f"{quote_identifier(f.column)} = ANY(%s)")
                params.append(list(f.value))
            elif f.op == "gte":
                clauses.append(f"{quote_identifier(f.column)} >= %s")
                params.append(f.value)
            elif f.op == "lte":
                clauses.append(f"{quote_identifier(f.column)} <= %s")
                params.append(f.value)
            else:
                raise ValueError(f"Unsupported filter operation: {f.op}")
        return " AND ".join(clauses)
