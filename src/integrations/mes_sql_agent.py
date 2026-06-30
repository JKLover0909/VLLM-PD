"""Validated Text-to-SQL harness for the local MES snapshot."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError


class MesSqlAgentError(RuntimeError):
    """Raised when a generated MES SQL query is unsafe or cannot be executed."""


@dataclass(frozen=True)
class MesSqlPlan:
    can_answer: bool
    sql: str = ""
    reason: str = ""


@dataclass(frozen=True)
class MesSqlQueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    imported_at: str
    truncated: bool

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "source": "mes_snapshot",
            "snapshot_imported_at": self.imported_at,
            "filters": {"exclude_test_data": True},
            "columns": self.columns,
            "rows": self.rows,
            "truncated": self.truncated,
        }


class MesSqlAgent:
    """Generate prompts, validate SQL AST and execute read-only MES queries."""

    ALLOWED_VIEWS = {
        "v_lot_error_summary",
        "v_lot_error_breakdown",
        "v_product_error_summary",
        "v_error_details",
    }
    INTERNAL_TABLES = {
        "lots",
        "error_events",
        "error_catalog",
    }
    PROHIBITED_NODE_KEYS = {
        "alter",
        "attach",
        "command",
        "commit",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "execute",
        "grant",
        "insert",
        "merge",
        "pragma",
        "rollback",
        "set",
        "transaction",
        "truncate",
        "update",
        "use",
    }
    ALLOWED_FUNCTIONS = {
        "abs",
        "avg",
        "coalesce",
        "count",
        "date",
        "datetime",
        "ifnull",
        "instr",
        "length",
        "like",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "replace",
        "round",
        "rtrim",
        "strftime",
        "substr",
        "substring",
        "sum",
        "time",
        "total",
        "trim",
        "upper",
    }

    def __init__(
        self,
        db_path: Path | str,
        semantic_model_path: Path | str,
        *,
        max_rows: int = 100,
        timeout_seconds: float = 2.0,
        max_sql_length: int = 8000,
    ):
        self.db_path = Path(db_path)
        self.semantic_model_path = Path(semantic_model_path)
        self.max_rows = max(1, min(int(max_rows), 500))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_sql_length = max(500, int(max_sql_length))
        self._semantic_model: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "MesSqlAgent | None":
        enabled = os.getenv("MES_SQL_AGENT_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(
            db_path=os.getenv("MES_DATABASE_PATH", "data/mes.sqlite"),
            semantic_model_path=os.getenv(
                "MES_SEMANTIC_MODEL_PATH",
                "config/mes_semantic_model.json",
            ),
            max_rows=int(os.getenv("MES_SQL_AGENT_MAX_ROWS", "100")),
            timeout_seconds=float(os.getenv("MES_SQL_AGENT_TIMEOUT", "2")),
        )

    @property
    def available(self) -> bool:
        return self.db_path.is_file() and self.semantic_model_path.is_file()

    def planner_messages(
        self,
        question: str,
        previous_error: str = "",
    ) -> list[dict[str, str]]:
        semantic_model = json.dumps(
            self.semantic_model(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        retry = (
            f"\nTruy vấn trước bị từ chối hoặc lỗi: {previous_error}\n"
            "Hãy sửa kế hoạch, không lặp lại lỗi này."
            if previous_error
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    "Bạn là bộ lập kế hoạch SQL cho MES snapshot SQLite. "
                    "Chỉ dùng view và cột trong semantic model. Chỉ sinh một "
                    "SELECT hoặc WITH...SELECT. Không dùng markdown, comment, "
                    "PRAGMA, ATTACH, DDL hay câu lệnh ghi. Khi câu hỏi không thể "
                    "trả lời từ schema, đặt can_answer=false. Cụm 'loại lỗi' "
                    "nghĩa là nhóm error_id + error_name. Luôn đặt alias dễ hiểu "
                    "và LIMIT phù hợp. Các view MES đã loại dữ liệu Lot/sản phẩm "
                    "test; không cố truy xuất lại dữ liệu test. Nếu câu hỏi cần "
                    "suy luận từ Lot đứng đầu hoặc Lot có tổng lỗi cao nhất, final "
                    "SELECT phải giữ kèm lot_id và product_id trong từng dòng kết "
                    "quả để câu trả lời nêu được ngữ cảnh Lot. Chỉ trả đúng JSON: "
                    '{"can_answer":true,"sql":"...","reason":"..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Semantic model:\n{semantic_model}\n\n"
                    f"Câu hỏi: {question}{retry}\n\n"
                    "Ví dụ suy luận: nếu hỏi top lỗi trong Lot có tổng lỗi cao "
                    "nhất, dùng CTE tìm lot_id đứng đầu từ "
                    "v_lot_error_summary, sau đó lọc "
                    "v_lot_error_breakdown theo lot_id, SUM(total_error_qty), "
                    "GROUP BY lot_id,product_id,error_id,error_name và lấy top N; "
                    "final SELECT gồm lot_id, product_id, error_id, error_name, "
                    "total_error_qty.\n"
                    "Nếu hỏi lỗi theo ngày/tháng/khoảng thời gian, dùng "
                    "v_error_details.error_time, tổng hợp bằng SUM(quantity), "
                    "nhóm theo date(error_time) hoặc strftime('%Y-%m', error_time). "
                    "Nếu hỏi top Lot có tổng lỗi cao nhất trong ngày/tháng/năm, "
                    "cũng dùng v_error_details.error_time rồi GROUP BY lot_id, "
                    "product_id. Chỉ dùng v_lot_error_summary.produce_date khi "
                    "câu hỏi nói rõ 'ngày sản xuất'."
                ),
            },
        ]

    @staticmethod
    def parse_plan(content: str) -> MesSqlPlan:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise MesSqlAgentError("LLM không trả về kế hoạch JSON hợp lệ.")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise MesSqlAgentError("LLM trả về JSON kế hoạch không hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise MesSqlAgentError("Kế hoạch SQL phải là một JSON object.")
        can_answer = payload.get("can_answer") is True
        sql = str(payload.get("sql") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if can_answer and not sql:
            raise MesSqlAgentError("Kế hoạch thiếu câu SELECT.")
        return MesSqlPlan(can_answer=can_answer, sql=sql, reason=reason)

    def semantic_model(self) -> dict[str, Any]:
        if self._semantic_model is None:
            try:
                payload = json.loads(
                    self.semantic_model_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise MesSqlAgentError("Không thể đọc MES semantic model.") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("views"), dict):
                raise MesSqlAgentError("MES semantic model không hợp lệ.")
            unknown_views = set(payload["views"]) - self.ALLOWED_VIEWS
            if unknown_views:
                raise MesSqlAgentError(
                    f"Semantic model chứa view không được phép: {sorted(unknown_views)}"
                )
            self._semantic_model = payload
        return self._semantic_model

    def validate_sql(self, sql: str) -> str:
        if not sql or len(sql) > self.max_sql_length:
            raise MesSqlAgentError("SQL rỗng hoặc vượt quá độ dài cho phép.")
        try:
            statements = parse(sql, read="sqlite")
        except ParseError as exc:
            raise MesSqlAgentError("SQL không đúng cú pháp SQLite.") from exc
        if len(statements) != 1:
            raise MesSqlAgentError("Chỉ được phép chạy đúng một câu SELECT.")
        statement = statements[0]
        if statement.find(exp.Select) is None:
            raise MesSqlAgentError("Chỉ được phép chạy SELECT.")

        for node in statement.walk():
            if node.key.lower() in self.PROHIBITED_NODE_KEYS:
                raise MesSqlAgentError(
                    f"SQL chứa thao tác không được phép: {node.key}."
                )

        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(exp.CTE)
            if cte.alias_or_name
        }
        referenced_views: set[str] = set()
        for table in statement.find_all(exp.Table):
            if table.catalog or table.db:
                raise MesSqlAgentError("Không được truy cập database hoặc schema khác.")
            table_name = table.name.lower()
            if table_name in cte_names:
                continue
            if table_name not in self.ALLOWED_VIEWS:
                raise MesSqlAgentError(f"View không được phép: {table.name}.")
            referenced_views.add(table_name)
        if not referenced_views:
            raise MesSqlAgentError("SQL phải truy vấn ít nhất một MES view.")

        if statement.args.get("limit") is None:
            statement = statement.limit(self.max_rows)
        return statement.sql(dialect="sqlite")

    def execute(self, sql: str) -> MesSqlQueryResult:
        safe_sql = self.validate_sql(sql)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only = ON")
                connection.set_authorizer(self._authorizer)
                connection.set_progress_handler(
                    lambda: 1 if time.monotonic() > deadline else 0,
                    1000,
                )
                connection.execute(f"EXPLAIN QUERY PLAN {safe_sql}").fetchall()
                cursor = connection.execute(safe_sql)
                raw_rows = cursor.fetchmany(self.max_rows + 1)
                columns = [item[0] for item in (cursor.description or [])]
        except sqlite3.Error as exc:
            message = "Truy vấn bị timeout." if "interrupted" in str(exc).lower() else "SQL không thể chạy trên MES snapshot."
            raise MesSqlAgentError(message) from exc

        truncated = len(raw_rows) > self.max_rows
        rows = [dict(row) for row in raw_rows[: self.max_rows]]
        return MesSqlQueryResult(
            columns=columns,
            rows=rows,
            imported_at=self._imported_at(),
            truncated=truncated,
        )

    def answer_messages(
        self,
        question: str,
        result: MesSqlQueryResult,
    ) -> list[dict[str, str]]:
        payload = json.dumps(
            result.prompt_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý MES MKAC. Chỉ trả lời từ JSON kết quả. "
                    "Trả lời tiếng Việt tự nhiên, đúng trọng tâm; giữ nguyên mã "
                    "và số liệu, dùng dấu chấm phân cách hàng nghìn. Không nhắc "
                    "SQL, JSON hoặc tên field kỹ thuật. Chỉ trả về câu trả lời "
                    "thuần, không bọc trong JSON. Nói rõ là MES snapshot. "
                    "Tên lỗi rỗng phải nói '*Lỗi chưa rõ tên*'. Không suy đoán."
                ),
            },
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nKết quả đã kiểm chứng:\n{payload}",
            },
        ]

    @staticmethod
    def fallback_answer(result: MesSqlQueryResult) -> str:
        if not result.rows:
            return "MES snapshot không có dữ liệu phù hợp với câu hỏi này."
        first_row = result.rows[0]
        error_quantity_column = next(
            (
                column
                for column in (
                    "error_quantity",
                    "total_error_qty",
                    "quantity",
                    "qty",
                    "error_count",
                )
                if column in first_row
            ),
            "",
        )
        if {"lot_id", "error_id"}.issubset(first_row) and error_quantity_column:
            lot_id = str(first_row.get("lot_id") or "chưa rõ")
            product_id = first_row.get("product_id")
            prefix = f"Theo MES snapshot, trong Lot {lot_id}"
            if product_id:
                prefix += f" của mã hàng {product_id}"
            lot_total = first_row.get("lot_total_error_qty")
            if lot_total is not None:
                prefix += (
                    f", tổng {MesSqlAgent._format_vietnamese_number(lot_total)} lỗi"
                )
            items = []
            for row in result.rows[:10]:
                error_id = str(row.get("error_id") or "chưa rõ")
                error_name = row.get("error_name")
                error_label = (
                    f"{error_id} - {error_name or '*Lỗi chưa rõ tên*'}"
                )
                items.append(
                    f"{error_label}: "
                    f"{MesSqlAgent._format_vietnamese_number(row.get(error_quantity_column))}"
                )
            return prefix + ", các loại lỗi nhiều nhất là " + "; ".join(items) + "."
        if {"lot_id", "product_id"}.issubset(first_row) and error_quantity_column:
            items = []
            for row in result.rows[:10]:
                items.append(
                    f"Lot {row.get('lot_id') or 'chưa rõ'}, mã hàng "
                    f"{row.get('product_id') or 'chưa rõ'}: "
                    f"{MesSqlAgent._format_vietnamese_number(row.get(error_quantity_column))}"
                )
            return (
                "Theo MES snapshot, các Lot có tổng lỗi cao nhất là "
                + "; ".join(items)
                + "."
            )
        if "error_id" in first_row and error_quantity_column:
            time_column = MesSqlAgent._find_time_column(first_row)
            time_prefix = ""
            if time_column:
                time_prefix = f" trong {first_row.get(time_column)}"
            items = []
            for row in result.rows[:10]:
                error_id = str(row.get("error_id") or "chưa rõ")
                error_name = row.get("error_name")
                items.append(
                    f"{error_id} - {error_name or '*Lỗi chưa rõ tên*'}: "
                    f"{MesSqlAgent._format_vietnamese_number(row.get(error_quantity_column))}"
                )
            return (
                f"Theo MES snapshot, các loại lỗi nhiều nhất{time_prefix} là "
                + "; ".join(items)
                + "."
            )
        time_column = MesSqlAgent._find_time_column(first_row)
        if time_column and error_quantity_column:
            items = []
            for row in result.rows[:10]:
                time_value = row.get(time_column)
                items.append(
                    f"{time_value or 'chưa rõ'}: "
                    f"{MesSqlAgent._format_vietnamese_number(row.get(error_quantity_column))}"
                )
            return (
                "Theo MES snapshot, kết quả lỗi theo thời gian là "
                + "; ".join(items)
                + "."
            )
        summaries = [
            " · ".join(
                MesSqlAgent._format_vietnamese_number(value)
                if isinstance(value, (int, float))
                else str(value) if value is not None else "chưa rõ"
                for value in row.values()
            )
            for row in result.rows[:10]
        ]
        return "Kết quả từ MES snapshot: " + "; ".join(summaries) + "."

    @staticmethod
    def _find_time_column(row: dict[str, Any]) -> str:
        preferred = (
            "error_date",
            "error_day",
            "day",
            "date",
            "error_month",
            "month",
            "period",
            "produce_date",
            "error_time",
        )
        for column in preferred:
            if column in row:
                return column
        for column in row:
            normalized = column.lower()
            if "date" in normalized or "time" in normalized or "month" in normalized:
                return column
        return ""

    @staticmethod
    def _format_vietnamese_number(value: Any) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return f"{value:,}".replace(",", ".")
        if isinstance(value, float) and value.is_integer():
            return f"{int(value):,}".replace(",", ".")
        if isinstance(value, float):
            return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        return str(value)

    def _imported_at(self) -> str:
        try:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key='imported_at' LIMIT 1"
                ).fetchone()
            return str(row[0]) if row else ""
        except sqlite3.Error:
            return ""

    def _authorizer(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        del database
        allowed_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        recursive = getattr(sqlite3, "SQLITE_RECURSIVE", None)
        if recursive is not None:
            allowed_actions.add(recursive)
        if action in allowed_actions:
            if action != sqlite3.SQLITE_READ:
                return sqlite3.SQLITE_OK
            table_name = (arg1 or "").lower()
            source_name = (source or "").lower()
            if (
                table_name in self.ALLOWED_VIEWS
                or table_name in self.INTERNAL_TABLES
                or source_name in self.ALLOWED_VIEWS
                or table_name == "sqlite_master"
            ):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg2 or arg1 or "").lower()
            return (
                sqlite3.SQLITE_OK
                if function_name in self.ALLOWED_FUNCTIONS
                else sqlite3.SQLITE_DENY
            )
        return sqlite3.SQLITE_DENY
