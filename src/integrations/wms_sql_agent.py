"""LLM-planned, read-only SQL fallback over the WMS snapshot.

Mirrors :mod:`src.integrations.mes_sql_agent` but targets the WMS snapshot and
keeps its own allowlist. Answers produced here are LLM-planned and therefore
never contract-verified; callers must mark them as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os
import sqlite3
import time

from sqlglot import exp, parse
from sqlglot.errors import ParseError


class WmsSqlAgentError(RuntimeError):
    """Raised when a WMS SQL plan is rejected or cannot be executed."""


@dataclass(frozen=True)
class WmsSqlPlan:
    can_answer: bool
    sql: str = ""
    reason: str = ""


@dataclass(frozen=True)
class WmsSqlQueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    imported_at: str = ""
    truncated: bool = False
    domain: str = "CURRENT_BALANCE"
    evidence_domains: tuple[str, ...] = ("CURRENT_BALANCE",)

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "source": "mes_wms_snapshot",
            "snapshot_imported_at": self.imported_at,
            "domain": self.domain,
            "evidence_domains": list(self.evidence_domains),
            "columns": self.columns,
            "rows": self.rows,
            "truncated": self.truncated,
        }

    def is_empty(self) -> bool:
        return not self.rows


class WmsSqlAgent:
    """Generate prompts, validate SQL AST and execute read-only WMS queries."""

    ALLOWED_VIEWS = {
        "v_wms_current_balance_by_process_item",
        "v_wms_current_quality",
        "v_wms_legacy_archive_exact_key",
        "v_wms_raw_transaction_audit",
    }
    INTERNAL_TABLES = {
        "wms_current_balances",
        "wms_processes",
        "wms_legacy_archive_records",
        "wms_raw_transaction_headers",
        "wms_raw_transaction_details",
        "wms_raw_transaction_definitions",
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
        "cast",
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
        max_rows: int = 50,
        timeout_seconds: float = 2.0,
        max_sql_length: int = 8000,
    ):
        self.db_path = Path(db_path)
        self.semantic_model_path = Path(semantic_model_path)
        self.max_rows = max(1, min(int(max_rows), 200))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_sql_length = max(500, int(max_sql_length))
        self._semantic_model: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "WmsSqlAgent | None":
        enabled = os.getenv("WMS_SQL_AGENT_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(
            db_path=os.getenv("MES_WMS_DATABASE_PATH", "data/mes_wms.sqlite"),
            semantic_model_path=os.getenv(
                "WMS_SEMANTIC_MODEL_PATH",
                "config/wms_semantic_model.json",
            ),
            max_rows=int(os.getenv("WMS_SQL_AGENT_MAX_ROWS", "50")),
            timeout_seconds=float(os.getenv("WMS_SQL_AGENT_TIMEOUT", "2")),
        )

    @property
    def available(self) -> bool:
        return self.db_path.is_file() and self.semantic_model_path.is_file()

    def semantic_model(self) -> dict[str, Any]:
        if self._semantic_model is None:
            try:
                payload = json.loads(
                    self.semantic_model_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise WmsSqlAgentError("Không thể đọc WMS semantic model.") from exc
            if not isinstance(payload, dict) or not isinstance(
                payload.get("views"), dict
            ):
                raise WmsSqlAgentError("WMS semantic model không hợp lệ.")
            unknown_views = set(payload["views"]) - self.ALLOWED_VIEWS
            if unknown_views:
                raise WmsSqlAgentError(
                    f"Semantic model chứa view không được phép: {sorted(unknown_views)}"
                )
            self._semantic_model = payload
        return self._semantic_model

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
                    "Bạn là bộ lập kế hoạch SQL cho WMS snapshot SQLite của nhà "
                    "máy MKHC. Chỉ dùng view và cột trong semantic model. Chỉ sinh "
                    "một SELECT hoặc WITH...SELECT. Không dùng markdown, comment, "
                    "PRAGMA, ATTACH, DDL hay câu lệnh ghi. BẮT BUỘC có LIMIT (tối "
                    "đa 50).\n"
                    "Được phép lập SQL phân tích tự do trên các view allowlist, kể "
                    "cả phép cộng nhiều item_code, đối chiếu current/legacy/raw và "
                    "các proxy cho WIP, bottleneck, trend hay min-stock. Semantic "
                    "model mô tả các giới hạn của nguồn; hãy thể hiện giả định bằng "
                    "alias rõ ràng nhưng không tự từ chối chỉ vì giới hạn đó. Chỉ "
                    "đặt can_answer=false khi không thể biểu diễn câu hỏi bằng bất "
                    "kỳ cột nào trong các view allowlist.\n"
                    "Đếm mã vật tư hoặc đếm công đoạn thì hợp lệ vì không cần đơn "
                    "vị tính. Chỉ trả đúng JSON: "
                    '{"can_answer":true,"sql":"...","reason":"..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Semantic model:\n{semantic_model}\n\n"
                    f"Câu hỏi: {question}{retry}\n\n"
                    "Ví dụ suy luận: 'công đoạn nào có nhiều mã vật tư nhất' là "
                    "SELECT process_id, process_name, COUNT(DISTINCT item_code) "
                    "AS distinct_item_count FROM "
                    "v_wms_current_balance_by_process_item GROUP BY process_id, "
                    "process_name ORDER BY distinct_item_count DESC LIMIT 1. Chỉ "
                    "dùng LIMIT lớn hơn 1 khi câu hỏi yêu cầu danh sách hoặc top N.\n"
                    "LƯU Ý: quantity_decimal là TEXT, muốn so sánh số phải "
                    "CAST(quantity_decimal AS REAL) và chỉ trong cùng một "
                    "item_code. Khi lọc thời gian dùng latest_update_unix hoặc "
                    "trans_date_unix."
                ),
            },
        ]

    @staticmethod
    def parse_plan(content: str) -> WmsSqlPlan:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise WmsSqlAgentError("LLM không trả về kế hoạch JSON hợp lệ.")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise WmsSqlAgentError("LLM trả về JSON kế hoạch không hợp lệ.") from exc
        if not isinstance(payload, dict):
            raise WmsSqlAgentError("Kế hoạch SQL phải là một JSON object.")
        can_answer = payload.get("can_answer") is True
        sql = str(payload.get("sql") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if can_answer and not sql:
            raise WmsSqlAgentError("Kế hoạch thiếu câu SELECT.")
        return WmsSqlPlan(can_answer=can_answer, sql=sql, reason=reason)

    def validate_sql(self, sql: str) -> str:
        if not sql or len(sql) > self.max_sql_length:
            raise WmsSqlAgentError("SQL rỗng hoặc vượt quá độ dài cho phép.")
        try:
            statements = parse(sql, read="sqlite")
        except ParseError as exc:
            raise WmsSqlAgentError("SQL không đúng cú pháp SQLite.") from exc
        if len(statements) != 1:
            raise WmsSqlAgentError("Chỉ được phép chạy đúng một câu SELECT.")
        statement = statements[0]
        if statement.find(exp.Select) is None:
            raise WmsSqlAgentError("Chỉ được phép chạy SELECT.")

        for node in statement.walk():
            if node.key.lower() in self.PROHIBITED_NODE_KEYS:
                raise WmsSqlAgentError(
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
                raise WmsSqlAgentError("Không được truy cập database hoặc schema khác.")
            table_name = table.name.lower()
            if table_name in cte_names:
                continue
            if table_name not in self.ALLOWED_VIEWS:
                raise WmsSqlAgentError(f"View không được phép: {table.name}.")
            referenced_views.add(table_name)
        if not referenced_views:
            raise WmsSqlAgentError("SQL phải truy vấn ít nhất một WMS view.")

        if statement.args.get("limit") is None:
            statement = statement.limit(self.max_rows)
        return statement.sql(dialect="sqlite")

    def execute(self, sql: str) -> WmsSqlQueryResult:
        safe_sql = self.validate_sql(sql)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(
                uri, uri=True, timeout=self.timeout_seconds
            ) as connection:
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
            message = (
                "Truy vấn bị timeout."
                if "interrupted" in str(exc).lower()
                else "SQL không thể chạy trên WMS snapshot."
            )
            raise WmsSqlAgentError(message) from exc

        truncated = len(raw_rows) > self.max_rows
        rows = [dict(row) for row in raw_rows[: self.max_rows]]
        evidence_domains = self._evidence_domains_for_sql(safe_sql)
        return WmsSqlQueryResult(
            columns=columns,
            rows=rows,
            imported_at=self._imported_at(),
            truncated=truncated,
            domain=evidence_domains[0],
            evidence_domains=evidence_domains,
        )

    @staticmethod
    def _evidence_domains_for_sql(sql: str) -> tuple[str, ...]:
        lowered = sql.lower()
        domains = []
        if (
            "v_wms_current_balance_by_process_item" in lowered
            or "v_wms_current_quality" in lowered
        ):
            domains.append("CURRENT_BALANCE")
        if "v_wms_legacy_archive_exact_key" in lowered:
            domains.append("LEGACY_ARCHIVE")
        if "v_wms_raw_transaction_audit" in lowered:
            domains.append("RAW_TRANSACTION_AUDIT")
        return tuple(domains) or ("CURRENT_BALANCE",)

    def answer_messages(
        self,
        question: str,
        result: WmsSqlQueryResult,
        *,
        language: str = "vi",
    ) -> list[dict[str, str]]:
        payload = json.dumps(
            result.prompt_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        language_instruction = (
            "自然な日本語で回答してください。"
            if language == "ja"
            else "Trả lời bằng tiếng Việt tự nhiên."
        )
        return [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý WMS MKHC. Chỉ trả lời từ JSON kết quả, không "
                    "thêm dữ liệu ngoài. "
                    f"{language_instruction} "
                    "Giữ nguyên mã vật tư và mã công đoạn. Không nhắc SQL, JSON "
                    "hay tên field kỹ thuật. Nói rõ đây là WMS snapshot.\n"
                    "Số lượng trong snapshot KHÔNG có đơn vị tính: không gọi tên "
                    "đơn vị và phải nói rõ khi kết quả cộng nhiều mã vật tư. Nếu "
                    "SQL đối chiếu current, legacy hoặc raw thì phải nói rõ các "
                    "miền này khác grain và kết quả chỉ là suy luận. Với WIP, "
                    "bottleneck, trend hoặc min-stock, gọi rõ đó là proxy/suy luận "
                    "từ snapshot chứ không phải KPI nghiệp vụ đã xác minh. "
                    "process_name rỗng thì giữ nguyên mã công đoạn, không suy "
                    "đoán tên. Nếu kết quả có nhiều hơn 3 dòng, trình bày mỗi dòng trên "
                    "một mục gạch đầu dòng markdown riêng (bắt đầu bằng '- ')."
                ),
            },
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nKết quả truy vấn:\n{payload}",
            },
        ]

    @staticmethod
    def answer_is_natural(answer: str) -> bool:
        normalized = (answer or "").strip().lower()
        if not normalized:
            return False
        forbidden = (
            "select ",
            " from ",
            "v_wms_",
            "```sql",
            '{"answer"',
            '"rows"',
        )
        return not any(marker in normalized for marker in forbidden)

    @staticmethod
    def answer_matches_result(answer: str, result: WmsSqlQueryResult) -> bool:
        if result.is_empty():
            normalized = answer.lower()
            return any(
                marker in normalized
                for marker in (
                    "không có dữ liệu",
                    "không tìm thấy",
                    "データがありません",
                    "見つかりません",
                )
            )

        normalized = answer.lower()
        normalized_numbers = answer.replace(".", "").replace(",", "")
        identifier_markers = (
            "process_id",
            "item_code",
            "trans_id",
            "trans_code",
            "archive_id",
            "item_lot_id",
        )
        metric_markers = ("count", "quantity", "qty", "total", "sum")
        for row in result.rows[:5]:
            for key, value in row.items():
                if value in (None, ""):
                    continue
                key_lower = key.lower()
                if any(marker in key_lower for marker in identifier_markers):
                    if str(value).lower() not in normalized:
                        return False
                elif isinstance(value, (int, float)) and any(
                    marker in key_lower for marker in metric_markers
                ):
                    compact = str(int(value) if float(value).is_integer() else value)
                    if compact not in normalized_numbers:
                        return False
        return True

    def fallback_answer(
        self,
        result: WmsSqlQueryResult,
        *,
        language: str = "vi",
    ) -> str:
        if result.is_empty():
            return (
                "WMSスナップショットに該当するデータがありません。"
                if language == "ja"
                else "WMS snapshot không có dữ liệu phù hợp với câu hỏi này."
            )
        lines = [
            "- " + ", ".join(self._display_parts(row, language=language))
            for row in result.rows
        ]
        prefix = (
            "WMSスナップショットの結果："
            if language == "ja"
            else "Kết quả từ WMS snapshot:"
        )
        notices = self._unverified_notices(result, language=language)
        notice_text = "\n\n" + "\n".join(notices) if notices else ""
        suffix = (
            (
                f"\n（先頭{self.max_rows}件に制限しています。）"
                if language == "ja"
                else f"\n(Đã giới hạn {self.max_rows} dòng đầu.)"
            )
            if result.truncated
            else ""
        )
        return prefix + "\n" + "\n".join(lines) + notice_text + suffix

    @staticmethod
    def _unverified_notices(
        result: WmsSqlQueryResult,
        *,
        language: str,
    ) -> list[str]:
        keys = {
            key.lower()
            for row in result.rows
            for key in row
        }
        notices = []
        has_quantity_aggregate = any(
            ("quantity" in key or "qty" in key)
            and ("total" in key or "sum" in key)
            for key in keys
        )
        if has_quantity_aggregate:
            notices.append(
                (
                    "注意：この合計はUOM未標準化の生計算であり、業務上の"
                    "検証済み合計ではありません。"
                    if language == "ja"
                    else "Lưu ý: tổng này là phép tính thô chưa chuẩn hóa UOM, "
                    "không phải tổng nghiệp vụ đã kiểm chứng."
                )
            )
        if len(result.evidence_domains) > 1:
            notices.append(
                (
                    "注意：異なる粒度の複数データ領域を横断した未検証の比較です。"
                    if language == "ja"
                    else "Lưu ý: đây là đối chiếu chưa kiểm chứng giữa nhiều miền "
                    "dữ liệu có grain khác nhau."
                )
            )
        proxy_markers = ("wip", "bottleneck", "trend", "min_stock", "minimum_stock")
        if any(marker in key for key in keys for marker in proxy_markers):
            notices.append(
                (
                    "注意：この指標はスナップショットから推定した未検証の"
                    "プロキシです。"
                    if language == "ja"
                    else "Lưu ý: chỉ số này là proxy chưa kiểm chứng được suy ra "
                    "từ snapshot."
                )
            )
        return notices

    @staticmethod
    def _display_parts(row: dict[str, Any], *, language: str) -> list[str]:
        labels = {
            "process_id": ("Mã công đoạn", "工程コード"),
            "process_name": ("Tên công đoạn", "工程名"),
            "item_code": ("Mã vật tư", "品目コード"),
            "trans_id": ("Mã giao dịch", "トランザクションID"),
            "trans_code": ("Loại giao dịch", "トランザクション種別"),
            "archive_id": ("Mã lưu trữ", "アーカイブID"),
            "item_lot_id": ("Lot vật tư", "品目ロット"),
            "distinct_item_count": ("Số mã vật tư", "品目コード数"),
            "item_count": ("Số mã vật tư", "品目コード数"),
            "distinct_process_count": ("Số công đoạn", "工程数"),
            "process_count": ("Số công đoạn", "工程数"),
            "raw_quantity_sum": ("Tổng thô chưa chuẩn hóa UOM", "UOM未標準化の生合計"),
            "total_quantity": ("Tổng thô chưa chuẩn hóa UOM", "UOM未標準化の生合計"),
            "quantity_sum": ("Tổng thô chưa chuẩn hóa UOM", "UOM未標準化の生合計"),
        }
        parts = []
        for key, value in row.items():
            if value in (None, ""):
                continue
            label_pair = labels.get(key)
            label = label_pair[1 if language == "ja" else 0] if label_pair else key.replace("_", " ")
            parts.append(f"{label}: {value}")
        return parts or (["Không có giá trị"] if language != "ja" else ["値なし"])

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
