"""LLM-planned, read-only SQL fallback over the CCTVAI reporting replica.

Ported from :mod:`src.integrations.wms_sql_agent`, but the target is PostgreSQL
rather than SQLite, and that changes the safety story materially.

SQLite gave us four runtime guardrails; only two survive the port:

===========================  ==============================================
SQLite (WMS)                 PostgreSQL (CCTVAI)
===========================  ==============================================
``?mode=ro`` URI             role has ``default_transaction_read_only=on``
``PRAGMA query_only = ON``   ``SET TRANSACTION READ ONLY`` per transaction
``set_authorizer`` (deny)    **no equivalent — nothing replaces it**
``set_progress_handler``     server ``statement_timeout`` + client timeout
===========================  ==============================================

Losing ``set_authorizer`` means the sqlglot AST is the only client-side barrier
in front of a generated statement. It is NOT a sandbox and must never be
described as one. What actually keeps this safe is defence in depth:

* the replica role is read-only and cannot write, whatever we send it;
* GRANTs are column-level, so the camera credential columns are unreachable
  even if a statement asked for them;
* the connection sets an EMPTY ``search_path``, so the server itself rejects
  any unqualified table name — the ``cctvai.`` prefix rule is enforced
  server-side, not just by our AST walk;
* every answer produced through this module carries
  ``CCTVAI_SQL_AGENT_ANSWER_UNVERIFIED`` so the user sees it as low-confidence.

The hardest constraint is semantic, not syntactic: ``del_flag`` is this
schema's only de-duplication mechanism, and omitting it inflates counts by
+44% while the query still succeeds. A general checker for "did you de-dup
correctly" is not writable, so instead we NARROW THE ACCEPTED GRAMMAR: a
soft-delete table may appear in exactly three shapes, and anything else is
rejected with a message telling the planner how to rewrite. See
:meth:`CctvaiSqlAgent._validate_soft_delete_shapes`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import os

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from . import cctvai_contract as contract


class CctvaiSqlAgentError(RuntimeError):
    """Raised when a CCTVAI SQL plan is rejected or cannot be executed."""


@dataclass(frozen=True)
class CctvaiSqlPlan:
    can_answer: bool
    sql: str = ""
    reason: str = ""


@dataclass(frozen=True)
class CctvaiSqlQueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    latest_event_at: str = ""
    tables: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "source": "cctvai_reporting_replica",
            "replica_lag_state": contract.REPLICA_LAG_STATE_UNVERIFIED,
            "latest_event_at": self.latest_event_at,
            "tables": list(self.tables),
            "columns": self.columns,
            "rows": self.rows,
            "truncated": self.truncated,
        }

    def is_empty(self) -> bool:
        return not self.rows


class CctvaiSqlAgent:
    """Generate prompts and validate LLM-planned read-only CCTVAI SQL."""

    ALLOWED_TABLES = frozenset(contract.REPORTABLE_TABLES)
    SOFT_DELETE_TABLES = frozenset(contract.SOFT_DELETE_TABLES)
    REQUIRED_SCHEMA = contract.SOURCE_SCHEMA

    # Inherited from the WMS agent. `set`, `copy` and `command` matter far more
    # here: they are how `SET search_path`, `COPY ... TO PROGRAM` and assorted
    # utility statements would arrive.
    PROHIBITED_NODE_KEYS = frozenset(
        {
            "alter",
            "analyze",
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
            "lock",  # FOR UPDATE / FOR SHARE parse to this key
            "merge",
            "pragma",
            "refresh",
            "rollback",
            "set",
            "transaction",
            "truncate",
            "update",
            "use",
        }
    )

    # Allowlist is the primary barrier. Keyed on sqlglot's CANONICAL name, not
    # the Postgres spelling: verified live on sqlglot 30.11.0, a parsed
    # `to_timestamp(...)` becomes exp.UnixToTime whose sql_name() is
    # "UNIX_TO_TIME", and `now()` becomes "CURRENT_TIMESTAMP". Keying on the
    # Postgres spelling would reject every query in
    # config/cctvai_semantic_model.json's verified_query_examples.
    #
    # Note also that node.name is EMPTY for modelled functions — only
    # exp.Anonymous carries the raw name. Hence the two-branch check in
    # _validate_functions.
    ALLOWED_FUNCTION_NAMES = frozenset(
        {
            "ABS",
            "AND",
            "ANY_VALUE",
            "AVG",
            "CAST",
            "CEIL",
            "COALESCE",
            "CONCAT",
            "COUNT",
            "CURRENT_TIMESTAMP",
            "DATE_TRUNC",
            "EXISTS",
            "EXTRACT",
            "FLOOR",
            "GREATEST",
            "IF",
            "LEAST",
            "LENGTH",
            "LOWER",
            "LTRIM",
            "MAX",
            "MIN",
            "NULLIF",
            "OR",
            "ROUND",
            "RTRIM",
            "SUM",
            "TIME_TO_STR",
            "TRIM",
            "UNIX_TO_TIME",
            "UPPER",
        }
    )

    # Second layer behind the allowlist, matched against exp.Anonymous names,
    # so a future sqlglot release that stops modelling one of these still
    # cannot slip it through as a "known" function.
    DENIED_FUNCTION_PREFIXES = (
        "pg_",
        "dblink",
        "lo_",
        "pgp_",
        "current_setting",
        "set_config",
        "query_to_xml",
    )
    DENIED_SCHEMAS = frozenset({"pg_catalog", "information_schema", "pg_temp", "public"})

    def __init__(
        self,
        semantic_model_path: Path | str,
        *,
        max_rows: int = 50,
        max_sql_length: int = 8000,
    ):
        self.semantic_model_path = Path(semantic_model_path)
        self.max_rows = max(1, min(int(max_rows), 200))
        self.max_sql_length = max(500, int(max_sql_length))
        self._semantic_model: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "CctvaiSqlAgent | None":
        enabled = os.getenv("CCTVAI_SQL_AGENT_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(
            semantic_model_path=os.getenv(
                "CCTVAI_SEMANTIC_MODEL_PATH",
                "config/cctvai_semantic_model.json",
            ),
            max_rows=int(os.getenv("CCTVAI_SQL_AGENT_MAX_ROWS", "50")),
        )

    @property
    def available(self) -> bool:
        return self.semantic_model_path.is_file()

    def semantic_model(self) -> dict[str, Any]:
        if self._semantic_model is None:
            try:
                payload = json.loads(
                    self.semantic_model_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise CctvaiSqlAgentError(
                    "Không thể đọc CCTVAI semantic model."
                ) from exc
            if not isinstance(payload, dict) or not isinstance(
                payload.get("tables"), dict
            ):
                raise CctvaiSqlAgentError("CCTVAI semantic model không hợp lệ.")
            unknown = set(payload["tables"]) - self.ALLOWED_TABLES
            if unknown:
                raise CctvaiSqlAgentError(
                    f"Semantic model chứa bảng không được phép: {sorted(unknown)}"
                )
            # The model is what the planner "knows". If it ever describes a
            # column the role cannot read, the planner will emit SQL that dies
            # on a permission error — so refuse to load it at all.
            for table, spec in payload["tables"].items():
                columns = spec.get("columns")
                if not isinstance(columns, dict):
                    raise CctvaiSqlAgentError(
                        f"Semantic model thiếu mô tả cột cho bảng {table}."
                    )
                blocked = contract.unreadable_columns(table, columns)
                if blocked:
                    raise CctvaiSqlAgentError(
                        f"Semantic model mô tả cột không đọc được ở {table}: "
                        f"{blocked}"
                    )
            self._semantic_model = payload
        return self._semantic_model

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

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
        example_answer = json.dumps(
            {
                "can_answer": True,
                "sql": (
                    "SELECT c.camera_id, c.camera_name FROM cctvai.cctvai_cameras c "
                    f"WHERE NOT c.del_flag LIMIT {self.max_rows}"
                ),
                "reason": "danh sách camera đang hoạt động",
            },
            ensure_ascii=False,
        )
        return [
            {
                "role": "system",
                "content": (
                    "Bạn là bộ lập kế hoạch SQL cho CCTVAI (PostgreSQL, schema "
                    "cctvai). CHỈ xuất một JSON object, không viết gì khác — "
                    "không markdown, không code fence, không giải thích.\n"
                    'Định dạng: {"can_answer":true,"sql":"SELECT ...",'
                    '"reason":"..."}\n'
                    f"Ví dụ đúng: {example_answer}\n"
                    "Quy tắc SQL (vi phạm bất kỳ quy tắc nào → bị từ chối):\n"
                    "- Chỉ SELECT hoặc WITH...SELECT. Không DDL/DML/SET/COPY/"
                    "LISTEN/FOR UPDATE/pg_catalog/information_schema.\n"
                    "- Chỉ bảng/cột có trong semantic model, luôn viết "
                    "prefix cctvai.<bảng> (search_path rỗng).\n"
                    f"- BẮT BUỘC LIMIT (tối đa {self.max_rows}). Không "
                    "SELECT *, không count(*) — đếm số dòng bằng "
                    "count(<alias>.<cột thật của CHÍNH bảng đang đếm>), ví "
                    "dụ đếm camera thì count(c.camera_id); không cần join "
                    "thêm bảng nào chỉ để có cột đếm. Đếm event_snapshots "
                    "thì dùng count(<alias>.id).\n"
                    "- JOIN tới cctvai_cameras/cctvai_violation_types/"
                    "cctvai_lines: AND NOT <alias>.del_flag PHẢI nằm trong "
                    "ON, không phải WHERE (WHERE biến LEFT JOIN thành INNER "
                    "JOIN, mất dữ liệu). Bảng đó đứng đầu FROM: WHERE NOT "
                    "<alias>.del_flag. Trong EXISTS: NOT <alias>.del_flag "
                    "trong WHERE của subquery.\n"
                    "- detected_time/end_time là epoch milliseconds: "
                    "to_timestamp(cột / 1000.0).\n"
                    "- Không truy vấn đăng nhập/URL camera, không suy diễn "
                    "tên/email người phụ trách. can_answer=false chỉ khi câu "
                    "hỏi không thể biểu diễn bằng cột nào trong semantic "
                    "model.\n"
                    "JSON PHẢI có đủ CẢ BA field can_answer, sql, reason — "
                    "thiếu field nào cũng bị coi là từ chối, kể cả khi đã "
                    "viết sql. CHỈ trả JSON, không viết gì thêm."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Semantic model:\n{semantic_model}\n\n"
                    f"Câu hỏi: {question}{retry}\n\n"
                    "verified_query_examples trong semantic model là mẫu "
                    "đúng, rejected_query_examples là mẫu phải tránh. Chỉ "
                    "trả JSON."
                ),
            },
        ]

    @staticmethod
    def parse_plan(content: str) -> CctvaiSqlPlan:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise CctvaiSqlAgentError("LLM không trả về kế hoạch JSON hợp lệ.")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise CctvaiSqlAgentError(
                "LLM trả về JSON kế hoạch không hợp lệ."
            ) from exc
        if not isinstance(payload, dict):
            raise CctvaiSqlAgentError("Kế hoạch SQL phải là một JSON object.")
        sql = str(payload.get("sql") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if "can_answer" in payload:
            can_answer = payload.get("can_answer") is True
        else:
            # Small local coder models reliably emit {"sql": "..."} but often
            # drop the can_answer/reason metadata fields even when the prompt
            # repeats the requirement — verified empirically against
            # local-qwen-coder (see cctvai_sql_agent planner_messages). This
            # is NOT a security boundary: validate_sql() is what actually
            # gates execution, can_answer only decides whether to attempt
            # validation at all. Treating a present, non-empty sql as intent
            # to answer (when the model didn't bother restating the flag) is
            # safe and avoids discarding an otherwise-usable plan.
            can_answer = bool(sql)
        if can_answer and not sql:
            raise CctvaiSqlAgentError("Kế hoạch thiếu câu SELECT.")
        return CctvaiSqlPlan(can_answer=can_answer, sql=sql, reason=reason)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_sql(self, sql: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """Validate and normalize a planned statement.

        Returns ``(safe_sql, referenced_tables, reason_codes)``. Raises
        :class:`CctvaiSqlAgentError` with a planner-actionable message on any
        violation. Reason codes flag things that are permitted but must be
        disclosed in the answer (e.g. an INNER JOIN dropping orphan events).
        """
        if not sql or len(sql) > self.max_sql_length:
            raise CctvaiSqlAgentError("SQL rỗng hoặc vượt quá độ dài cho phép.")
        try:
            statements = parse(sql, read="postgres")
        except ParseError as exc:
            raise CctvaiSqlAgentError("SQL không đúng cú pháp PostgreSQL.") from exc
        statements = [item for item in statements if item is not None]
        if len(statements) != 1:
            raise CctvaiSqlAgentError("Chỉ được phép chạy đúng một câu SELECT.")
        statement = statements[0]
        if statement.find(exp.Select) is None:
            raise CctvaiSqlAgentError("Chỉ được phép chạy SELECT.")

        self._validate_node_keys(statement)
        self._validate_no_quoted_identifiers(statement)
        self._validate_no_star(statement)
        self._validate_functions(statement)
        referenced = self._validate_tables(statement)
        self._validate_columns(statement)
        self._validate_soft_delete_shapes(statement)
        self._validate_epoch_conversion(statement)
        self._validate_open_event_guard(statement)
        reason_codes = self._advisory_reason_codes(statement)

        if statement.args.get("limit") is None:
            statement = statement.limit(self.max_rows)
        return statement.sql(dialect="postgres"), referenced, reason_codes

    def _validate_node_keys(self, statement: exp.Expression) -> None:
        for node in statement.walk():
            key = node.key.lower()
            if key in self.PROHIBITED_NODE_KEYS:
                raise CctvaiSqlAgentError(
                    f"SQL chứa thao tác không được phép: {node.key}."
                )
            # LISTEN/NOTIFY/UNLISTEN and other utility statements land here.
            if isinstance(node, exp.Command):
                raise CctvaiSqlAgentError("SQL chứa câu lệnh tiện ích không được phép.")

    @staticmethod
    def _validate_no_quoted_identifiers(statement: exp.Expression) -> None:
        """Reject any double-quoted identifier (table, schema, column, alias).

        Every real object here (``cctvai``, ``event_snapshots``, ``del_flag``...)
        is plain lowercase snake_case; nothing in this schema ever legitimately
        needs quoting. Table/schema/column matching elsewhere in this class
        lower-cases before comparing (so ``CCTVAI``/``CcTvAi`` unquoted resolve
        identically, matching Postgres' own unquoted-identifier folding). But a
        QUOTED identifier — e.g. ``"CCTVAI".event_snapshots`` — is
        case-*preserving* in Postgres, so it verified as passing this validator
        (case-insensitive) while at execution time it addresses a literally
        different, case-sensitive object name than the intended ``cctvai``
        schema. There is no such object today so it only fails closed with a
        DB-level error, but relying on "no matching object exists yet" is not a
        security boundary — refusing quoting outright removes the ambiguity at
        the validator instead.
        """
        for identifier in statement.find_all(exp.Identifier):
            if identifier.quoted:
                raise CctvaiSqlAgentError(
                    "Không được dùng tên định danh trong dấu ngoặc kép "
                    f'("{identifier.this}"). Mọi bảng/cột/schema trong CCTVAI '
                    "đều viết thường không cần quoting."
                )

    @staticmethod
    def _validate_no_star(statement: exp.Expression) -> None:
        """Reject every ``*``, which covers two different hazards at once.

        ``SELECT *`` fails outright on ``cctvai_cameras`` because GRANTs are
        column-level, and ``count(*)`` counts duplicated rows when a JOIN has
        fanned out. One rule closes both.
        """
        if statement.find(exp.Star) is not None:
            raise CctvaiSqlAgentError(
                "Không được dùng '*'. Liệt kê cột tường minh (grant ở mức "
                "cột). Đang đếm số dòng thì dùng count(<alias>.<một cột có "
                "thật của CHÍNH bảng đang đếm>) — ví dụ đếm camera thì "
                "count(c.camera_id), KHÔNG cần join thêm bảng nào khác chỉ "
                "để có cột đếm. Chỉ khi đếm event_snapshots mới dùng "
                "count(<alias event_snapshots>.id)."
            )

    def _validate_functions(self, statement: exp.Expression) -> None:
        for node in statement.find_all(exp.Func):
            # exp.Anonymous is any function sqlglot does not model — pg_sleep,
            # dblink, pg_read_file all arrive this way (verified). Refusing the
            # whole class is stricter and more durable than naming them, and it
            # is the single rule that replaces SQLite's set_authorizer function
            # allowlist.
            if isinstance(node, exp.Anonymous):
                raw = (node.name or "").lower()
                raise CctvaiSqlAgentError(
                    f"Hàm không được phép: {raw or 'không rõ'}."
                )
            canonical = (node.sql_name() or "").upper()
            if canonical not in self.ALLOWED_FUNCTION_NAMES:
                raise CctvaiSqlAgentError(f"Hàm không được phép: {canonical.lower()}.")
            lowered = canonical.lower()
            if any(
                lowered.startswith(prefix)
                for prefix in self.DENIED_FUNCTION_PREFIXES
            ):
                raise CctvaiSqlAgentError(f"Hàm không được phép: {lowered}.")

    def _validate_tables(self, statement: exp.Expression) -> tuple[str, ...]:
        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(exp.CTE)
            if cte.alias_or_name
        }
        referenced: set[str] = set()
        for table in statement.find_all(exp.Table):
            table_name = (table.name or "").lower()
            if table_name in cte_names and not table.db:
                continue
            if table.catalog:
                raise CctvaiSqlAgentError(
                    "Không được truy cập database khác (cross-database)."
                )
            schema = (table.db or "").lower()
            # Inverted from the WMS agent: WMS rejects any schema prefix,
            # CCTVAI *requires* exactly `cctvai` because the role's search_path
            # is empty and unqualified names fail server-side anyway.
            if not schema:
                raise CctvaiSqlAgentError(
                    f"Bảng {table.name} thiếu prefix schema. Phải viết "
                    f"{self.REQUIRED_SCHEMA}.{table.name}."
                )
            if schema in self.DENIED_SCHEMAS or schema != self.REQUIRED_SCHEMA:
                raise CctvaiSqlAgentError(
                    f"Không được truy cập schema {table.db}. Chỉ dùng "
                    f"{self.REQUIRED_SCHEMA}."
                )
            if table_name not in self.ALLOWED_TABLES:
                raise CctvaiSqlAgentError(f"Bảng không được phép: {table.name}.")
            referenced.add(table_name)
        if not referenced:
            raise CctvaiSqlAgentError("SQL phải truy vấn ít nhất một bảng CCTVAI.")
        return tuple(sorted(referenced))

    def _validate_columns(self, statement: exp.Expression) -> None:
        """Reject any column the replica role cannot read.

        Fails closed here rather than letting PostgreSQL answer with
        ``permission denied``, which would leak the blocked column names into
        an error path and waste a round trip.
        """
        alias_to_table = self._alias_map(statement)
        for column in statement.find_all(exp.Column):
            name = (column.name or "").lower()
            qualifier = (column.table or "").lower()
            if qualifier and qualifier in alias_to_table:
                table = alias_to_table[qualifier]
                if not contract.is_column_readable(table, name):
                    raise CctvaiSqlAgentError(
                        f"Cột không đọc được hoặc không tồn tại: {qualifier}.{name}."
                    )
                continue
            # Unqualified, or qualified by a CTE/derived-table alias we cannot
            # resolve to a base table. Only the blocked-name check applies.
            if name in contract.BLOCKED_CAMERA_COLUMNS:
                raise CctvaiSqlAgentError(
                    f"Cột {name} không nằm trong quyền đọc của replica."
                )

    @staticmethod
    def _alias_map(statement: exp.Expression) -> dict[str, str]:
        """Map lowercase alias (or bare table name) to the base table name."""
        mapping: dict[str, str] = {}
        for table in statement.find_all(exp.Table):
            table_name = (table.name or "").lower()
            if table_name not in contract.REPORTABLE_TABLES:
                continue
            alias = (table.alias or "").lower() or table_name
            mapping[alias] = table_name
        return mapping

    # ------------------------------------------------------------------
    # The del_flag shape check
    # ------------------------------------------------------------------

    def _validate_soft_delete_shapes(self, statement: exp.Expression) -> None:
        """Require ``del_flag`` de-duplication in one of three known shapes.

        ``del_flag`` correctness is semantic: no general checker can decide
        whether an arbitrary statement de-duplicates properly. So instead of
        checking correctness, this narrows the accepted grammar. A soft-delete
        table is allowed in exactly these positions:

        =========================  ==========================================
        Shape                      Required predicate
        =========================  ==========================================
        Leading table in ``FROM``  ``WHERE NOT <alias>.del_flag``
        Any ``JOIN``               ``AND NOT <alias>.del_flag`` inside ``ON``
        Inside ``EXISTS``          ``NOT <alias>.del_flag`` in sub-``WHERE``
        =========================  ==========================================

        Everything else — a derived table, an implicit cross join
        (``FROM a, b``), a soft-delete table nested in a CTE we cannot follow —
        is rejected by default with a message telling the planner to rewrite.
        Rejecting a valid query is the acceptable failure direction; accepting
        a silently-inflated one is not.

        A LEFT JOIN carrying ``del_flag`` in ``WHERE`` instead of ``ON`` is
        called out specifically, because that is the subtle mistake: it turns
        the LEFT JOIN into an INNER JOIN and drops the 26 orphan events.
        """
        # find_all includes the statement itself when it is a Select, and each
        # EXISTS subquery arrives as its own scope — which is exactly the third
        # allowed shape, checked with the same rules as the outer query.
        for scope in statement.find_all(exp.Select):
            self._validate_scope_soft_delete(scope)

    def _validate_scope_soft_delete(self, select: exp.Select) -> None:
        where = select.args.get("where")
        where_expression = where.this if isinstance(where, exp.Where) else None

        # 1. Leading table in FROM.
        from_clause = _from_clause(select)
        if isinstance(from_clause, exp.From):
            leading = from_clause.this
            if isinstance(leading, exp.Table):
                table_name = (leading.name or "").lower()
                if table_name in self.SOFT_DELETE_TABLES:
                    alias = (leading.alias or "").lower() or table_name
                    if not self._has_del_flag_negation(where_expression, alias):
                        raise CctvaiSqlAgentError(
                            f"Bảng {leading.name} đứng dẫn đầu FROM phải có "
                            f"WHERE NOT {alias}.del_flag — đây là cách de-dup "
                            "duy nhất của schema này."
                        )
            elif leading is not None and self._contains_soft_delete_table(leading):
                raise CctvaiSqlAgentError(
                    "Bảng soft-delete xuất hiện ở vị trí không kiểm chứng được "
                    "(derived table/subquery trong FROM). Hãy viết lại thành "
                    "FROM cctvai.<bảng> trực tiếp hoặc JOIN có AND NOT "
                    "<alias>.del_flag trong ON."
                )

        # Implicit cross join: FROM a, b. sqlglot puts the extras here, and the
        # join condition has nowhere to live, so de-dup cannot be verified.
        for extra in select.args.get("joins") or []:
            if not isinstance(extra, exp.Join):
                continue
            target = extra.this
            if isinstance(target, exp.Table):
                table_name = (target.name or "").lower()
                if table_name not in self.SOFT_DELETE_TABLES:
                    continue
                alias = (target.alias or "").lower() or table_name
                on_expression = extra.args.get("on")
                if on_expression is None:
                    raise CctvaiSqlAgentError(
                        f"Bảng {target.name} được join không có mệnh đề ON "
                        "(cross join). Hãy dùng LEFT JOIN ... ON ... AND NOT "
                        f"{alias}.del_flag."
                    )
                if self._has_del_flag_negation(on_expression, alias):
                    continue
                if self._has_del_flag_negation(where_expression, alias):
                    raise CctvaiSqlAgentError(
                        f"NOT {alias}.del_flag đang ở WHERE thay vì trong ON "
                        f"của JOIN {target.name}. Đặt ở WHERE biến LEFT JOIN "
                        "thành INNER JOIN và âm thầm mất các sự kiện không "
                        "khớp master. Hãy chuyển vào mệnh đề ON."
                    )
                raise CctvaiSqlAgentError(
                    f"JOIN tới {target.name} thiếu AND NOT {alias}.del_flag "
                    "trong mệnh đề ON. Thiếu điều kiện này làm số dòng phồng "
                    "lên tới +44% mà query vẫn chạy thành công."
                )
            if target is not None and self._contains_soft_delete_table(target):
                raise CctvaiSqlAgentError(
                    "Bảng soft-delete nằm trong subquery được join — không "
                    "kiểm chứng được de-dup. Hãy join trực tiếp tới "
                    "cctvai.<bảng> với AND NOT <alias>.del_flag trong ON."
                )

    def _contains_soft_delete_table(self, node: exp.Expression) -> bool:
        return any(
            (table.name or "").lower() in self.SOFT_DELETE_TABLES
            for table in node.find_all(exp.Table)
        )

    @staticmethod
    def _has_del_flag_negation(
        expression: exp.Expression | None,
        alias: str,
    ) -> bool:
        """True when ``expression`` negates ``<alias>.del_flag``.

        Accepts the shapes a planner realistically emits: ``NOT a.del_flag``,
        ``a.del_flag = FALSE``, ``a.del_flag IS FALSE``, ``a.del_flag IS NOT
        TRUE``. Note that an unqualified ``del_flag`` is NOT accepted when an
        alias is in play — with three soft-delete tables in one statement, an
        unqualified reference is ambiguous about which one it de-dups.
        """
        if expression is None:
            return False

        def matches_column(node: exp.Expression | None) -> bool:
            if not isinstance(node, exp.Column):
                return False
            if (node.name or "").lower() != "del_flag":
                return False
            qualifier = (node.table or "").lower()
            return qualifier == alias

        for node in expression.walk():
            if isinstance(node, exp.Not) and matches_column(node.this):
                return True
            if isinstance(node, exp.EQ) and (
                (matches_column(node.this) and _is_false_literal(node.expression))
                or (matches_column(node.expression) and _is_false_literal(node.this))
            ):
                return True
            if isinstance(node, exp.Is):
                left, right = node.this, node.expression
                if matches_column(left) and _is_false_literal(right):
                    return True
                if (
                    matches_column(left)
                    and isinstance(right, exp.Not)
                    and _is_true_literal(right.this)
                ):
                    return True
            if (
                isinstance(node, exp.Not)
                and isinstance(node.this, exp.Is)
                and matches_column(node.this.this)
                and _is_true_literal(node.this.expression)
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # The other two silent-wrong-answer traps
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_epoch_conversion(statement: exp.Expression) -> None:
        """Require ``/ 1000.0`` inside every ``to_timestamp`` of an epoch column.

        ``detected_time`` and ``end_time`` hold epoch MILLISECONDS. Feeding one
        to ``to_timestamp`` unscaled yields a date tens of thousands of years
        out, and the query still succeeds — another mistake nothing downstream
        can catch.
        """
        for node in statement.find_all(exp.UnixToTime):
            argument = node.this
            if argument is None:
                continue
            columns = [
                column
                for column in argument.find_all(exp.Column)
                if (column.name or "").lower() in contract.EPOCH_MILLISECOND_COLUMNS
            ]
            if not columns:
                continue
            if not isinstance(argument, exp.Div):
                raise CctvaiSqlAgentError(
                    f"{columns[0].name} là epoch milliseconds. Phải viết "
                    f"to_timestamp({columns[0].sql(dialect='postgres')} / "
                    f"{contract.EPOCH_MS_DIVISOR})."
                )
            divisor = argument.expression
            divisor_text = (
                str(divisor.this) if isinstance(divisor, exp.Literal) else ""
            )
            try:
                divides_by_thousand = float(divisor_text) == 1000.0
            except ValueError:
                divides_by_thousand = False
            if not divides_by_thousand:
                raise CctvaiSqlAgentError(
                    f"{columns[0].name} phải chia đúng "
                    f"{contract.EPOCH_MS_DIVISOR}, không phải {divisor_text or '?'}."
                )

    @staticmethod
    def _validate_open_event_guard(statement: exp.Expression) -> None:
        """Require an ``end_time IS NOT NULL`` guard when computing a duration.

        166 events are still in progress with a NULL ``end_time``. Subtracting
        gives NULL, which sorts unpredictably and quietly skews any duration
        ranking.
        """
        uses_duration = False
        for node in statement.find_all(exp.Sub):
            names = {
                (column.name or "").lower() for column in node.find_all(exp.Column)
            }
            if "end_time" in names and "detected_time" in names:
                uses_duration = True
                break
        if not uses_duration:
            return
        for node in statement.find_all(exp.Is):
            if not isinstance(node.expression, exp.Null):
                continue
            column = node.this
            if not isinstance(column, exp.Column):
                continue
            if (column.name or "").lower() != "end_time":
                continue
            # `IS NOT NULL` parses as Not(Is(col, Null)).
            if isinstance(node.parent, exp.Not):
                return
        raise CctvaiSqlAgentError(
            "Tính thời lượng sự kiện phải kèm WHERE e.end_time IS NOT NULL — "
            "hiện có 166 sự kiện đang diễn ra với end_time NULL."
        )

    def _advisory_reason_codes(self, statement: exp.Expression) -> tuple[str, ...]:
        """Reason codes for things that are legal but must be disclosed."""
        codes: list[str] = []
        for join in statement.find_all(exp.Join):
            target = join.this
            if not isinstance(target, exp.Table):
                continue
            if (target.name or "").lower() not in self.SOFT_DELETE_TABLES:
                continue
            side = (join.side or "").upper()
            kind = (join.kind or "").upper()
            if side in {"LEFT", "FULL"}:
                continue
            if kind == "CROSS":
                continue
            # INNER JOIN from the event table: legal, but it drops the 26
            # events whose camera is absent from the master.
            codes.append(contract.REASON_INNER_JOIN_DROPS_ORPHANS)
            break
        return tuple(dict.fromkeys(codes))

    # ------------------------------------------------------------------
    # Answer shaping
    # ------------------------------------------------------------------

    def answer_messages(
        self,
        question: str,
        result: CctvaiSqlQueryResult,
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
                    "Bạn là trợ lý báo cáo CCTVAI. Chỉ trả lời từ JSON kết "
                    "quả, không thêm dữ liệu ngoài. "
                    f"{language_instruction} "
                    "Giữ nguyên mã camera và mã loại vi phạm. Không nhắc SQL, "
                    "JSON hay tên field kỹ thuật.\n"
                    "Cuối câu trả lời thêm một dòng cảnh báo ngắn: "
                    "\"⚠️ Dữ liệu từ replica, độ tin cậy thấp, độ trễ chưa "
                    "xác minh.\" — chỉ một dòng duy nhất, không lặp lại ở đầu.\n"
                    "Không suy diễn tên/email người phụ trách và không nhắc "
                    "thông tin đăng nhập camera. Camera không có trong master "
                    "thì giữ nguyên mã, không suy đoán tên. "
                    "Trình bày kết quả rõ ràng, xuống dòng cách đoạn hợp lý. "
                    "Khi có từ 2 mục hoặc nhiều số liệu, luôn trình bày mỗi mục "
                    "trên một gạch đầu dòng markdown riêng (bắt đầu bằng '- '), "
                    "in đậm thông tin chính, không dồn thành một đoạn văn dài."
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
            "cctvai.",
            "event_snapshots",
            "del_flag",
            "```sql",
            '{"answer"',
            '"rows"',
        )
        return not any(marker in normalized for marker in forbidden)

    @staticmethod
    def answer_matches_result(answer: str, result: CctvaiSqlQueryResult) -> bool:
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
        identifier_markers = ("camera_id", "violation_code", "violation_type")
        metric_markers = ("count", "total", "events", "sum")
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
        result: CctvaiSqlQueryResult,
        *,
        language: str = "vi",
    ) -> str:
        if result.is_empty():
            return (
                "CCTVAIレポートレプリカに該当するデータがありません。"
                if language == "ja"
                else "Replica báo cáo CCTVAI không có dữ liệu phù hợp với câu "
                "hỏi này."
            )
        lines = [
            "- " + ", ".join(self._display_parts(row, language=language))
            for row in result.rows
        ]
        prefix = (
            "CCTVAIレポートデータの結果："
            if language == "ja"
            else "Kết quả từ dữ liệu báo cáo CCTVAI:"
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
        result: CctvaiSqlQueryResult,
        *,
        language: str,
    ) -> list[str]:
        notices = [
            (
                "注意：この回答は自動生成した集計クエリによる推定であり、"
                "確定値としては未検証です。データはリアルタイムではありません。"
                if language == "ja"
                else "Lưu ý: câu trả lời này do hệ thống tự dựng truy vấn để "
                "suy ra, chưa được kiểm chứng như các mẫu báo cáo cố định; "
                "dữ liệu không phải thời gian thực."
            )
        ]
        if contract.REASON_INNER_JOIN_DROPS_ORPHANS in result.reason_codes:
            notices.append(
                (
                    "注意：INNER JOINのため、マスタに存在しないカメラの"
                    "イベントは集計から除外されています。"
                    if language == "ja"
                    else "Lưu ý: truy vấn dùng INNER JOIN nên đã loại các sự "
                    "kiện của camera không còn trong danh mục."
                )
            )
        if contract.REASON_FANOUT_SUSPECTED in result.reason_codes:
            notices.append(
                (
                    "警告：行数が実測ベースラインを超えており、JOINによる"
                    "行の重複が疑われます。数値は信頼できません。"
                    if language == "ja"
                    else "Cảnh báo: số dòng vượt mốc đã đo, nghi JOIN nhân bản "
                    "dòng. Không nên tin số liệu này."
                )
            )
        return notices

    @staticmethod
    def _display_parts(row: dict[str, Any], *, language: str) -> list[str]:
        """Render one result row using the shared contract display labels.

        Same labels/value formatting as the deterministic templates
        (``cctvai_database._format_rows``) so a user cannot tell which path
        answered from the wording alone.
        """
        parts = []
        for key, value in row.items():
            if value is None or value == "":
                continue
            label = contract.display_label(key, language)
            rendered = contract.format_display_value(key, value, language)
            parts.append(f"{label}: {rendered}")
        return parts or (["Không có giá trị"] if language != "ja" else ["値なし"])


def _from_clause(select: exp.Select) -> exp.From | None:
    """Return the ``FROM`` clause of ``select`` across sqlglot arg-key names.

    sqlglot 30.11.0 stores it under ``from_``; older releases used ``from``.
    Reading the wrong key returns None silently, which would make the whole
    leading-table ``del_flag`` check a no-op — verified the hard way.
    """
    clause = select.args.get("from_") or select.args.get("from")
    return clause if isinstance(clause, exp.From) else None


def _is_false_literal(node: exp.Expression | None) -> bool:
    if isinstance(node, exp.Boolean):
        return node.this is False
    return isinstance(node, exp.Literal) and str(node.this).lower() in {"false", "0"}


def _is_true_literal(node: exp.Expression | None) -> bool:
    if isinstance(node, exp.Boolean):
        return node.this is True
    return isinstance(node, exp.Literal) and str(node.this).lower() in {"true", "1"}
