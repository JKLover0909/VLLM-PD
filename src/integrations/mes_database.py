"""Read-only query service for the local MES SQLite snapshot."""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.integrations.mes_answer_format import format_item_list


class MesDatabaseError(RuntimeError):
    """Raised when the MES snapshot cannot be queried safely."""


@dataclass(frozen=True)
class MesDatabaseResult:
    intent: str
    rows: list[dict[str, Any]]
    imported_at: str
    fallback_answer: str
    required_terms: tuple[str, ...] = ()

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "source": "mes_snapshot",
            "snapshot_imported_at": self.imported_at,
            "filters": {"exclude_test_data": True},
            "intent": self.intent,
            "rows": self.rows,
        }


def normalize_mes_text(value: str) -> str:
    original = value or ""
    normalized = unicodedata.normalize("NFD", original.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9_-]+", " ", normalized).strip()
    normalized = re.sub(r"\bnhiu\b", "nhieu", normalized)

    # UI tiếng Nhật được dịch nhẹ ở tầng giữa, nhưng các câu MES đơn giản nên
    # vẫn tự route được khi translation tắt hoặc dịch chưa ổn định. Ta append
    # các marker Việt/Anh để tái sử dụng bộ rule deterministic hiện có.
    jp_tokens: list[str] = []
    top_match = re.search(r"(?:上位|トップ)\s*(\d+)", original)
    if top_match:
        jp_tokens.append(f"top {top_match.group(1)}")
    if re.search(r"(ロット|Lot)", original):
        jp_tokens.append("lot")
    if re.search(r"(品番|製品|製品コード)", original):
        jp_tokens.append("ma hang san pham product")
    if re.search(r"(エラー|不良|欠陥)", original):
        jp_tokens.append("loi error defect")
    if re.search(r"(記録|件数|何件)", original):
        jp_tokens.append("ban ghi record bao nhieu")
    if re.search(r"(種類|異なる|エラーID|エラーコード)", original):
        jp_tokens.append("loai loi ma loi distinct")
    if re.search(r"(総|合計|総数)", original):
        jp_tokens.append("tong total sum")
    if re.search(r"(いくつ|何ロット|何種類|何人|何個)", original):
        jp_tokens.append("bao nhieu")
    if re.search(r"(最も|一番|最大|多い|最高)", original):
        jp_tokens.append("nhieu nhat cao nhat top most highest")
    if re.search(r"(少ない|最小|最低)", original):
        jp_tokens.append("it loi nhat thap nhat lowest")
    if re.search(r"(平均|1ロットあたり|ロットあたり)", original):
        jp_tokens.append("trung binh moi lot average per lot")
    if re.search(r"(比較|比べ|差)", original):
        jp_tokens.append("so sanh compare")
    if re.search(r"(一覧|列挙|リスト|すべて)", original):
        jp_tokens.append("liet ke danh sach list")
    if re.search(r"(工程|プロセス)", original):
        jp_tokens.append("process cong doan")
    if re.search(r"(進捗|現在の工程|最新の工程|どの工程)", original):
        jp_tokens.append("tien do dang o cong doan moi nhat")
    if re.search(r"(工程履歴|通過した工程|工程一覧)", original):
        jp_tokens.append("lich su liet ke cong doan da qua")
    if re.search(r"(数量|数はいくつ)", original):
        jp_tokens.append("so luong quantity")
    if re.search(r"(表形式|表で)", original):
        jp_tokens.append("bang table")
    if re.search(r"(作業者|作業員|生産者|生産しました)", original):
        jp_tokens.append("cong nhan nguoi san xuat operator worker")
    if re.search(r"(費用|修理費|コスト)", original):
        jp_tokens.append("chi phi cost expense")
    if re.search(r"(顧客|客先|お客様)", original):
        jp_tokens.append("khach hang customer")
    if re.search(r"(予測|予想|来月|将来)", original):
        jp_tokens.append("du doan du bao thang sau tuong lai forecast predict")
    if re.search(r"(一度も.*エラー.*ない|エラー.*ない製品|不良.*ない製品)", original):
        jp_tokens.append("chua tung bi loi khong bi loi never had error no error")
    if re.search(r"(数字.*1つ|1つの数字|数字だけ)", original):
        jp_tokens.append("mot so duy nhat one number only")

    if jp_tokens:
        normalized = " ".join(part for part in (normalized, *jp_tokens) if part)
    return normalized


class MesDatabase:
    """Allowlisted, parameterized queries against the MES snapshot."""

    LOT_PATTERN = re.compile(r"(?<!\d)\d{6}(?:-\d{2})?-\d{3}(?:-\d{2})?(?!\d)")
    SNAPSHOT_MARKERS = (
        "snapshot",
        "database",
        "co so du lieu",
        "du lieu cuc bo",
        "du lieu da luu",
    )
    DEFAULT_LIST_LIMIT = 25

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    @classmethod
    def from_env(cls) -> "MesDatabase | None":
        enabled = os.getenv("MES_DATABASE_ENABLED", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(os.getenv("MES_DATABASE_PATH", "data/mes.sqlite"))

    @property
    def available(self) -> bool:
        return self.db_path.is_file()

    def snapshot_version(self) -> str:
        """Phiên bản snapshot (imported_at) để làm khóa cache; rỗng nếu chưa có.

        Nhẹ hơn ``status()`` (chỉ một SELECT metadata) nên gọi được mỗi query.
        """
        if not self.available:
            return ""
        return self._imported_at()

    def status(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "db_path": str(self.db_path)}
        try:
            with self._connect() as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM schema_metadata")
                )
                raw_lots = int(metadata.get("lot_count", 0))
                raw_error_events = int(metadata.get("error_event_count", 0))
                display_lots = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM lots
                    WHERE {self._exclude_test_filter()}
                    """
                ).fetchone()[0]
                display_error_events = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM error_events AS e
                    LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
                    WHERE {self._exclude_test_filter("l.product_id", "e.lot_id")}
                    """
                ).fetchone()[0]
            return {
                "available": True,
                "db_path": str(self.db_path),
                "imported_at": metadata.get("imported_at", ""),
                "lots": int(display_lots or 0),
                "raw_lots": raw_lots,
                "excluded_test_lots": max(raw_lots - int(display_lots or 0), 0),
                "error_events": int(display_error_events or 0),
                "raw_error_events": raw_error_events,
                "excluded_test_error_events": max(
                    raw_error_events - int(display_error_events or 0),
                    0,
                ),
                "error_catalog": int(metadata.get("error_catalog_count", 0)),
                "process_steps": int(metadata.get("process_step_count", 0)),
                "orphan_process_steps": int(
                    metadata.get("orphan_process_step_count", 0)
                ),
                "unmapped_error_names": int(
                    metadata.get("unmapped_error_name_count", 0)
                ),
            }
        except (sqlite3.Error, ValueError) as exc:
            return {
                "available": False,
                "db_path": str(self.db_path),
                "error": str(exc),
            }

    def is_snapshot_question(self, question: str) -> bool:
        normalized = normalize_mes_text(question)
        return any(marker in normalized for marker in self.SNAPSHOT_MARKERS)

    def query_question(
        self,
        question: str,
        *,
        allow_highest_lot: bool = False,
    ) -> MesDatabaseResult | None:
        if not self.available:
            return None

        normalized = normalize_mes_text(question)
        unsupported_reason = self._unsupported_scope_reason(normalized)
        if unsupported_reason:
            return self._unsupported_scope(unsupported_reason)

        lot_id = self._extract_lot_id(question)
        error_name_query = self._extract_quoted_error_name(question)
        if self._is_lowest_lot_error_question(normalized):
            return self._lowest_error_lots(limit=self._extract_top_limit(normalized))
        lot_rank = self._extract_rank_position(normalized, question)
        if lot_rank and self._is_highest_lot_error_question(normalized):
            return self._ranked_error_lot(lot_rank)
        if allow_highest_lot and self._is_highest_lot_error_question(normalized):
            return self._highest_error_lots(limit=self._extract_top_limit(normalized))

        product_id = self._extract_code_after(
            question,
            (
                r"(?:mã\s+hàng|mã\s+sản\s+phẩm|sản\s+phẩm|"
                r"\bsp\b|\bproduct\s+code\b|\bproduct\s+id\b|"
                r"\bpart\s+number\b|\bitem\s+code\b|品番|製品|製品コード)"
            ),
        )
        error_id = self._extract_code_after(
            question,
            r"(?:mã\s+lỗi|lỗi\s+mã|\berror\s+code\b|\bdefect\s+code\b|エラーコード|エラーID)",
        )
        has_error = self._has_error_marker(normalized)

        if error_name_query:
            if self._asks_lots_for_error(normalized) or (
                "xuat hien" in normalized and re.search(r"\b(lot|lots|lo)\b", normalized)
            ):
                return self._lots_for_error_name(error_name_query)
            if self._asks_quantity(normalized):
                return self._error_quantity_by_name(error_name_query)
            return self._error_name_search(error_name_query)

        product_candidates = self._extract_product_candidates(question)
        if len(product_candidates) >= 2 and self._asks_compare(normalized):
            return self._compare_product_errors(product_candidates[:2])

        process_id = self._extract_process_id(question)
        if process_id and has_error:
            return self._process_error_types(process_id)

        # Câu đếm tổng chỉ áp dụng khi hỏi toàn hệ thống, không kèm mã cụ thể.
        # Nếu có lot_id/product_id/error_id, để các nhánh chi tiết bên dưới xử lý.
        if not lot_id and not product_id and not error_id:
            if self._is_ambiguous_lot_count_question(normalized):
                return self._ambiguous_lot_count()
            if self._is_count_error_records_question(normalized):
                return self._count_error_records(
                    number_only=self._asks_number_only(normalized)
                )
            if self._is_count_lots_with_errors_question(normalized):
                return self._count_lots_with_errors()
        if (
            self._is_lot_listing_question(normalized)
            and not lot_id
            and not product_id
            and not error_id
            and not has_error
        ):
            return self._list_lots()
        if lot_id and self._is_lot_process_question(normalized):
            if self._asks_process_metrics(normalized):
                return self._lot_process_metrics(lot_id)
            if self._asks_process_step_count(normalized):
                return self._lot_process_step_count(lot_id)
            if self._asks_process_history(normalized):
                return self._lot_process_steps(lot_id)
            return self._lot_process_progress(lot_id)
        if lot_id and self._is_lot_error_record_count_question(normalized):
            return self._lot_record_count(lot_id)
        if lot_id and self._is_lot_distinct_error_count_question(normalized):
            return self._lot_distinct_error_count(lot_id)
        if lot_id and has_error:
            return self._lot_error_breakdown(lot_id)
        if lot_id:
            return self._lot_details(lot_id)
        if error_id and self._asks_lots_for_error(normalized):
            return self._lots_for_error(error_id)
        if error_id:
            return self._error_name(error_id)
        product_rank = self._extract_rank_position(normalized, question)
        if product_rank and self._is_ranked_product_error_question(normalized, question):
            return self._ranked_error_product(product_rank)
        if self._is_highest_product_error_question(normalized):
            return self._highest_error_products(
                limit=self._extract_top_limit(
                    normalized, unit_pattern=self.PRODUCT_UNIT_PATTERN
                )
            )
        if product_id and self._asks_average_per_lot(normalized):
            return self._product_average_errors_per_lot(product_id)
        if product_id and self._asks_product_lot_count(normalized) and self._asks_total_quantity(normalized):
            return self._product_summary(product_id)
        if product_id and self._asks_product_lot_count(normalized):
            return self._product_lot_count(product_id)
        if product_id and has_error:
            if self._asks_breakdown(normalized):
                return self._product_error_breakdown(product_id)
            return self._product_summary(product_id)
        return None

    def _connect(self) -> sqlite3.Connection:
        try:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=3.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error as exc:
            raise MesDatabaseError("Không thể mở MES snapshot ở chế độ chỉ đọc.") from exc

    def _fetch_all(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                return [dict(row) for row in connection.execute(sql, parameters)]
        except sqlite3.Error as exc:
            raise MesDatabaseError("Không thể truy vấn MES snapshot.") from exc

    def _imported_at(self) -> str:
        rows = self._fetch_all(
            "SELECT value FROM schema_metadata WHERE key = 'imported_at' LIMIT 1"
        )
        return str(rows[0]["value"]) if rows else ""

    def _result(
        self,
        intent: str,
        rows: list[dict[str, Any]],
        answer: str,
        required_terms: tuple[str, ...] = (),
    ) -> MesDatabaseResult:
        return MesDatabaseResult(
            intent=intent,
            rows=rows,
            imported_at=self._imported_at(),
            fallback_answer=answer,
            required_terms=required_terms,
        )

    @staticmethod
    def _exclude_test_filter(
        product_column: str | None = "product_id",
        lot_column: str | None = "lot_id",
    ) -> str:
        # Dữ liệu test trong MES đang có nhiều kiểu tên: Test_..., Testlot,
        # M_Test_..., 1504_DTest_... Vì vậy phải loại theo token "test"
        # ở cả mã hàng và mã Lot, không chỉ prefix Test_.
        clauses: list[str] = []
        if product_column:
            clauses.append(
                f"LOWER(COALESCE({product_column}, '')) NOT LIKE '%test%'"
            )
        if lot_column:
            clauses.append(f"LOWER(COALESCE({lot_column}, '')) NOT LIKE '%test%'")
        return " AND ".join(clauses) if clauses else "1 = 1"

    def _highest_error_lots(self, limit: int = 1) -> MesDatabaseResult:
        exclude_test = self._exclude_test_filter()
        if limit > 1:
            rows = self._fetch_all(
                f"""
                SELECT lot_id, product_id, total_error_qty, error_record_count,
                       distinct_error_count, unmapped_error_record_count
                FROM v_lot_error_summary
                WHERE {exclude_test}
                  AND total_error_qty > 0
                ORDER BY total_error_qty DESC, lot_id
                LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = self._fetch_all(
                f"""
                SELECT lot_id, product_id, total_error_qty, error_record_count,
                       distinct_error_count, unmapped_error_record_count
                FROM v_lot_error_summary
                WHERE {exclude_test}
                  AND total_error_qty = (
                    SELECT MAX(total_error_qty) FROM v_lot_error_summary
                    WHERE {exclude_test}
                  )
                ORDER BY lot_id
                """
            )
        # Lấy thêm top lỗi chi tiết của mỗi Lot để model có thể trình bày phong phú hơn
        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            top_errors = self._fetch_all(
                f"""
                SELECT error_id, error_name, total_error_qty AS error_qty
                FROM v_lot_error_breakdown
                WHERE lot_id = ?
                  AND {self._exclude_test_filter()}
                ORDER BY total_error_qty DESC, error_id
                LIMIT 3
                """,
                (row["lot_id"],),
            )
            enriched_rows.append({**row, "top_errors": top_errors})
        answer = self._format_ranked_lots_with_errors(enriched_rows)
        terms = tuple(
            str(value)
            for row in rows
            for value in (row["lot_id"], row["product_id"], row["total_error_qty"])
        )
        return self._result("highest_error_lot", enriched_rows, answer, terms)

    def _lowest_error_lots(self, limit: int = 1) -> MesDatabaseResult:
        exclude_test = self._exclude_test_filter()
        if limit > 1:
            rows = self._fetch_all(
                f"""
                SELECT lot_id, product_id, total_error_qty, error_record_count,
                       distinct_error_count
                FROM v_lot_error_summary
                WHERE {exclude_test}
                  AND total_error_qty > 0
                ORDER BY total_error_qty ASC, lot_id
                LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = self._fetch_all(
                f"""
                SELECT lot_id, product_id, total_error_qty, error_record_count,
                       distinct_error_count
                FROM v_lot_error_summary
                WHERE {exclude_test}
                  AND total_error_qty = (
                    SELECT MIN(total_error_qty) FROM v_lot_error_summary
                    WHERE {exclude_test} AND total_error_qty > 0
                )
                ORDER BY lot_id
                """
            )
        if not rows:
            return self._result(
                "lowest_error_lot",
                [],
                "MES snapshot chưa có dữ liệu Lot có lỗi.",
            )
        answer = "Theo MES snapshot, Lot có tổng lỗi thấp nhất là: " + format_item_list(
            [
                f"Lot {row['lot_id']}, mã hàng {row['product_id']}, "
                f"{self._number(row['total_error_qty'])} lỗi"
                for row in rows
            ]
        ) + "."
        terms = tuple(
            str(value)
            for row in rows
            for value in (row["lot_id"], row["product_id"], row["total_error_qty"])
        )
        return self._result("lowest_error_lot", rows, answer, terms)

    def _ranked_error_lot(self, rank: int) -> MesDatabaseResult:
        exclude_test = self._exclude_test_filter()
        rows = self._fetch_all(
            f"""
            SELECT lot_id, product_id, total_error_qty, error_record_count,
                   distinct_error_count, unmapped_error_record_count
            FROM v_lot_error_summary
            WHERE {exclude_test}
              AND total_error_qty > 0
            ORDER BY total_error_qty DESC, lot_id
            LIMIT 1 OFFSET ?
            """,
            (max(0, rank - 1),),
        )
        if not rows:
            return self._result(
                "ranked_error_lot",
                [],
                f"MES snapshot chưa có Lot đứng thứ {rank} theo tổng lỗi.",
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, Lot đứng thứ {rank} theo tổng lỗi là "
            f"Lot {row['lot_id']}, mã hàng {row['product_id']}, "
            f"{self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "ranked_error_lot",
            rows,
            answer,
            (str(row["lot_id"]), str(row["product_id"]), str(row["total_error_qty"])),
        )

    def _count_lots_with_errors(self) -> MesDatabaseResult:
        exclude_test = self._exclude_test_filter()
        rows = self._fetch_all(
            f"""
            SELECT COUNT(*) AS lot_count
            FROM v_lot_error_summary
            WHERE {exclude_test} AND error_record_count > 0
            """
        )
        count = int(rows[0]["lot_count"]) if rows else 0
        answer = (
            f"Theo MES snapshot, có {self._number(count)} Lot đang ghi nhận "
            f"dữ liệu lỗi."
        )
        return self._result(
            "count_lots_with_errors",
            [{"lot_count": count}],
            answer,
            (str(count),),
        )

    def _ambiguous_lot_count(self) -> MesDatabaseResult:
        answer = (
            "Câu hỏi chưa rõ phạm vi: bạn muốn đếm tất cả Lot trong MES snapshot "
            "hay chỉ các Lot có ghi nhận lỗi?"
        )
        return self._result("ambiguous_lot_count", [], answer)

    def _count_error_records(self, *, number_only: bool = False) -> MesDatabaseResult:
        exclude_test = self._exclude_test_filter("l.product_id", "e.lot_id")
        rows = self._fetch_all(
            f"""
            SELECT COUNT(*) AS record_count
            FROM error_events AS e
            LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
            WHERE {exclude_test}
            """
        )
        count = int(rows[0]["record_count"]) if rows else 0
        answer = (
            str(count)
            if number_only
            else (
                f"Theo MES snapshot, tổng số bản ghi lỗi (error records) là "
                f"{self._number(count)}."
            )
        )
        return self._result(
            "count_error_records",
            [{"record_count": count}],
            answer,
            (str(count),),
        )

    def _list_lots(self, limit: int = DEFAULT_LIST_LIMIT) -> MesDatabaseResult:
        exclude_test_lots = self._exclude_test_filter()
        exclude_test_lots_alias = self._exclude_test_filter(
            "l.product_id", "l.lot_id"
        )
        count_rows = self._fetch_all(
            f"""
            SELECT COUNT(*) AS total_lot_count
            FROM lots
            WHERE {exclude_test_lots}
            """
        )
        total_lot_count = int(count_rows[0]["total_lot_count"]) if count_rows else 0
        rows = self._fetch_all(
            f"""
            SELECT l.lot_id, l.product_id, l.status, l.pcs_lot, l.produce_date,
                   COALESCE(s.total_error_qty, 0) AS total_error_qty
            FROM lots AS l
            LEFT JOIN v_lot_error_summary AS s ON s.lot_pk = l.lot_pk
            WHERE {exclude_test_lots_alias}
            ORDER BY
                CASE WHEN l.produce_date IS NULL THEN 1 ELSE 0 END,
                l.produce_date DESC,
                l.lot_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        if not rows:
            return self._result("list_lots", [], "MES snapshot chưa có dữ liệu Lot.")

        descriptions = format_item_list(
            [
                f"{row['lot_id']} ({row['product_id']}): "
                f"{self._number(row['pcs_lot'])} PCS, "
                f"{self._number(row['total_error_qty'])} lỗi"
                for row in rows
            ]
        )
        answer = (
            f"Theo MES snapshot, hiện có {self._number(total_lot_count)} Lot "
            f"đủ điều kiện hiển thị. Dưới đây là {len(rows)} Lot mới nhất: "
            f"{descriptions}."
        )
        return self._result(
            "list_lots",
            [{"total_lot_count": total_lot_count, "items": rows}],
            answer,
            (str(total_lot_count), str(rows[0]["lot_id"]), str(rows[0]["product_id"])),
        )

    def _lot_details(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT s.lot_id, s.product_id, l.status, l.is_release, l.pcs_lot,
                   l.produce_date, l.release_date, s.error_record_count,
                   s.distinct_error_count, s.total_error_qty,
                   s.unmapped_error_record_count
            FROM v_lot_error_summary AS s
            JOIN lots AS l ON l.lot_pk = s.lot_pk
            WHERE s.lot_id = ?
            LIMIT 1
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_details",
                [],
                f"Không tìm thấy Lot {lot_id} trong MES snapshot.",
                (lot_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, Lot {row['lot_id']} thuộc mã hàng "
            f"{row['product_id']}, mã trạng thái "
            f"{row['status'] or 'chưa rõ'} (snapshot chưa có bảng giải nghĩa), "
            f"có {self._number(row['pcs_lot'])} PCS, "
            f"{self._number(row['error_record_count'])} bản ghi lỗi, "
            f"{self._number(row['distinct_error_count'])} loại lỗi khác nhau "
            f"và tổng {self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "lot_details",
            rows,
            answer,
            (str(row["lot_id"]), str(row["product_id"]), str(row["total_error_qty"])),
        )

    def _lot_process_steps(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, route_id, process_id, process_order,
                   t1_date, t2_date, t3_date, t4_date, is_move_step,
                   moving_status
            FROM v_lot_process_steps
            WHERE lot_id = ?
            ORDER BY process_order, process_step_pk
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_process_steps",
                [],
                f"Không tìm thấy lịch sử công đoạn của Lot {lot_id} "
                "trong MES snapshot.",
                (lot_id,),
            )
        descriptions = format_item_list(
            [
                f"bước {self._number(row['process_order'])}: "
                f"process {row['process_id']}"
                for row in rows
            ]
        )
        answer = (
            f"Theo MES snapshot, Lot {lot_id} có {len(rows)} công đoạn đã ghi nhận "
            f"theo thứ tự: {descriptions}. Đây là lịch sử ghi nhận từ D_MAIN, "
            "không phải kế hoạch sản xuất."
        )
        return self._result(
            "lot_process_steps",
            rows,
            answer,
            (lot_id, str(rows[0]["process_id"]), str(rows[-1]["process_id"])),
        )

    def _lot_process_metrics(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, process_id, process_order,
                   p_ok, p_ng_defect, p_ng_scrap,
                   s_ok, s_ng_defect, s_ng_scrap,
                   b_ok, b_ng_defect, b_ng_scrap,
                   output_max_p, output_max_s, output_max_b
            FROM v_lot_process_steps
            WHERE lot_id = ?
            ORDER BY process_order, process_step_pk
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_process_metrics",
                [],
                f"Không tìm thấy số liệu công đoạn của Lot {lot_id} "
                "trong MES snapshot.",
                (lot_id,),
            )
        descriptions = format_item_list(
            [
                self._format_process_metrics(row)
                for row in rows
            ]
        )
        answer = (
            f"Theo MES snapshot, số liệu P/S/B đã ghi nhận của Lot {lot_id} là: "
            f"{descriptions}. Giá trị chưa rõ không được tính là 0; các nhóm P/S/B "
            "được giữ riêng và không phải tổng lỗi D_ERROR hay tỷ lệ yield."
        )
        return self._result(
            "lot_process_metrics",
            rows,
            answer,
            (lot_id, str(rows[0]["process_id"])),
        )

    @classmethod
    def _format_process_metrics(cls, row: dict[str, Any]) -> str:
        values = (
            (
                "P",
                row.get("p_ok"),
                row.get("p_ng_defect"),
                row.get("p_ng_scrap"),
                row.get("output_max_p"),
            ),
            (
                "S",
                row.get("s_ok"),
                row.get("s_ng_defect"),
                row.get("s_ng_scrap"),
                row.get("output_max_s"),
            ),
            (
                "B",
                row.get("b_ok"),
                row.get("b_ng_defect"),
                row.get("b_ng_scrap"),
                row.get("output_max_b"),
            ),
        )
        groups = ", ".join(
            f"{name}[OK={cls._number(ok)}, NG defect={cls._number(defect)}, "
            f"NG scrap={cls._number(scrap)}, output max={cls._number(output_max)}]"
            for name, ok, defect, scrap, output_max in values
        )
        return (
            f"bước {cls._number(row['process_order'])} process "
            f"{row['process_id']}: {groups}"
        )

    def _lot_process_step_count(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, COUNT(*) AS step_count
            FROM v_lot_process_steps
            WHERE lot_id = ?
            GROUP BY lot_id, product_id
            LIMIT 1
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_process_step_count",
                [],
                f"Không tìm thấy lịch sử công đoạn của Lot {lot_id} "
                "trong MES snapshot.",
                (lot_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, Lot {lot_id} có "
            f"{self._number(row['step_count'])} công đoạn đã ghi nhận trong D_MAIN."
        )
        return self._result(
            "lot_process_step_count",
            rows,
            answer,
            (lot_id, str(row["step_count"])),
        )

    def _lot_process_progress(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, route_id, step_count,
                   latest_process_id, latest_process_order, latest_recorded_at,
                   is_move_step, moving_status, lot_mapped
            FROM v_lot_process_progress
            WHERE lot_id = ?
            LIMIT 1
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_process_progress",
                [],
                f"Không tìm thấy tiến độ công đoạn của Lot {lot_id} "
                "trong MES snapshot.",
                (lot_id,),
            )
        row = rows[0]
        recorded_at = row.get("latest_recorded_at") or "chưa rõ thời điểm"
        answer = (
            f"Theo MES snapshot, bước được ghi nhận mới nhất của Lot {lot_id} là "
            f"process {row['latest_process_id']} (thứ tự "
            f"{self._number(row['latest_process_order'])}), thời điểm "
            f"{recorded_at}. Tổng cộng có {self._number(row['step_count'])} "
            "công đoạn đã ghi nhận. Đây là bước mới nhất trong snapshot D_MAIN, "
            "không phải diễn giải trạng thái hay kế hoạch sản xuất."
        )
        return self._result(
            "lot_process_progress",
            rows,
            answer,
            (
                lot_id,
                str(row["latest_process_id"]),
                str(row["latest_process_order"]),
            ),
        )

    def _lot_record_count(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, error_record_count, total_error_qty
            FROM v_lot_error_summary
            WHERE lot_id = ?
            LIMIT 1
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_error_record_count",
                [],
                f"Không tìm thấy Lot {lot_id} trong MES snapshot.",
                (lot_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, Lot {lot_id} có "
            f"{self._number(row['error_record_count'])} bản ghi lỗi, "
            f"tổng {self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "lot_error_record_count",
            rows,
            answer,
            (lot_id, str(row["error_record_count"])),
        )

    def _lot_distinct_error_count(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, distinct_error_count, total_error_qty
            FROM v_lot_error_summary
            WHERE lot_id = ?
            LIMIT 1
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_distinct_error_count",
                [],
                f"Không tìm thấy Lot {lot_id} trong MES snapshot.",
                (lot_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, Lot {lot_id} có "
            f"{self._number(row['distinct_error_count'])} loại lỗi khác nhau."
        )
        return self._result(
            "lot_distinct_error_count",
            rows,
            answer,
            (lot_id, str(row["distinct_error_count"])),
        )

    def _lot_error_breakdown(self, lot_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT lot_id, product_id, error_id, error_name, process_id,
                   error_type, error_record_count, total_error_qty
            FROM v_lot_error_breakdown
            WHERE lot_id = ?
            ORDER BY total_error_qty DESC, error_id, process_id
            LIMIT 10
            """,
            (lot_id,),
        )
        if not rows:
            return self._result(
                "lot_error_breakdown",
                [],
                f"Không tìm thấy dữ liệu lỗi của Lot {lot_id} trong MES snapshot.",
                (lot_id,),
            )
        descriptions = format_item_list(
            [
                f"{row['error_id']} ({row['error_name'] or '*Lỗi chưa rõ tên*'}): "
                f"{self._number(row['total_error_qty'])}"
                for row in rows
            ]
        )
        answer = (
            f"Theo MES snapshot, các lỗi có số lượng cao nhất của Lot {lot_id} là: "
            f"{descriptions}."
        )
        first = rows[0]
        terms = (lot_id, str(first["error_id"]), str(first["total_error_qty"]))
        return self._result("lot_error_breakdown", rows, answer, terms)

    def _error_name(self, error_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT error_id, process_id, error_type,
                   COALESCE(error_name_vi, error_name, error_name_en) AS error_name,
                   error_name_vi, error_name_en
            FROM error_catalog
            WHERE error_id = ? AND is_canonical = 1
            ORDER BY process_id, error_type
            LIMIT 10
            """,
            (error_id,),
        )
        if not rows:
            return self._result(
                "error_name",
                [],
                f"Chưa tìm thấy tên của mã lỗi {error_id} trong MES snapshot.",
                (error_id,),
            )
        names = []
        for row in rows:
            description = row["error_name"] or "*Lỗi chưa rõ tên*"
            names.append(f"{description} tại công đoạn {row['process_id']}")
        answer = f"Trong MES snapshot, mã lỗi {error_id} được ghi nhận là " + "; ".join(names) + "."
        terms = (error_id, str(rows[0]["error_name"] or ""))
        return self._result("error_name", rows, answer, terms)

    def _error_name_search(self, error_name_query: str) -> MesDatabaseResult:
        rows = self._matching_error_name_rows(error_name_query, grouped=True)
        if not rows:
            return self._result(
                "error_name_search",
                [],
                f"Không tìm thấy lỗi \"{error_name_query}\" trong MES snapshot.",
                (error_name_query,),
            )
        answer = (
            f"Trong MES snapshot, lỗi \"{error_name_query}\" được ghi nhận ở: "
            + format_item_list(
                [
                    f"mã lỗi {row['error_id']} - {row['error_name']}, "
                    f"process {row['process_id']}, "
                    f"quantity {self._number(row['total_error_qty'])}"
                    for row in rows[:10]
                ]
            )
            + "."
        )
        first = rows[0]
        return self._result(
            "error_name_search",
            rows,
            answer,
            (
                str(first["error_id"]),
                str(first["error_name"]),
                str(first["process_id"]),
            ),
        )

    def _error_quantity_by_name(self, error_name_query: str) -> MesDatabaseResult:
        rows = self._matching_error_name_rows(error_name_query, grouped=False)
        if not rows:
            return self._result(
                "error_quantity_by_name",
                [],
                f"Không tìm thấy lỗi \"{error_name_query}\" trong MES snapshot.",
                (error_name_query,),
            )
        top = rows[0]
        total = sum(int(row["quantity"] or 0) for row in rows)
        answer = (
            f"Theo MES snapshot, lỗi \"{error_name_query}\" có tổng quantity "
            f"{self._number(total)}. Bản ghi quantity cao nhất là "
            f"{self._number(top['quantity'])} tại Lot {top['lot_id']}, "
            f"mã hàng {top['product_id']}, process {top['process_id']}."
        )
        return self._result(
            "error_quantity_by_name",
            [{"total_error_qty": total, "top_record": top, "items": rows[:10]}],
            answer,
            (str(top["quantity"]), str(top["lot_id"]), str(top["process_id"])),
        )

    def _lots_for_error_name(self, error_name_query: str) -> MesDatabaseResult:
        rows = self._matching_error_name_rows(error_name_query, grouped_by_lot=True)
        if not rows:
            return self._result(
                "lots_for_error_name",
                [],
                f"Không tìm thấy Lot nào có lỗi \"{error_name_query}\" trong MES snapshot.",
                (error_name_query,),
            )
        answer = (
            f"Theo MES snapshot, các Lot có lỗi \"{error_name_query}\" nhiều nhất là: "
            + format_item_list(
                [
                    f"{row['lot_id']} ({row['product_id']}): "
                    f"{self._number(row['total_error_qty'])}"
                    for row in rows
                ]
            )
            + "."
        )
        return self._result(
            "lots_for_error_name",
            rows,
            answer,
            (error_name_query, str(rows[0]["lot_id"]), str(rows[0]["total_error_qty"])),
        )

    def _product_summary(self, product_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT product_id, lot_count, error_record_count, total_error_qty
            FROM v_product_error_summary
            WHERE product_id = ?
            LIMIT 1
            """,
            (product_id,),
        )
        if not rows:
            return self._result(
                "product_error_summary",
                [],
                f"Không tìm thấy mã hàng {product_id} trong MES snapshot.",
                (product_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, mã hàng {product_id} có tổng "
            f"{self._number(row['total_error_qty'])} lỗi, nằm trong "
            f"{self._number(row['lot_count'])} Lot và "
            f"{self._number(row['error_record_count'])} bản ghi lỗi."
        )
        return self._result(
            "product_error_summary",
            rows,
            answer,
            (product_id, str(row["total_error_qty"])),
        )

    def _product_lot_count(self, product_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT product_id, lot_count, error_record_count, total_error_qty
            FROM v_product_error_summary
            WHERE product_id = ?
            LIMIT 1
            """,
            (product_id,),
        )
        if not rows:
            return self._result(
                "product_lot_count",
                [],
                f"Không tìm thấy mã hàng {product_id} trong MES snapshot.",
                (product_id,),
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, mã hàng {product_id} có "
            f"{self._number(row['lot_count'])} Lot ghi nhận lỗi."
        )
        return self._result(
            "product_lot_count",
            rows,
            answer,
            (product_id, str(row["lot_count"])),
        )

    def _product_average_errors_per_lot(self, product_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT product_id, lot_count, total_error_qty
            FROM v_product_error_summary
            WHERE product_id = ?
            LIMIT 1
            """,
            (product_id,),
        )
        if not rows:
            return self._result(
                "product_average_errors_per_lot",
                [],
                f"Không tìm thấy mã hàng {product_id} trong MES snapshot.",
                (product_id,),
            )
        row = rows[0]
        lot_count = int(row["lot_count"] or 0)
        total_error_qty = int(row["total_error_qty"] or 0)
        average = total_error_qty / lot_count if lot_count else 0
        display_average = (
            f"{average:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        )
        answer = (
            f"Theo MES snapshot, mã hàng {product_id} có trung bình "
            f"{display_average} lỗi mỗi Lot "
            f"({self._number(total_error_qty)} lỗi / {self._number(lot_count)} Lot)."
        )
        return self._result(
            "product_average_errors_per_lot",
            [
                {
                    "product_id": product_id,
                    "lot_count": lot_count,
                    "total_error_qty": total_error_qty,
                    "average_error_qty_per_lot": average,
                }
            ],
            answer,
            (product_id, str(total_error_qty), str(lot_count)),
        )

    def _compare_product_errors(self, product_ids: list[str]) -> MesDatabaseResult:
        placeholders = ",".join("?" for _ in product_ids)
        rows = self._fetch_all(
            f"""
            SELECT product_id, lot_count, error_record_count, total_error_qty
            FROM v_product_error_summary
            WHERE product_id IN ({placeholders})
            ORDER BY total_error_qty DESC, product_id
            """,
            tuple(product_ids),
        )
        found = {str(row["product_id"]) for row in rows}
        missing = [product_id for product_id in product_ids if product_id not in found]
        if len(rows) < 2:
            if missing:
                answer = (
                    "Không đủ dữ liệu MES snapshot để so sánh: "
                    + ", ".join(f"không tìm thấy mã hàng {item}" for item in missing)
                    + "."
                )
            else:
                answer = "Không đủ dữ liệu MES snapshot để so sánh hai mã hàng."
            return self._result(
                "product_error_comparison",
                rows,
                answer,
                tuple(product_ids),
            )

        higher, lower = rows[0], rows[1]
        diff = int(higher["total_error_qty"] or 0) - int(lower["total_error_qty"] or 0)
        answer = (
            f"Theo MES snapshot, mã hàng {higher['product_id']} có tổng "
            f"{self._number(higher['total_error_qty'])} lỗi, cao hơn mã hàng "
            f"{lower['product_id']} có {self._number(lower['total_error_qty'])} lỗi "
            f"là {self._number(diff)} lỗi."
        )
        if missing:
            answer += " Không tìm thấy dữ liệu cho: " + ", ".join(missing) + "."
        return self._result(
            "product_error_comparison",
            rows,
            answer,
            tuple(
                product_ids
                + [str(higher["total_error_qty"]), str(lower["total_error_qty"])]
            ),
        )

    def _product_error_breakdown(self, product_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            f"""
            SELECT l.product_id, e.error_id,
                   COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
                   e.process_id, COUNT(e.error_pk) AS error_record_count,
                   SUM(e.quantity) AS total_error_qty
            FROM error_events AS e
            JOIN lots AS l ON l.lot_pk = e.lot_pk
            LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
            WHERE l.product_id = ?
              AND {self._exclude_test_filter("l.product_id", "l.lot_id")}
            GROUP BY l.product_id, e.error_id, e.process_id, e.error_catalog_pk
            ORDER BY total_error_qty DESC, e.error_id
            LIMIT 10
            """,
            (product_id,),
        )
        if not rows:
            return self._result(
                "product_error_breakdown",
                [],
                f"Không tìm thấy dữ liệu lỗi của mã hàng {product_id} trong MES snapshot.",
                (product_id,),
            )
        descriptions = format_item_list(
            [
                f"{row['error_id']} ({row['error_name'] or '*Lỗi chưa rõ tên*'}): "
                f"{self._number(row['total_error_qty'])}"
                for row in rows
            ]
        )
        answer = f"Theo MES snapshot, các lỗi chính của mã hàng {product_id} là: {descriptions}."
        return self._result(
            "product_error_breakdown",
            rows,
            answer,
            (product_id, str(rows[0]["error_id"]), str(rows[0]["total_error_qty"])),
        )

    def _process_error_types(self, process_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            f"""
            SELECT process_id, error_id, error_name,
                   SUM(quantity) AS total_error_qty,
                   COUNT(*) AS error_record_count
            FROM v_error_details
            WHERE process_id = ?
              AND {self._exclude_test_filter("product_id", "lot_id")}
            GROUP BY process_id, error_id, error_name
            ORDER BY total_error_qty DESC, error_id
            """,
            (process_id,),
        )
        if not rows:
            return self._result(
                "process_error_types",
                [],
                f"Không tìm thấy dữ liệu lỗi cho process {process_id} trong MES snapshot.",
                (process_id,),
            )
        descriptions = format_item_list(
            [
                f"{row['error_id']} - {row['error_name'] or '*Lỗi chưa rõ tên*'}: "
                f"{self._number(row['total_error_qty'])} lỗi"
                for row in rows
            ]
        )
        answer = f"Theo MES snapshot, process {process_id} ghi nhận các loại lỗi: {descriptions}."
        return self._result(
            "process_error_types",
            rows,
            answer,
            (process_id, str(rows[0]["error_id"]), str(rows[0]["total_error_qty"])),
        )

    def _highest_error_products(self, limit: int = 1) -> MesDatabaseResult:
        exclude_test_products = self._exclude_test_filter("product_id", None)
        if limit > 1:
            rows = self._fetch_all(
                f"""
                SELECT product_id, lot_count, error_record_count, total_error_qty
                FROM v_product_error_summary
                WHERE {exclude_test_products}
                  AND total_error_qty > 0
                ORDER BY total_error_qty DESC, product_id
                LIMIT ?
                """,
                (limit,),
            )
        else:
            rows = self._fetch_all(
                f"""
                SELECT product_id, lot_count, error_record_count, total_error_qty
                FROM v_product_error_summary
                WHERE {exclude_test_products}
                  AND total_error_qty = (
                    SELECT MAX(total_error_qty) FROM v_product_error_summary
                    WHERE {exclude_test_products}
                )
                ORDER BY product_id
                """
            )
        if not rows:
            return self._result("highest_error_product", [], "MES snapshot chưa có dữ liệu sản phẩm.")
        if len(rows) == 1:
            prefix = "Theo MES snapshot, mã hàng có tổng lỗi cao nhất là: "
        else:
            prefix = f"Theo MES snapshot, {len(rows)} mã hàng có tổng lỗi cao nhất là: "
        answer = prefix + format_item_list(
            [
                f"{row['product_id']} với {self._number(row['total_error_qty'])} lỗi"
                for row in rows
            ]
        ) + "."
        terms = tuple(
            str(value)
            for row in rows
            for value in (row["product_id"], row["total_error_qty"])
        )
        return self._result("highest_error_product", rows, answer, terms)

    def _ranked_error_product(self, rank: int) -> MesDatabaseResult:
        exclude_test_products = self._exclude_test_filter("product_id", None)
        rows = self._fetch_all(
            f"""
            SELECT product_id, lot_count, error_record_count, total_error_qty
            FROM v_product_error_summary
            WHERE {exclude_test_products}
              AND total_error_qty > 0
            ORDER BY total_error_qty DESC, product_id
            LIMIT 1 OFFSET ?
            """,
            (max(0, rank - 1),),
        )
        if not rows:
            return self._result(
                "ranked_error_product",
                [],
                f"MES snapshot chưa có mã hàng đứng thứ {rank} theo tổng lỗi.",
            )
        row = rows[0]
        answer = (
            f"Theo MES snapshot, mã hàng đứng thứ {rank} theo tổng lỗi là "
            f"{row['product_id']} với {self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "ranked_error_product",
            rows,
            answer,
            (str(row["product_id"]), str(row["total_error_qty"])),
        )

    def _lots_for_error(self, error_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            f"""
            SELECT e.lot_id, l.product_id, e.error_id,
                   COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
                   SUM(e.quantity) AS total_error_qty
            FROM error_events AS e
            LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
            LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
            WHERE e.error_id = ?
              AND {self._exclude_test_filter("l.product_id", "e.lot_id")}
            GROUP BY e.lot_id, l.product_id, e.error_id, e.error_catalog_pk
            ORDER BY total_error_qty DESC, e.lot_id
            LIMIT 10
            """,
            (error_id,),
        )
        if not rows:
            return self._result(
                "lots_for_error",
                [],
                f"Không tìm thấy Lot nào có mã lỗi {error_id} trong MES snapshot.",
                (error_id,),
            )
        answer = f"Theo MES snapshot, các Lot có mã lỗi {error_id} nhiều nhất là: " + format_item_list(
            [
                f"{row['lot_id']} ({row['product_id'] or 'chưa rõ mã hàng'}): "
                f"{self._number(row['total_error_qty'])}"
                for row in rows
            ]
        ) + "."
        return self._result(
            "lots_for_error",
            rows,
            answer,
            (error_id, str(rows[0]["lot_id"]), str(rows[0]["total_error_qty"])),
        )

    def _matching_error_name_rows(
        self,
        error_name_query: str,
        *,
        grouped: bool = False,
        grouped_by_lot: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_query = normalize_mes_text(error_name_query)
        if not normalized_query:
            return []
        if grouped_by_lot:
            sql = f"""
                SELECT lot_id, product_id, error_id, error_name, process_id,
                       SUM(quantity) AS total_error_qty,
                       COUNT(*) AS error_record_count
                FROM v_error_details
                WHERE {self._exclude_test_filter()}
                GROUP BY lot_id, product_id, error_id, error_name, process_id
                ORDER BY total_error_qty DESC, lot_id
            """
        elif grouped:
            sql = f"""
                SELECT error_id, error_name, process_id, error_type,
                       SUM(quantity) AS total_error_qty,
                       COUNT(*) AS error_record_count
                FROM v_error_details
                WHERE {self._exclude_test_filter()}
                GROUP BY error_id, error_name, process_id, error_type
                ORDER BY total_error_qty DESC, error_id, process_id
            """
        else:
            sql = f"""
                SELECT lot_id, product_id, error_id, error_name, process_id,
                       error_type, quantity
                FROM v_error_details
                WHERE {self._exclude_test_filter()}
                ORDER BY quantity DESC, lot_id
            """
        rows = self._fetch_all(sql)
        exact = [
            row
            for row in rows
            if normalize_mes_text(str(row.get("error_name") or "")) == normalized_query
        ]
        if exact:
            return exact[:20]
        return [
            row
            for row in rows
            if normalized_query in normalize_mes_text(str(row.get("error_name") or ""))
            or normalize_mes_text(str(row.get("error_name") or "")) in normalized_query
        ][:20]

    @classmethod
    def _extract_lot_id(cls, question: str) -> str | None:
        match = cls.LOT_PATTERN.search(question)
        return match.group(0) if match else None

    @staticmethod
    def _extract_quoted_error_name(question: str) -> str | None:
        match = re.search(r"[\"“”'「」『』]([^\"“”'「」『』]+)[\"“”'「」『』]", question or "")
        if not match:
            return None
        candidate = match.group(1).strip()
        if not candidate or re.search(r"\d{6}(?:-\d{2})?-\d{3}", candidate):
            return None
        if normalize_mes_text(candidate) in {
            "co loi",
            "khong co loi",
            "has error",
            "has errors",
            "with error",
            "with errors",
            "loi",
            "error",
            "errors",
        }:
            return None
        return candidate

    @staticmethod
    def _extract_code_after(question: str, label_pattern: str) -> str | None:
        match = re.search(
            rf"{label_pattern}\s*(?:là|:|#)?\s*([A-Za-z0-9_Đđ-]+)",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        code = match.group(1).strip()
        normalized_code = normalize_mes_text(code)
        natural_language_tokens = {
            "nao",
            "gi",
            "co",
            "la",
            "cua",
            "xuat",
            "hien",
            "nhieu",
            "cao",
            "lon",
            "trong",
            "voi",
            "the",
            "a",
            "an",
            "by",
            "for",
            "of",
            "with",
            "quantity",
            "quantities",
            "name",
            "names",
            "top",
            "highest",
            "largest",
            "greatest",
            "most",
            "production",
            "real",
            "lots",
            "lot",
        }
        if normalized_code in natural_language_tokens:
            return None
        if not re.search(r"\d|[_-]", code):
            return None
        return code

    @classmethod
    def _extract_product_candidates(cls, question: str) -> list[str]:
        candidates = re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+(?![A-Za-z0-9])",
            question or "",
        )
        products: list[str] = []
        for candidate in candidates:
            if cls.LOT_PATTERN.fullmatch(candidate):
                continue
            if normalize_mes_text(candidate) in {"error_events", "v_error_details"}:
                continue
            if candidate not in products:
                products.append(candidate)
        return products

    @staticmethod
    def _extract_process_id(question: str) -> str | None:
        match = re.search(
            r"(?<![A-Za-z0-9])\d{3}-[A-Za-z0-9]+-[A-Za-z](?![A-Za-z0-9])",
            question or "",
        )
        return match.group(0) if match else None

    @staticmethod
    def _has_error_marker(normalized: str) -> bool:
        return bool(re.search(r"\b(ng|loi|error|errors|defect|defects)\b", normalized))

    @staticmethod
    def _asks_breakdown(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in ("nhung loi", "cac loi", "loi nao", "chi tiet loi", "pho bien")
        ) or any(
            marker in normalized
            for marker in (
                "breakdown",
                "error breakdown",
                "error code",
                "error codes",
                "defect code",
                "defect codes",
                "top error",
                "top errors",
                "by error",
            )
        )

    @staticmethod
    def _asks_lots_for_error(normalized: str) -> bool:
        return bool(re.search(r"\b(lot|lots|lo)\b", normalized)) and any(
            marker in normalized
            for marker in (
                "nao",
                "danh sach",
                "top",
                "nhieu nhat",
                "list",
                "show",
                "which",
            )
        )

    @staticmethod
    def _asks_count(normalized: str) -> bool:
        return any(
            marker in normalized for marker in ("bao nhieu", "may", "tong so")
        ) or bool(re.search(r"\b(how many|total|number of)\b", normalized))

    @staticmethod
    def _asks_quantity(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in (
                "quantity",
                "so luong",
                "tong",
                "tong so",
                "bao nhieu",
                "total",
                "sum",
            )
        )

    @staticmethod
    def _asks_total_quantity(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in ("tong", "tong so", "total", "sum")
        )

    @staticmethod
    def _asks_number_only(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in (
                "chi tra loi bang 1 so",
                "mot so duy nhat",
                "1 so duy nhat",
                "only one number",
                "one number only",
                "just one number",
            )
        )

    @staticmethod
    def _asks_compare(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in ("so sanh", "compare", "comparison")
        )

    @staticmethod
    def _is_lot_process_question(normalized: str) -> bool:
        if MesDatabase._asks_process_metrics(normalized):
            return True
        has_process = any(
            marker in normalized
            for marker in (
                "cong doan",
                "process step",
                "process steps",
                "process history",
                "process progress",
                "tien do process",
                "lich su process",
                "工程",
                "プロセス",
            )
        )
        has_progress = any(
            marker in normalized
            for marker in (
                "tien do",
                "dang o",
                "hien tai",
                "moi nhat",
                "da qua",
                "lich su",
                "liet ke",
                "danh sach",
                "step count",
                "latest",
                "current",
                "history",
                "list",
            )
        ) or MesDatabase._asks_process_step_count(normalized)
        return has_process and has_progress

    @staticmethod
    def _asks_process_history(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in (
                "da qua",
                "lich su",
                "liet ke",
                "danh sach",
                "process history",
                "list process",
                "工程履歴",
                "工程一覧",
            )
        )

    @staticmethod
    def _asks_process_metrics(normalized: str) -> bool:
        metric_text = normalized.replace("_", " ").replace("-", " ")
        return any(
            marker in metric_text
            for marker in (
                "p ok",
                "p ng",
                "s ok",
                "s ng",
                "b ok",
                "b ng",
                "ok ng",
                "ng defect",
                "ng scrap",
                "output max",
                "so lieu p s b",
                "process metrics",
            )
        )

    @staticmethod
    def _asks_process_step_count(normalized: str) -> bool:
        asks_count = MesDatabase._asks_count(normalized) or bool(
            re.search(r"\b(may|step count|process step count)\b", normalized)
        )
        has_step = any(
            marker in normalized
            for marker in (
                "cong doan",
                "process step",
                "process steps",
            )
        )
        return asks_count and has_step

    @staticmethod
    def _is_lot_error_record_count_question(normalized: str) -> bool:
        return MesDatabase._asks_count(normalized) and any(
            marker in normalized
            for marker in ("ban ghi", "record", "records", "su kien")
        )

    @staticmethod
    def _is_lot_distinct_error_count_question(normalized: str) -> bool:
        return MesDatabase._asks_count(normalized) and any(
            marker in normalized
            for marker in (
                "loai loi",
                "ma loi khac nhau",
                "distinct",
                "different error",
                "different errors",
            )
        )

    @staticmethod
    def _is_count_error_records_question(normalized: str) -> bool:
        return MesDatabase._asks_count(normalized) and (
            "ban ghi" in normalized
            or any(
                marker in normalized
                for marker in ("error record", "error records", "su kien loi")
            )
        )

    @staticmethod
    def _is_count_lots_with_errors_question(normalized: str) -> bool:
        has_lot = bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        is_average = bool(re.search(r"\b(trung binh|average)\b", normalized))
        if is_average:
            return False
        return (
            MesDatabase._asks_count(normalized)
            and has_lot
            and MesDatabase._has_error_marker(normalized)
        )

    @staticmethod
    def _is_ambiguous_lot_count_question(normalized: str) -> bool:
        compact = re.sub(r"\s+", " ", normalized or "").strip(" ?.!。")
        if compact in {
            "co bao nhieu lot",
            "bao nhieu lot",
            "co may lot",
            "may lot",
            "how many lots",
        }:
            return True
        has_lot = bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        if has_lot and MesDatabase._asks_count(normalized) and any(
            marker in normalized
            for marker in (
                "khong noi ro",
                "chua ro",
                "mo ho",
                "not specify",
                "unclear",
            )
        ):
            return True
        return (
            has_lot
            and MesDatabase._asks_count(normalized)
            and not MesDatabase._has_error_marker(normalized)
        )

    @staticmethod
    def _is_lot_listing_question(normalized: str) -> bool:
        has_lot = bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        has_listing = any(
            marker in normalized
            for marker in (
                "liet ke",
                "danh sach",
                "nhung lot nao",
                "cac lot",
                "lot nao dang co",
                "lot hien co",
                "list",
                "show",
                "which lots",
                "current lots",
                "available lots",
            )
        )
        return has_lot and has_listing

    @staticmethod
    def _is_lowest_lot_error_question(normalized: str) -> bool:
        has_lot = bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        has_error = MesDatabase._has_error_marker(normalized)
        has_minimum = bool(
            re.search(r"\b(it|thap|nho)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
        ) or any(
            marker in normalized
            for marker in (
                "it loi nhat",
                "thap nhat",
                "nho nhat",
                "min",
                "minimum",
                "least",
                "lowest",
                "smallest",
            )
        )
        return has_lot and has_error and has_minimum

    @staticmethod
    def _is_highest_lot_error_question(normalized: str) -> bool:
        has_lot = bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        has_error = MesDatabase._has_error_marker(normalized)
        has_maximum = bool(
            re.search(r"\b(nhieu|cao|lon)\b(?:\s+\w+){0,3}\s+nhat\b", normalized)
        ) or any(
            marker in normalized
            for marker in (
                "nhieu nhat",
                "cao nhat",
                "lon nhat",
                "dung dau",
                "top",
                "max",
                "maximum",
                "most",
                "highest",
                "largest",
                "greatest",
            )
        )
        return has_lot and has_error and has_maximum

    @staticmethod
    def _extract_top_limit(
        normalized: str,
        default: int = 1,
        maximum: int = 50,
        *,
        unit_pattern: str = r"lot|lots|lo",
    ) -> int:
        match = re.search(
            rf"\btop\s*(\d+)(?:\s+\w+){{0,5}}\s+(?:{unit_pattern})\b",
            normalized,
        )
        if not match:
            match = re.search(rf"\b(\d+)\s+(?:{unit_pattern})\b", normalized)
        if not match:
            word_numbers = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
            }
            word_match = re.search(
                (
                    r"\btop\s+(one|two|three|four|five|six|seven|eight|nine|ten)"
                    rf"(?:\s+\w+){{0,5}}\s+(?:{unit_pattern})\b"
                ),
                normalized,
            )
            if word_match:
                return word_numbers[word_match.group(1)]
        if not match:
            return default
        return max(1, min(maximum, int(match.group(1))))

    PRODUCT_UNIT_PATTERN = r"ma hang|san pham|product|products|item|items|part|parts"

    @staticmethod
    def _is_highest_product_error_question(normalized: str) -> bool:
        has_product = any(
            marker in normalized
            for marker in (
                "san pham",
                "ma hang",
                "product code",
                "product id",
                "part number",
                "item code",
            )
        )
        has_maximum = any(
            marker in normalized
            for marker in (
                "nhieu nhat",
                "cao nhat",
                "lon nhat",
                "dung dau",
                "top 1",
                "max",
                "most",
                "highest",
                "largest",
                "greatest",
            )
        )
        return has_product and MesDatabase._has_error_marker(normalized) and has_maximum

    @staticmethod
    def _is_ranked_product_error_question(normalized: str, original_question: str = "") -> bool:
        has_product = any(
            marker in normalized
            for marker in (
                "san pham",
                "ma hang",
                "product code",
                "product id",
                "part number",
                "item code",
            )
        ) or bool(re.search(r"(品番|製品|製品コード)", original_question or ""))
        return has_product and MesDatabase._has_error_marker(normalized)

    @staticmethod
    def _extract_rank_position(
        normalized: str,
        original_question: str = "",
        maximum: int = 50,
    ) -> int | None:
        rank_words = {
            "mot": 1,
            "nhat": 1,
            "hai": 2,
            "ba": 3,
            "bon": 4,
            "tu": 4,
            "nam": 5,
            "sau": 6,
            "bay": 7,
            "tam": 8,
            "chin": 9,
            "muoi": 10,
            "one": 1,
            "first": 1,
            "two": 2,
            "second": 2,
            "three": 3,
            "third": 3,
            "four": 4,
            "fourth": 4,
            "five": 5,
            "fifth": 5,
        }
        match = re.search(
            r"\b(?:dung\s+)?thu\s+(\d+)\b|\b(?:rank|ranking|position)\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)\b",
            normalized,
        )
        if match:
            for group in match.groups():
                if group:
                    return max(1, min(maximum, int(group)))
        word_match = re.search(
            r"\b(?:dung\s+)?thu\s+(mot|nhat|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi)\b|\b(second|third|fourth|fifth|first)\b",
            normalized,
        )
        if word_match:
            for group in word_match.groups():
                if group:
                    return rank_words.get(group)
        jp_match = re.search(r"(\d+)\s*番目|第\s*(\d+)", original_question or "")
        if jp_match:
            for group in jp_match.groups():
                if group:
                    return max(1, min(maximum, int(group)))
        if re.search(r"(二番目|2番目|第2)", original_question or ""):
            return 2
        return None

    @staticmethod
    def _asks_product_lot_count(normalized: str) -> bool:
        return (
            MesDatabase._asks_count(normalized)
            and bool(re.search(r"\b(lot|lots|lo)\b", normalized))
        )

    @staticmethod
    def _asks_average_per_lot(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in (
                "trung binh",
                "binh quan",
                "moi lot",
                "per lot",
                "average",
                "avg",
            )
        )

    @staticmethod
    def _unsupported_scope_reason(normalized: str) -> str:
        if any(
            marker in normalized
            for marker in (
                "cong nhan",
                "nguoi san xuat",
                "nguoi van hanh",
                "operator",
                "worker",
            )
        ):
            return (
                "MES snapshot không cung cấp danh tính công nhân/người vận hành "
                "cho chức năng hỏi đáp vì đây là dữ liệu nhân sự nhạy cảm."
            )
        if any(
            marker in normalized
            for marker in ("chi phi", "gia tien", "bao nhieu tien", "cost", "expense")
        ):
            return "MES snapshot hiện không có cột chi phí sửa lỗi, nên không thể tính chi phí."
        if any(
            marker in normalized
            for marker in (
                "du doan",
                "du bao",
                "thang sau",
                "tuong lai",
                "forecast",
                "predict",
                "prediction",
                "next month",
            )
        ):
            return (
                "MES snapshot là dữ liệu đã ghi nhận, không phải mô hình dự báo; "
                "không thể dự đoán lỗi tương lai từ dữ liệu này."
            )
        if any(marker in normalized for marker in ("khach hang", "customer", "client")):
            return "MES snapshot hiện không có thông tin khách hàng, nên không thể xác định khách hàng của sản phẩm."
        if any(
            marker in normalized
            for marker in (
                "chua tung bi loi",
                "khong bi loi",
                "khong co loi",
                "never had error",
                "never defective",
                "without error",
                "no error",
            )
        ):
            return (
                "MES snapshot hiện là view dữ liệu lỗi; không đủ cơ sở để liệt kê "
                "sản phẩm chưa từng bị lỗi."
            )
        return ""

    def _unsupported_scope(self, reason: str) -> MesDatabaseResult:
        return self._result(
            "unsupported_mes_scope",
            [{"reason": reason}],
            reason,
            (),
        )

    @staticmethod
    def _format_ranked_lots(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "MES snapshot chưa có dữ liệu lỗi theo Lot."
        descriptions = format_item_list(
            [
                f"Lot {row['lot_id']}, mã hàng {row['product_id']}, "
                f"{MesDatabase._number(row['total_error_qty'])} lỗi"
                for row in rows
            ]
        )
        return f"Theo MES snapshot, Lot có tổng lỗi cao nhất là: {descriptions}."

    @staticmethod
    def _format_ranked_lots_with_errors(rows: list[dict[str, Any]]) -> str:
        """Trả về mô tả đầy đủ: tổng lỗi + top 3 lỗi chi tiết của từng Lot."""
        if not rows:
            return "MES snapshot chưa có dữ liệu lỗi theo Lot."
        parts: list[str] = []
        for row in rows:
            base = (
                f"Lot {row['lot_id']}, mã hàng {row['product_id']}, "
                f"có tổng {MesDatabase._number(row['total_error_qty'])} lỗi"
            )
            top_errors: list[dict[str, Any]] = row.get("top_errors") or []
            if top_errors:
                error_lines = format_item_list(
                    [
                        f"{e['error_id']} - {e['error_name'] or '*Lỗi chưa rõ tên*'}: "
                        f"{MesDatabase._number(e['error_qty'])}"
                        for e in top_errors
                    ]
                )
                base += f". Trong đó các lỗi có số lượng lớn nhất: {error_lines}"
            parts.append(base)
        if len(rows) == 1:
            prefix = "Theo MES snapshot, Lot có tổng lỗi cao nhất là: "
        else:
            prefix = f"Theo MES snapshot, {len(rows)} Lot có tổng lỗi cao nhất là: "
        return prefix + format_item_list(parts) + "."

    @staticmethod
    def _number(value: Any) -> str:
        if value is None:
            return "chưa rõ"
        return f"{int(value):,}".replace(",", ".")
