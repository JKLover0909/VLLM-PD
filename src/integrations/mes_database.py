"""Read-only query service for the local MES SQLite snapshot."""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
            "filters": {"exclude_test_data": False},
            "intent": self.intent,
            "rows": self.rows,
        }


def normalize_mes_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", (value or "").lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z0-9_-]+", " ", normalized).strip()


class MesDatabase:
    """Allowlisted, parameterized queries against the MES snapshot."""

    LOT_PATTERN = re.compile(r"\b\d{6}(?:-\d{2})?-\d{3}(?:-\d{2})?\b")
    SNAPSHOT_MARKERS = (
        "snapshot",
        "database",
        "co so du lieu",
        "du lieu cuc bo",
        "du lieu da luu",
    )
    # Điều kiện loại Lot/sản phẩm test ra khỏi các truy vấn thống kê mặc định.
    # Lot test có product_id bắt đầu bằng "Test_" (không phân biệt hoa thường).
    EXCLUDE_TEST_FILTER = "LOWER(product_id) NOT LIKE 'test\\_%' ESCAPE '\\'"

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

    def status(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "db_path": str(self.db_path)}
        try:
            with self._connect() as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM schema_metadata")
                )
            return {
                "available": True,
                "db_path": str(self.db_path),
                "imported_at": metadata.get("imported_at", ""),
                "lots": int(metadata.get("lot_count", 0)),
                "error_events": int(metadata.get("error_event_count", 0)),
                "error_catalog": int(metadata.get("error_catalog_count", 0)),
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
        lot_id = self._extract_lot_id(question)
        product_id = self._extract_code_after(
            question,
            r"(?:mã\s+hàng|mã\s+sản\s+phẩm|sản\s+phẩm|product)",
        )
        error_id = self._extract_code_after(
            question,
            r"(?:mã\s+lỗi|lỗi\s+mã|error\s+code)",
        )
        has_error = self._has_error_marker(normalized)

        if lot_id and has_error:
            return self._lot_error_breakdown(lot_id)
        if lot_id:
            return self._lot_details(lot_id)
        if error_id and self._asks_lots_for_error(normalized):
            return self._lots_for_error(error_id)
        if error_id:
            return self._error_name(error_id)
        if self._is_highest_product_error_question(normalized):
            return self._highest_error_products()
        if product_id and has_error:
            if self._asks_breakdown(normalized):
                return self._product_error_breakdown(product_id)
            return self._product_summary(product_id)
        if allow_highest_lot and self._is_highest_lot_error_question(normalized):
            return self._highest_error_lots()
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

    def _highest_error_lots(self) -> MesDatabaseResult:
        rows = self._fetch_all(
            f"""
            SELECT lot_id, product_id, total_error_qty, error_record_count,
                   distinct_error_count, unmapped_error_record_count
            FROM v_lot_error_summary
            WHERE {self.EXCLUDE_TEST_FILTER}
              AND total_error_qty = (
                SELECT MAX(total_error_qty) FROM v_lot_error_summary
                WHERE {self.EXCLUDE_TEST_FILTER}
            )
            ORDER BY lot_id
            """
        )
        # Lấy thêm top lỗi chi tiết của mỗi Lot để model có thể trình bày phong phú hơn
        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            top_errors = self._fetch_all(
                """
                SELECT error_id, error_name, total_error_qty AS error_qty
                FROM v_lot_error_breakdown
                WHERE lot_id = ?
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
            f"{row['product_id']}, trạng thái {row['status'] or 'chưa rõ'}, "
            f"có {self._number(row['pcs_lot'])} PCS và tổng "
            f"{self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "lot_details",
            rows,
            answer,
            (str(row["lot_id"]), str(row["product_id"]), str(row["total_error_qty"])),
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
        descriptions = "; ".join(
            f"{row['error_id']} ({row['error_name'] or '*Lỗi chưa rõ tên*'}): "
            f"{self._number(row['total_error_qty'])}"
            for row in rows
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
            f"{self._number(row['total_error_qty'])} lỗi."
        )
        return self._result(
            "product_error_summary",
            rows,
            answer,
            (product_id, str(row["total_error_qty"])),
        )

    def _product_error_breakdown(self, product_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT l.product_id, e.error_id,
                   COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
                   e.process_id, COUNT(e.error_pk) AS error_record_count,
                   SUM(e.quantity) AS total_error_qty
            FROM error_events AS e
            JOIN lots AS l ON l.lot_pk = e.lot_pk
            LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
            WHERE l.product_id = ?
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
        descriptions = "; ".join(
            f"{row['error_id']} ({row['error_name'] or '*Lỗi chưa rõ tên*'}): "
            f"{self._number(row['total_error_qty'])}"
            for row in rows
        )
        answer = f"Theo MES snapshot, các lỗi chính của mã hàng {product_id} là: {descriptions}."
        return self._result(
            "product_error_breakdown",
            rows,
            answer,
            (product_id, str(rows[0]["error_id"]), str(rows[0]["total_error_qty"])),
        )

    def _highest_error_products(self) -> MesDatabaseResult:
        rows = self._fetch_all(
            f"""
            SELECT product_id, lot_count, error_record_count, total_error_qty
            FROM v_product_error_summary
            WHERE {self.EXCLUDE_TEST_FILTER}
              AND total_error_qty = (
                SELECT MAX(total_error_qty) FROM v_product_error_summary
                WHERE {self.EXCLUDE_TEST_FILTER}
            )
            ORDER BY product_id
            """
        )
        if not rows:
            return self._result("highest_error_product", [], "MES snapshot chưa có dữ liệu sản phẩm.")
        answer = "Theo MES snapshot, mã hàng có tổng lỗi cao nhất là: " + "; ".join(
            f"{row['product_id']} với {self._number(row['total_error_qty'])} lỗi"
            for row in rows
        ) + "."
        terms = tuple(
            str(value)
            for row in rows
            for value in (row["product_id"], row["total_error_qty"])
        )
        return self._result("highest_error_product", rows, answer, terms)

    def _lots_for_error(self, error_id: str) -> MesDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT e.lot_id, l.product_id, e.error_id,
                   COALESCE(c.error_name_vi, c.error_name, c.error_name_en) AS error_name,
                   SUM(e.quantity) AS total_error_qty
            FROM error_events AS e
            LEFT JOIN lots AS l ON l.lot_pk = e.lot_pk
            LEFT JOIN error_catalog AS c ON c.error_catalog_pk = e.error_catalog_pk
            WHERE e.error_id = ?
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
        answer = f"Theo MES snapshot, các Lot có mã lỗi {error_id} nhiều nhất là: " + "; ".join(
            f"{row['lot_id']} ({row['product_id'] or 'chưa rõ mã hàng'}): "
            f"{self._number(row['total_error_qty'])}"
            for row in rows
        ) + "."
        return self._result(
            "lots_for_error",
            rows,
            answer,
            (error_id, str(rows[0]["lot_id"]), str(rows[0]["total_error_qty"])),
        )

    @classmethod
    def _extract_lot_id(cls, question: str) -> str | None:
        match = cls.LOT_PATTERN.search(question)
        return match.group(0) if match else None

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
        if normalize_mes_text(code) in {"nao", "gi", "co", "la", "cua"}:
            return None
        return code

    @staticmethod
    def _has_error_marker(normalized: str) -> bool:
        return bool(re.search(r"\b(ng|loi|error|defect)\b", normalized))

    @staticmethod
    def _asks_breakdown(normalized: str) -> bool:
        return any(
            marker in normalized
            for marker in ("nhung loi", "cac loi", "loi nao", "chi tiet loi", "pho bien")
        )

    @staticmethod
    def _asks_lots_for_error(normalized: str) -> bool:
        return bool(re.search(r"\b(lot|lo)\b", normalized)) and any(
            marker in normalized for marker in ("nao", "danh sach", "top", "nhieu nhat")
        )

    @staticmethod
    def _is_highest_lot_error_question(normalized: str) -> bool:
        has_lot = bool(re.search(r"\b(lot|lo)\b", normalized))
        has_error = MesDatabase._has_error_marker(normalized)
        has_maximum = any(
            marker in normalized
            for marker in ("nhieu nhat", "cao nhat", "lon nhat", "dung dau", "top 1", "max", "most")
        )
        return has_lot and has_error and has_maximum

    @staticmethod
    def _is_highest_product_error_question(normalized: str) -> bool:
        has_product = any(marker in normalized for marker in ("san pham", "ma hang", "product"))
        has_maximum = any(
            marker in normalized
            for marker in ("nhieu nhat", "cao nhat", "lon nhat", "dung dau", "top 1", "max", "most")
        )
        return has_product and MesDatabase._has_error_marker(normalized) and has_maximum

    @staticmethod
    def _format_ranked_lots(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "MES snapshot chưa có dữ liệu lỗi theo Lot."
        descriptions = "; ".join(
            f"Lot {row['lot_id']}, mã hàng {row['product_id']}, "
            f"{MesDatabase._number(row['total_error_qty'])} lỗi"
            for row in rows
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
                error_lines = "; ".join(
                    f"{e['error_id']} - {e['error_name'] or '*Lỗi chưa rõ tên*'}: "
                    f"{MesDatabase._number(e['error_qty'])}"
                    for e in top_errors
                )
                base += f". Trong đó các lỗi có số lượng lớn nhất: {error_lines}"
            parts.append(base)
        return "Theo MES snapshot, Lot có tổng lỗi cao nhất là: " + "; ".join(parts) + "."

    @staticmethod
    def _number(value: Any) -> str:
        if value is None:
            return "chưa rõ"
        return f"{int(value):,}".replace(",", ".")
