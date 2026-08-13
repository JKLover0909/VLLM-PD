"""Deterministic read-only queries for the MKHC process-warehouse snapshot."""

from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.integrations.mes_wms_contract import (
    DATASET_AUDIT,
    DATASET_CURRENT,
    DATASET_LEGACY,
    DATA_CONTRACT_VERSION,
    KPI_SUPPRESSION_REASONS,
    REASON_COMPLETED_MOVEMENTS_UNAVAILABLE,
    REASON_CROSS_ERA_UNCOMPARABLE,
    REASON_CURRENT_LOT_UNAVAILABLE,
    REASON_DATASET_NOT_OBSERVED,
    REASON_SNAPSHOT_INCOMPATIBLE,
    REASON_SNAPSHOT_UNAVAILABLE,
    REASON_SOURCE_AS_OF_UNCONFIRMED,
    REASON_SQL_AGENT_UNVERIFIED,
    REASON_TRANSACTION_SEMANTICS_UNAVAILABLE,
    SCHEMA_VERSION,
    SEMANTIC_CONTRACT_VERSION,
    validate_snapshot_connection,
)


class MesWmsDatabaseError(RuntimeError):
    """Raised when the WMS snapshot cannot be queried safely."""


@dataclass(frozen=True)
class MesWmsDatabaseResult:
    intent: str
    rows: list[dict[str, Any]]
    imported_at: str
    source_as_of: str
    fallback_answer: str
    required_terms: tuple[str, ...] = ()
    status: str = "AVAILABLE"
    reason_codes: tuple[str, ...] = ()
    domain: str = ""
    schema_version: str = ""
    data_contract_version: str = ""
    semantic_contract_version: str = ""
    semantic_epoch: str = ""
    source_as_of_state: str = ""
    source_as_of_basis: str = ""
    source_timezone: str = "unverified"
    dataset_evidence: tuple[dict[str, Any], ...] = ()
    grain: str = ""
    pagination: dict[str, Any] | None = None

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.schema_version or SCHEMA_VERSION,
            "data_contract_version": (
                self.data_contract_version or DATA_CONTRACT_VERSION
            ),
            "semantic_contract_version": (
                self.semantic_contract_version or SEMANTIC_CONTRACT_VERSION
            ),
            "intent": self.intent,
            "domain": self.domain,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "imported_at": self.imported_at,
            "source_as_of": self.source_as_of,
            "source_as_of_state": self.source_as_of_state,
            "source_as_of_basis": self.source_as_of_basis,
            "source_timezone": self.source_timezone,
            "semantic_epoch": self.semantic_epoch,
            "dataset_evidence": list(self.dataset_evidence),
            "grain": self.grain,
            "pagination": self.pagination,
        }

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "source": "mes_wms_snapshot",
            **self.metadata_payload(),
            "rows": self.rows,
        }


def normalize_wms_text(value: str) -> str:
    original = value or ""
    normalized = unicodedata.normalize("NFD", original.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"[^a-z0-9_-]+", " ", normalized).strip()

    japanese_tokens: list[str] = []
    if re.search(r"(在庫|倉庫|保管)", original):
        japanese_tokens.append("ton kho wms")
    if re.search(r"(資材|品目|品番|材料コード|資材コード)", original):
        japanese_tokens.append("vat tu")
    if re.search(r"(工程|プロセス)", original):
        japanese_tokens.append("cong doan process")
    if re.search(r"(品目|品番|材料コード|資材コード)", original):
        japanese_tokens.append("ma vat tu item code")
    if re.search(r"(総合|概要|全体|状況|レポート|報告書)", original):
        japanese_tokens.append("tong quan tinh trang bao cao")
    if re.search(r"(数量|いくつ|どのくらい)", original):
        japanese_tokens.append("so luong bao nhieu")
    if re.search(r"(仕掛品|製造中|生産中|WIP)", original, flags=re.IGNORECASE):
        japanese_tokens.append("wip dang san xuat")
    if japanese_tokens:
        normalized = " ".join((normalized, *japanese_tokens)).strip()
    return normalized


class MesWmsDatabase:
    """Allowlisted queries over a dedicated local WMS reporting snapshot."""

    ITEM_LABEL = (
        r"(?:mã\s+(?:vật\s+tư|nguyên\s+vật\s+liệu)|"
        r"vật\s+tư|nguyên\s+vật\s+liệu|item\s+code|material\s+(?:id|code)|"
        r"品目|品番|材料コード|資材コード|資材(?!ロット))"
    )
    PROCESS_LABEL = r"(?:công\s+đoạn|kho\s+công\s+đoạn|process|工程|プロセス)"
    ITEM_LOT_LABEL = (
        r"(?:lot\s+vật\s+tư|material\s+lot|item\s+lot(?:\s+id)?|"
        r"ロット(?:ID)?|資材ロット)"
    )
    # Separator between a label and its code. Vietnamese/English use spaces,
    # "là", ":" or "#"; Japanese glues the code to the label or joins it with a
    # full-width colon or a topic/subject particle. Keeping both families in one
    # optional group avoids per-language extraction paths.
    LABEL_CODE_CONNECTOR = (
        r"(?:\s|　)*(?:là|は|が|の|:|：|=|#)?(?:\s|　)*"
    )
    NON_CODE_TOKENS = frozenset(
        {
            "nao",
            "gi",
            "bao nhieu",
            "co",
            "cua",
            "hien tai",
            "ton",
            "ton kho",
        }
    )
    # Units of measure seen in MKHC material master and everyday phrasing.
    # Used only to reject "<number><unit>" captures, never to convert values.
    MEASUREMENT_UNITS = frozenset(
        {
            "kg",
            "g",
            "mg",
            "tan",
            "t",
            "m",
            "mm",
            "cm",
            "km",
            "m2",
            "m3",
            "l",
            "ml",
            "pcs",
            "pc",
            "piece",
            "set",
            "box",
            "hop",
            "thung",
            "cuon",
            "cai",
            "chiec",
            "tui",
            "goi",
            "bo",
            "tam",
            "to",
            "ngay",
            "gio",
            "phut",
            "thang",
            "nam",
            "tuan",
            "個",
            "本",
            "枚",
            "箱",
            "袋",
            "台",
        }
    )
    TRANSACTION_LABEL = (
        r"(?:mã\s+giao\s+dịch|transaction(?:\s+id)?|trans(?:\s+id)?|"
        r"取引ID|トランザクションID)"
    )
    SNAPSHOT_MARKERS = (
        "snapshot",
        "lich su snapshot",
        "ban ghi snapshot",
        "snapshot history",
        "legacy archive",
        "archive history",
        "archive",
        "luu tru lich su",
        "アーカイブ",
        "履歴",
    )
    AUDIT_MARKERS = (
        "audit",
        "giao dich",
        "transaction",
        "lich su giao dich",
    )
    PRESENCE_MARKERS = (
        "con trong current",
        "co trong current",
        "hien dien",
        "presence",
        "still present",
    )
    WMS_MARKERS = (
        "ton kho",
        "kho cong doan",
        "wms",
        "kho vat tu",
        "nguyen vat lieu",
        "warehouse stock",
        "process warehouse",
        "inventory",
    )
    WIP_MARKERS = (
        "wip",
        "hang do dang",
        "dang gia cong",
        "dang san xuat",
        "tren chuyen",
        "production wip",
    )
    PROCESS_CATALOG_MARKERS = (
        "danh sach cong doan",
        "danh sach ma cong doan",
        "danh sach macongdoan",
        "liet ke cong doan",
        "liet ke ma cong doan",
        "liet ke cac cong doan",
        "liet ke cac ma cong doan",
        "liet ke tat ca cong doan",
        "liet ke tat ca ma cong doan",
        "liet ke tat ca cac cong doan",
        "liet ke tat ca cac ma cong doan",
        "co nhung cong doan nao",
        "gom cac cong doan nao",
        "danh sach cac cong doan",
        "danh sach cac ma cong doan",
        "process list",
        "list processes",
    )
    PROCESS_CATALOG_MARKERS_JA = (
        "工程一覧",
        "工程リスト",
        "どのような工程",
        "どんな工程",
    )
    EXECUTIVE_MARKERS = (
        "tinh trang",
        "tong quan",
        "tong hop",
        "bao cao",
        "toan nha may",
        "overview",
        "summary",
    )
    SNAPSHOT_COUNT_MARKERS = (
        "bao nhieu ma vat tu",
        "bao nhieu ma cong doan",
        "so ma vat tu",
        "so ma cong doan",
        "how many material codes",
        "how many process codes",
    )
    SNAPSHOT_COUNT_MARKERS_JA = (
        "いくつの資材コード",
        "資材コードはいくつ",
        "いくつの工程コード",
        "工程コードはいくつ",
    )
    AGGREGATE_REQUEST_MARKERS = (
        "tong luong ton",
        "tong so luong ton",
        "tong ton kho",
        "total inventory",
        "total stock",
        "tong cong",
    )
    RANKING_REQUEST_MARKERS = (
        "nhieu nhat",
        "it nhat",
        "top",
        "xep hang",
        "cao nhat",
        "thap nhat",
        "most inventory",
        "least inventory",
        "ranking",
    )
    AGGREGATE_REQUEST_MARKERS_JA = (
        "在庫合計",
        "総在庫",
        "総在庫数",
        "合計在庫",
    )
    RANKING_REQUEST_MARKERS_JA = (
        "在庫が最も多",
        "在庫が最も少な",
        "ランキング",
        "上位",
    )
    UNSUPPORTED_KPI_MARKERS = {
        "MIN_STOCK": (
            "ton toi thieu",
            "duoi min",
            "safety stock",
            "minimum stock",
        ),
        "EXPIRY": ("han su dung", "het han", "expiry", "expired"),
        "WINDOW_TIME": (
            "window time",
            "thoi gian luu kho",
            "holding time",
        ),
        "TREND": (
            "xu huong",
            "dien bien",
            "bien dong",
            "tang hay giam",
            "tang",
            "giam",
            "thay doi",
            "chenh lech",
            "so sanh snapshot",
            "delta",
            "net movement",
            "trend",
            "increase",
            "decrease",
            "change",
            "difference",
            "snapshot comparison",
        ),
        "PRODUCTION_WIP": WIP_MARKERS,
        "BOTTLENECK": ("bottleneck", "nut that", "nghen cong doan"),
    }
    UNSUPPORTED_KPI_MARKERS_JA = {
        "MIN_STOCK": ("最低在庫", "安全在庫"),
        "EXPIRY": ("使用期限", "消費期限", "有効期限", "期限切れ"),
        "WINDOW_TIME": ("ウィンドウタイム", "保管時間", "滞留時間"),
        "TREND": (
            "在庫推移",
            "在庫傾向",
            "在庫変動",
            "増加",
            "減少",
            "増減",
            "差分",
            "デルタ",
            "スナップショット比較",
            "比較",
            "入出庫",
        ),
        "PRODUCTION_WIP": ("仕掛品", "製造中", "生産中", "WIP"),
        "BOTTLENECK": ("ボトルネック",),
    }
    COMPLETED_MOVEMENT_MARKERS = (
        "completed movement",
        "completed movements",
        "giao dich hoan thanh",
        "nhap xuat hoan thanh",
        "movement completed",
    )
    COMPLETED_MOVEMENT_MARKERS_JA = (
        "完了済み移動",
        "完了済み入出庫",
        "完了取引",
    )
    SUPPRESSION_CODES = tuple(KPI_SUPPRESSION_REASONS.values())
    DEFAULT_LIST_LIMIT = 20

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    @classmethod
    def from_env(cls) -> "MesWmsDatabase | None":
        enabled = os.getenv("MES_WMS_DATABASE_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        return cls(os.getenv("MES_WMS_DATABASE_PATH", "data/mes_wms.sqlite"))

    @property
    def available(self) -> bool:
        return self.db_path.is_file()

    def compatibility(self) -> dict[str, Any]:
        if not self.available:
            return {
                "compatible": False,
                "schema_version": "",
                "reason_codes": (REASON_SNAPSHOT_UNAVAILABLE,),
            }
        try:
            with self._connect() as connection:
                return validate_snapshot_connection(connection)
        except MesWmsDatabaseError:
            return {
                "compatible": False,
                "schema_version": "",
                "reason_codes": (REASON_SNAPSHOT_INCOMPATIBLE,),
            }

    def snapshot_version(self) -> str:
        if not self.available or not self.compatibility()["compatible"]:
            return ""
        metadata = self._metadata()
        current = self._dataset_evidence(DATASET_CURRENT)
        return "|".join(
            (
                current.get("source_as_of", ""),
                metadata.get("imported_at", ""),
                metadata.get("schema_version", ""),
                metadata.get("data_contract_version", ""),
            )
        )

    def status(self) -> dict[str, Any]:
        if not self.available:
            return {
                "enabled": True,
                "state": "UNAVAILABLE",
                "available": False,
                "compatible": False,
                "reason_codes": [REASON_SNAPSHOT_UNAVAILABLE],
            }
        compatibility = self.compatibility()
        if not compatibility["compatible"]:
            return {
                "enabled": True,
                "state": "INCOMPATIBLE",
                "available": False,
                "compatible": False,
                "schema_version": compatibility.get("schema_version", ""),
                "reason_codes": list(compatibility["reason_codes"]),
            }
        try:
            metadata = self._metadata()
            current = self._fetch_one("SELECT * FROM v_wms_current_quality") or {}
            evidence = self._fetch_all(
                "SELECT * FROM wms_dataset_evidence ORDER BY dataset"
            )
            current_evidence = next(
                (row for row in evidence if row.get("dataset") == DATASET_CURRENT),
                {},
            )
            capabilities = self._fetch_all(
                """
                SELECT capability, status, reason_code
                FROM wms_capability_status
                ORDER BY capability
                """
            )
            total = int(current.get("current_row_count") or 0)
            valid = int(current.get("valid_quantity_row_count") or 0)
            mapped = int(current.get("mapped_process_row_count") or 0)
            return {
                "enabled": True,
                "state": "READY",
                "available": True,
                "compatible": True,
                "schema_version": metadata.get("schema_version", ""),
                "plant_id": metadata.get("plant_id", ""),
                "source_schema": metadata.get("source_schema", ""),
                "data_contract_version": metadata.get(
                    "data_contract_version", ""
                ),
                "semantic_contract_version": metadata.get(
                    "semantic_contract_version", ""
                ),
                "imported_at": metadata.get("imported_at", ""),
                "source_as_of": current_evidence.get("source_as_of", "unconfirmed"),
                "source_as_of_state": current_evidence.get(
                    "source_as_of_state", "DERIVED_UNVERIFIED"
                ),
                "source_as_of_basis": current_evidence.get(
                    "source_as_of_basis", ""
                ),
                "source_timezone": current_evidence.get(
                    "source_timezone", "unverified"
                ),
                "datasets": evidence,
                "current_rows": total,
                "valid_quantity_rows": valid,
                "invalid_quantity_rows": max(total - valid, 0),
                "quantity_coverage": round(valid / total, 4) if total else 0.0,
                "mapped_process_rows": mapped,
                "process_mapping_coverage": round(mapped / total, 4) if total else 0.0,
                "distinct_items": int(current.get("distinct_item_count") or 0),
                "distinct_process_codes": int(
                    current.get("distinct_process_code_count") or 0
                ),
                "capabilities": capabilities,
                "suppressed_kpis": [
                    row["reason_code"]
                    for row in capabilities
                    if row["status"] == "SUPPRESSED"
                ],
            }
        except (sqlite3.Error, ValueError) as exc:
            return {
                "enabled": True,
                "state": "QUERY_ERROR",
                "available": False,
                "compatible": True,
                "error": str(exc),
            }

    @classmethod
    def is_wms_question(cls, question: str) -> bool:
        normalized = normalize_wms_text(question)
        return any(marker in normalized for marker in cls.WMS_MARKERS)

    @classmethod
    def is_wip_ambiguous_question(cls, question: str) -> bool:
        normalized = normalize_wms_text(question)
        has_wms = any(marker in normalized for marker in cls.WMS_MARKERS)
        has_wip = any(marker in normalized for marker in cls.WIP_MARKERS)
        return has_wms and has_wip

    @classmethod
    def is_cross_item_aggregate_question(cls, question: str) -> bool:
        original = question or ""
        normalized = normalize_wms_text(original)
        return any(
            marker in normalized
            for marker in cls.AGGREGATE_REQUEST_MARKERS
            + cls.RANKING_REQUEST_MARKERS
        ) or any(
            marker in original
            for marker in cls.AGGREGATE_REQUEST_MARKERS_JA
            + cls.RANKING_REQUEST_MARKERS_JA
        )

    @classmethod
    def is_snapshot_count_question(cls, question: str) -> bool:
        original = question or ""
        normalized = normalize_wms_text(original)
        return any(
            marker in normalized for marker in cls.SNAPSHOT_COUNT_MARKERS
        ) or any(
            marker in original for marker in cls.SNAPSHOT_COUNT_MARKERS_JA
        )

    @classmethod
    def unsupported_kpi(cls, question: str) -> tuple[str, str] | None:
        original = question or ""
        normalized = normalize_wms_text(original)
        has_wms_context = cls.is_wms_question(original)
        for capability, markers in cls.UNSUPPORTED_KPI_MARKERS.items():
            if has_wms_context and any(marker in normalized for marker in markers):
                return capability, KPI_SUPPRESSION_REASONS[capability]
            japanese_markers = cls.UNSUPPORTED_KPI_MARKERS_JA[capability]
            if any(marker in original for marker in japanese_markers):
                return capability, KPI_SUPPRESSION_REASONS[capability]
        return None

    def query_question(
        self,
        question: str,
        *,
        language: str = "vi",
        assume_wms: bool = False,
    ) -> MesWmsDatabaseResult | None:
        if not assume_wms and not self.is_wms_question(question):
            return None
        if not self.available:
            return self._unavailable_result(language)
        compatibility = self.compatibility()
        if not compatibility["compatible"]:
            return self._incompatible_result(language)
        if self.is_wip_ambiguous_question(question):
            return self._ambiguity_result(language)
        unsupported = self.unsupported_kpi(question)
        if unsupported is not None:
            capability, reason_code = unsupported
            return self._unsupported_kpi_result(
                capability, reason_code, language
            )

        item_code = self._extract_code_after(question, self.ITEM_LABEL)
        process_id = self._extract_code_after(question, self.PROCESS_LABEL)
        item_lot_id = self._extract_code_after(
            question,
            self.ITEM_LOT_LABEL,
            stop_label_pattern=self.PROCESS_LABEL,
        )
        trans_id = self._extract_code_after(question, self.TRANSACTION_LABEL)
        normalized = normalize_wms_text(question)
        original = question or ""
        has_completed_movement = any(
            marker in normalized for marker in self.COMPLETED_MOVEMENT_MARKERS
        ) or any(marker in original for marker in self.COMPLETED_MOVEMENT_MARKERS_JA)
        has_snapshot = any(
            marker in normalized for marker in self.SNAPSHOT_MARKERS
        ) or any(
            marker in original for marker in ("スナップショット", "アーカイブ", "履歴")
        )
        has_audit = any(
            marker in normalized for marker in self.AUDIT_MARKERS
        ) or any(marker in original for marker in ("監査", "取引", "トランザクション"))
        has_presence = any(
            marker in normalized for marker in self.PRESENCE_MARKERS
        ) or any(marker in original for marker in ("現在も存在", "現行に存在"))

        if has_completed_movement:
            return self._completed_movements_suppressed(language)
        if self.is_cross_item_aggregate_question(question) and not item_code:
            return self._cross_item_aggregate_suppression(language)

        if trans_id and has_audit:
            return self._transaction_audit_by_id(trans_id, language)
        if has_snapshot or has_audit or item_lot_id:
            if not (item_code and item_lot_id and process_id):
                return self._exact_key_required(language)
            if has_audit:
                return self._transaction_audit_by_key(
                    item_code, item_lot_id, process_id, language
                )
            if has_presence:
                return self._snapshot_presence(
                    item_code, item_lot_id, process_id, language
                )
            if has_snapshot:
                return self._snapshot_history(
                    item_code, item_lot_id, process_id, language, question
                )
            return self._current_lot_suppressed(
                item_code, item_lot_id, process_id, language
            )

        if any(
            marker in normalized for marker in self.PROCESS_CATALOG_MARKERS
        ) or any(marker in original for marker in self.PROCESS_CATALOG_MARKERS_JA):
            return self._process_catalog(language)
        if self.is_snapshot_count_question(question):
            return self._executive_overview(language)

        if item_code and process_id:
            return self._item_at_process(item_code, process_id, language)
        if item_code:
            return self._item_by_process(item_code, language)
        if process_id:
            return self._process_inventory(process_id, language)
        if any(marker in normalized for marker in self.EXECUTIVE_MARKERS):
            return self._executive_overview(language)
        return self._clarify_scope(language)

    def _connect(self) -> sqlite3.Connection:
        try:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=3.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error as exc:
            raise MesWmsDatabaseError(
                "Không thể mở WMS snapshot ở chế độ chỉ đọc."
            ) from exc

    def _fetch_all(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                return [dict(row) for row in connection.execute(sql, parameters)]
        except sqlite3.Error as exc:
            raise MesWmsDatabaseError("Không thể truy vấn WMS snapshot.") from exc

    def _fetch_one(
        self, sql: str, parameters: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        rows = self._fetch_all(sql, parameters)
        return rows[0] if rows else None

    def _metadata(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self._fetch_all("SELECT key, value FROM schema_metadata")
        }

    def _dataset_evidence(self, dataset: str) -> dict[str, Any]:
        return self._fetch_one(
            "SELECT * FROM wms_dataset_evidence WHERE dataset = ?",
            (dataset,),
        ) or {}

    def _capability(self, capability: str) -> dict[str, Any]:
        return self._fetch_one(
            """
            SELECT capability, status, reason_code, evidence_basis,
                   contract_version
            FROM wms_capability_status
            WHERE capability = ?
            """,
            (capability,),
        ) or {}

    def _dataset_suppressed_result(
        self,
        *,
        capability: str,
        intent: str,
        domain: str,
        language: str,
        required_terms: tuple[str, ...] = (),
    ) -> MesWmsDatabaseResult | None:
        capability_row = self._capability(capability)
        if capability_row.get("status") != "SUPPRESSED":
            return None
        reason_code = str(
            capability_row.get("reason_code") or REASON_DATASET_NOT_OBSERVED
        )
        if language == "ja":
            answer = (
                "このデータセットは今回のエクスポートで確認できないため、"
                "該当記録の有無を判定しません。"
            )
        else:
            answer = (
                "Dataset này chưa được quan sát trong export hiện tại nên tôi không "
                "kết luận có hay không có bản ghi."
            )
        return self._result(
            intent,
            [],
            answer,
            required_terms=required_terms,
            status="SUPPRESSED",
            reason_codes=(reason_code,),
            domain=domain,
        )

    def _capability_result_status(
        self,
        capability: str,
        default_status: str,
        default_reasons: tuple[str, ...] = (),
    ) -> tuple[str, tuple[str, ...]]:
        capability_row = self._capability(capability)
        status = str(capability_row.get("status") or default_status)
        reason = str(capability_row.get("reason_code") or "")
        return status, ((reason,) if reason else default_reasons)

    def _safe_evidence(self, datasets: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        allowed = (
            "dataset",
            "status",
            "reason_code",
            "source_state",
            "candidate_row_count",
            "inserted_row_count",
            "invalid_quantity_row_count",
            "source_as_of",
            "source_as_of_state",
            "source_as_of_basis",
            "source_timezone",
            "semantic_epoch",
        )
        return tuple(
            {
                key: row.get(key)
                for key in allowed
            }
            for dataset in datasets
            if (row := self._dataset_evidence(dataset))
        )

    def _result(
        self,
        intent: str,
        rows: list[dict[str, Any]],
        answer: str,
        *,
        required_terms: tuple[str, ...] = (),
        status: str = "AVAILABLE",
        reason_codes: tuple[str, ...] = (),
        domain: str = DATASET_CURRENT,
        evidence_domains: tuple[str, ...] | None = None,
        pagination: dict[str, Any] | None = None,
        grain: str | None = None,
    ) -> MesWmsDatabaseResult:
        metadata = self._metadata()
        if grain is None:
            grain = {
                DATASET_CURRENT: "process_id,item_code",
                DATASET_LEGACY: "item_code,item_lot_id,process_id,archive_date,source_id",
                DATASET_AUDIT: "raw_transaction_header_or_detail",
            }.get(domain, "")
        evidence_domains = evidence_domains or (domain,)
        evidence = self._safe_evidence(evidence_domains)
        primary = next(
            (row for row in evidence if row.get("dataset") == domain),
            evidence[0] if evidence else {},
        )
        source_as_of = str(primary.get("source_as_of") or "")
        source_timezone = str(primary.get("source_timezone") or "unverified")
        answer = self._with_freshness(
            answer,
            source_as_of=source_as_of,
            source_timezone=source_timezone,
        )
        return MesWmsDatabaseResult(
            intent=intent,
            rows=rows,
            imported_at=metadata.get("imported_at", ""),
            source_as_of=source_as_of,
            fallback_answer=answer,
            required_terms=required_terms,
            status=status,
            reason_codes=reason_codes,
            domain=domain,
            schema_version=metadata.get("schema_version", ""),
            data_contract_version=metadata.get("data_contract_version", ""),
            semantic_contract_version=metadata.get(
                "semantic_contract_version", ""
            ),
            semantic_epoch=str(primary.get("semantic_epoch") or ""),
            source_as_of_state=str(primary.get("source_as_of_state") or ""),
            source_as_of_basis=str(primary.get("source_as_of_basis") or ""),
            source_timezone=source_timezone,
            dataset_evidence=evidence,
            grain=grain,
            pagination=pagination,
        )

    def sql_agent_result(
        self,
        rows: list[dict[str, Any]],
        answer: str,
        *,
        domain: str = DATASET_CURRENT,
        evidence_domains: tuple[str, ...] | None = None,
    ) -> MesWmsDatabaseResult:
        """Wrap an LLM-planned query in the public WMS metadata contract."""
        return self._result(
            "wms_sql_agent_answer",
            rows,
            answer,
            status="PARTIAL",
            reason_codes=(REASON_SQL_AGENT_UNVERIFIED,),
            domain=domain,
            evidence_domains=evidence_domains,
        )

    @staticmethod
    def _with_freshness(
        answer: str,
        *,
        source_as_of: str,
        source_timezone: str,
    ) -> str:
        japanese_answer = bool(
            re.search(r"[぀-ヿ㐀-鿿]", answer or "")
        )
        if not source_as_of:
            if japanese_answer:
                freshness = "データ基準時点: 未確認（タイムゾーン未確認）。"
            else:
                freshness = "Mốc dữ liệu nguồn: chưa xác nhận (timezone chưa xác nhận)."
            return f"{answer}\n\n{freshness}"
        if japanese_answer:
            freshness = f"データ基準時点: {source_as_of}（タイムゾーン未確認）。"
        else:
            timezone_text = (
                "timezone chưa xác nhận"
                if not source_timezone or source_timezone == "unverified"
                else f"timezone {source_timezone}"
            )
            freshness = f"Mốc dữ liệu nguồn: {source_as_of} ({timezone_text})."
        if freshness in answer:
            return answer
        return f"{answer}\n\n{freshness}"

    def _executive_overview(self, language: str) -> MesWmsDatabaseResult:
        quality = self._fetch_one("SELECT * FROM v_wms_current_quality") or {}
        top_rows = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped,
                   COUNT(DISTINCT item_code) AS distinct_item_count,
                   COUNT(*) AS process_item_count,
                   MAX(latest_update) AS latest_update
            FROM v_wms_current_balance_by_process_item
            GROUP BY process_id, process_name, process_mapped
            ORDER BY distinct_item_count DESC, process_id
            LIMIT ?
            """,
            (5,),
        )
        total_rows = int(quality.get("current_row_count") or 0)
        mapped_rows = int(quality.get("mapped_process_row_count") or 0)
        mapping_rate = (mapped_rows / total_rows * 100) if total_rows else 0.0
        distinct_items = int(quality.get("distinct_item_count") or 0)
        distinct_processes = int(quality.get("distinct_process_code_count") or 0)
        as_of = str(quality.get("source_as_of") or "chưa xác nhận")
        if language == "ja":
            answer = (
                f"MKHC WMSの工程倉庫データは{as_of}時点（タイムゾーン未確認）です。"
                f"{distinct_items}件の資材コード、{distinct_processes}件の工程コードを記録し、"
                f"工程マッピング率は{mapping_rate:.1f}%です。単位マスターがないため、"
                "異なる資材を合算した工場全体の在庫数量やランキングは表示しません。"
                "最小在庫、期限、推移、仕掛品、ボトルネックのKPIも現在は抑止されています。"
            )
        else:
            answer = (
                f"Theo WMS MKHC, dữ liệu tồn nguyên vật liệu tại kho công đoạn có "
                f"mốc gần nhất {as_of} (timezone chưa xác nhận). Snapshot ghi nhận "
                f"{self._number(distinct_items)} mã vật tư trên "
                f"{self._number(distinct_processes)} mã công đoạn; độ phủ ánh xạ "
                f"công đoạn là {self._percent(mapping_rate)}. Tôi không cộng số lượng "
                "giữa các mã vật tư vì chưa có master đơn vị tính. Các KPI dưới tồn "
                "tối thiểu, hạn dùng, xu hướng, WIP và bottleneck đang bị khóa do chưa "
                "đủ dữ liệu kiểm chứng."
            )
        return self._result(
            "wms_executive_overview",
            [{"quality": quality, "processes": top_rows}],
            answer,
            required_terms=(str(distinct_items), str(distinct_processes)),
            status="PARTIAL",
            reason_codes=self.SUPPRESSION_CODES,
        )

    def get_executive_matrix_data(self, limit_processes: int = 10) -> dict[str, Any]:
        """Tạo ma trận report chỉ khi snapshot vượt compatibility contract v4."""
        if not self.available:
            raise MesWmsDatabaseError("WMS snapshot chưa sẵn sàng.")
        if not self.compatibility().get("compatible"):
            raise MesWmsDatabaseError("WMS snapshot không tương thích contract v4.")
        if limit_processes < 1 or limit_processes > 20:
            raise ValueError("limit_processes phải nằm trong khoảng 1..20")
        quality = self._fetch_one("SELECT * FROM v_wms_current_quality") or {}
        top_processes = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped,
                   COUNT(DISTINCT item_code) AS distinct_item_count,
                   COUNT(*) AS process_item_count
            FROM v_wms_current_balance_by_process_item
            GROUP BY process_id, process_name, process_mapped
            ORDER BY distinct_item_count DESC, process_id
            LIMIT ?
            """,
            (limit_processes,),
        )
        process_ids = [p["process_id"] for p in top_processes]
        if not process_ids:
            return {"quality": quality, "processes": [], "items": [], "matrix": {}}

        placeholders = ",".join("?" for _ in process_ids)
        matrix_rows = self._fetch_all(
            f"""
            SELECT process_id, item_code, quantity_decimal
            FROM v_wms_current_balance_by_process_item
            WHERE process_id IN ({placeholders})
            ORDER BY item_code, process_id
            """,
            tuple(process_ids),
        )

        items_set = {r["item_code"] for r in matrix_rows}
        matrix: dict[str, dict[str, Any]] = {}
        for r in matrix_rows:
            pid = r["process_id"]
            icode = r["item_code"]
            qty = r["quantity_decimal"]
            if pid not in matrix:
                matrix[pid] = {}
            matrix[pid][icode] = qty

        return {
            "quality": quality,
            "processes": top_processes,
            "items": sorted(items_set),
            "matrix": matrix,
        }

    @classmethod
    def _page_metadata(
        cls, total_count: int, returned_count: int
    ) -> dict[str, Any]:
        return {
            "page": 1,
            "page_size": cls.DEFAULT_LIST_LIMIT,
            "total_count": total_count,
            "has_more": returned_count < total_count,
        }

    def _item_by_process(
        self, item_code: str, language: str
    ) -> MesWmsDatabaseResult:
        count_row = self._fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM v_wms_current_balance_by_process_item
            WHERE item_code = ?
            """,
            (item_code,),
        ) or {}
        total_count = int(count_row.get("total") or 0)
        rows = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped, item_code,
                   quantity_decimal, latest_update
            FROM v_wms_current_balance_by_process_item
            WHERE item_code = ?
            ORDER BY process_mapped DESC, process_id
            LIMIT ?
            """,
            (item_code, self.DEFAULT_LIST_LIMIT),
        )
        if not rows:
            return self._invalid_current_quantity(
                item_code, None, language
            ) or self._not_found("item", item_code, language)
        pagination = self._page_metadata(total_count, len(rows))
        truncated_note = (
            f" 全{total_count}件中{len(rows)}件を表示しています。"
            if language == "ja" and pagination["has_more"]
            else f" Đang hiển thị {len(rows)}/{total_count} dòng."
            if pagination["has_more"]
            else ""
        )
        lines = [self._row_description(row, language) for row in rows]
        if len(lines) > 1:
            formatted_list = "\n- " + "\n- ".join(lines)
            sep = "\n\n"
        else:
            formatted_list = "; ".join(lines)
            sep = " "

        if language == "ja":
            answer = (
                f"MKHC WMSでは、資材コード{item_code}の工程倉庫別在庫は次のとおりです:{formatted_list}{sep}"
                "単位マスターが未確認のため、数量には単位を付けていません。"
                + truncated_note
            )
        else:
            answer = (
                f"Theo WMS MKHC, tồn của mã vật tư {item_code} theo kho công đoạn:{formatted_list}{sep}"
                "Snapshot chưa có master đơn vị tính nên số lượng được giữ riêng "
                "theo mã vật tư và không cộng với mã khác."
                + truncated_note
            )
        required = (item_code,) + tuple(
            str(row["quantity_decimal"]) for row in rows[:3]
        )
        return self._result(
            "wms_item_by_process",
            rows,
            answer,
            required_terms=required,
            status="PARTIAL",
            reason_codes=("UOM_MASTER_UNAVAILABLE",),
            pagination=pagination,
        )

    def _process_inventory(
        self, process_id: str, language: str
    ) -> MesWmsDatabaseResult:
        count_row = self._fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM v_wms_current_balance_by_process_item
            WHERE process_id = ?
            """,
            (process_id,),
        ) or {}
        total_count = int(count_row.get("total") or 0)
        rows = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped, item_code,
                   quantity_decimal, latest_update
            FROM v_wms_current_balance_by_process_item
            WHERE process_id = ?
            ORDER BY item_code
            LIMIT ?
            """,
            (process_id, self.DEFAULT_LIST_LIMIT),
        )
        if not rows:
            return self._not_found("process", process_id, language)
        pagination = self._page_metadata(total_count, len(rows))
        process_name = rows[0].get("process_name") or (
            "名称未マッピング" if language == "ja" else "chưa ánh xạ tên"
        )
        lines = [self._row_description(row, language) for row in rows]
        if len(lines) > 1:
            formatted_list = "\n- " + "\n- ".join(lines)
            sep = "\n\n"
        else:
            formatted_list = " " + "; ".join(lines)
            sep = " "

        if language == "ja":
            display_count = (
                f"全{total_count}件中{len(rows)}件"
                if pagination["has_more"]
                else f"{len(rows)}件"
            )
            answer = (
                f"MKHC WMSの工程{process_id}（{process_name}）では、"
                f"{display_count}の資材コードを表示します:{formatted_list}{sep}"
                "異なる資材の数量は合算していません。"
            )
        else:
            display_count = (
                f"{self._number(len(rows))}/{self._number(total_count)}"
                if pagination["has_more"]
                else self._number(len(rows))
            )
            answer = (
                f"Theo WMS MKHC, công đoạn {process_id} ({process_name}) hiển thị "
                f"{display_count} mã vật tư:{formatted_list}{sep}"
                "Tôi không cộng số lượng giữa các mã vật tư vì chưa có master đơn vị tính."
            )
        return self._result(
            "wms_process_inventory",
            rows,
            answer,
            required_terms=(process_id, str(rows[0]["item_code"])),
            status="PARTIAL",
            reason_codes=("UOM_MASTER_UNAVAILABLE",),
            pagination=pagination,
        )

    def _item_at_process(
        self, item_code: str, process_id: str, language: str
    ) -> MesWmsDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped, item_code,
                   quantity_decimal, latest_update
            FROM v_wms_current_balance_by_process_item
            WHERE item_code = ? AND process_id = ?
            LIMIT 1
            """,
            (item_code, process_id),
        )
        if not rows:
            return self._invalid_current_quantity(
                item_code, process_id, language
            ) or self._not_found(
                "item_process", f"{item_code}/{process_id}", language
            )
        row = rows[0]
        if language == "ja":
            answer = (
                f"MKHC WMSでは、工程{process_id}の資材コード{item_code}の在庫数量は"
                f"{self._quantity(row['quantity_decimal'], language)}です。最終更新は"
                f"{row.get('latest_update') or '未確認'}で、単位マスターは未確認です。"
            )
        else:
            answer = (
                f"Theo WMS MKHC, mã vật tư {item_code} tại công đoạn {process_id} "
                f"có số lượng {self._quantity(row['quantity_decimal'], language)}; cập nhật gần "
                f"nhất {row.get('latest_update') or 'chưa xác nhận'}. Snapshot chưa có "
                "master đơn vị tính nên tôi không tự gắn đơn vị."
            )
        return self._result(
            "wms_item_at_process",
            rows,
            answer,
            required_terms=(item_code, process_id, str(row["quantity_decimal"])),
            status="PARTIAL",
            reason_codes=("UOM_MASTER_UNAVAILABLE",),
        )

    def _process_catalog(self, language: str) -> MesWmsDatabaseResult:
        rows = self._fetch_all(
            """
            SELECT process_id, process_name, process_mapped,
                   COUNT(DISTINCT item_code) AS distinct_item_count
            FROM v_wms_current_balance_by_process_item
            GROUP BY process_id, process_name, process_mapped
            ORDER BY distinct_item_count DESC, process_id
            """
        )
        if not rows:
            return self._not_found("processes", "", language)

        lines = [
            f"{row['process_id']} ({row['process_name'] or ('名称未マッピング' if language == 'ja' else 'chưa ánh xạ tên')}): {row['distinct_item_count']} " + ("資材コード" if language == "ja" else "mã vật tư")
            for row in rows
        ]
        formatted_list = "\n- " + "\n- ".join(lines)
        if language == "ja":
            answer = (
                f"MKHC WMSのcurrent balanceに存在する工程コードの一覧（全{len(rows)}工程）は次のとおりです:{formatted_list}\n\n"
                "各工程に存在する資材コードの種類数を表示しています。これは工程マスター全件や"
                "資材確認が必要と設定された工程の一覧ではありません。"
            )
        else:
            answer = (
                f"Theo WMS MKHC, danh sách mã công đoạn xuất hiện trong current balance "
                f"snapshot (tổng cộng {len(rows)} công đoạn):{formatted_list}\n\n"
                "Số liệu thể hiện số chủng loại mã vật tư có trong snapshot tại mỗi công đoạn; "
                "đây không phải toàn bộ process master hoặc danh sách công đoạn được cấu hình "
                "bắt buộc kiểm tra vật tư."
            )
        return self._result(
            "wms_process_catalog",
            rows,
            answer,
            required_terms=(str(rows[0]["process_id"]),),
            status="AVAILABLE",
            domain="CURRENT_BALANCE",
        )

    def _current_lot_suppressed(
        self,
        item_code: str,
        item_lot_id: str,
        process_id: str,
        language: str,
    ) -> MesWmsDatabaseResult:
        if language == "ja":
            answer = (
                f"現行WMS在庫の業務キーは工程{process_id}と資材{item_code}です。"
                f"資材ロット{item_lot_id}は現行データでは意味のあるキーではないため、"
                "ロット別の現行数量は回答しません。"
            )
        else:
            answer = (
                f"Grain tồn current của WMS là công đoạn {process_id} + mã vật tư "
                f"{item_code}. Lot vật tư {item_lot_id} không còn là khóa có ý nghĩa "
                "trong current, nên tôi không trả số lượng current theo lot."
            )
        return self._result(
            "wms_current_lot_lookup_suppressed",
            [],
            answer,
            required_terms=(item_code, item_lot_id, process_id),
            status="SUPPRESSED",
            reason_codes=(REASON_CURRENT_LOT_UNAVAILABLE,),
            domain=DATASET_CURRENT,
        )

    @staticmethod
    def _archive_page(question: str) -> tuple[int, int] | None:
        normalized = normalize_wms_text(question)
        page_match = re.search(r"(?:trang|page)\s*(-?\d+)", normalized)
        ja_page = re.search(r"第\s*(-?\d+)\s*ページ", question or "")
        page_text = (
            page_match.group(1)
            if page_match
            else ja_page.group(1)
            if ja_page
            else "1"
        )
        size_match = re.search(
            r"(?:moi trang|page size|per page)\s*(-?\d+)", normalized
        )
        page_size = int(size_match.group(1)) if size_match else 20
        page = int(page_text)
        if page < 1 or page_size < 1 or page_size > 50:
            return None
        return page, page_size

    @staticmethod
    def _archive_date_range(question: str) -> tuple[str, str] | None:
        values = re.findall(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", question or "")
        if not values:
            return "", ""
        if len(values) > 2:
            return None
        try:
            parsed = [datetime.fromisoformat(value) for value in values]
        except ValueError:
            return None
        start = parsed[0].date().isoformat()
        end = parsed[-1].date().isoformat()
        if start > end:
            return None
        return start, end

    def _snapshot_history(
        self,
        item_code: str,
        item_lot_id: str,
        process_id: str,
        language: str,
        question: str,
    ) -> MesWmsDatabaseResult:
        suppressed = self._dataset_suppressed_result(
            capability="LEGACY_ARCHIVE_EXACT_KEY_QUERY",
            intent="wms_legacy_archive_unobserved",
            domain=DATASET_LEGACY,
            language=language,
            required_terms=(item_code, item_lot_id, process_id),
        )
        if suppressed is not None:
            return suppressed
        page_config = self._archive_page(question)
        date_range = self._archive_date_range(question)
        if page_config is None or date_range is None:
            answer = (
                "ページ番号は1以上、1ページ50件以下で、日付はYYYY-MM-DDの昇順で指定してください。"
                if language == "ja"
                else (
                    "Vui lòng dùng trang từ 1, tối đa 50 dòng/trang và khoảng ngày "
                    "YYYY-MM-DD theo thứ tự từ ngày đến ngày."
                )
            )
            return self._result(
                "wms_legacy_archive_parameters_invalid",
                [],
                answer,
                status="PARTIAL",
                domain=DATASET_LEGACY,
            )
        page, page_size = page_config
        start_date, end_date = date_range
        where = ["item_code = ?", "item_lot_id = ?", "process_id = ?"]
        parameters: list[Any] = [item_code, item_lot_id, process_id]
        if start_date:
            where.append("date(archive_date) >= date(?)")
            parameters.append(start_date)
        if end_date:
            where.append("date(archive_date) <= date(?)")
            parameters.append(end_date)
        condition = " AND ".join(where)
        count_row = self._fetch_one(
            f"SELECT COUNT(*) AS total FROM v_wms_legacy_archive_exact_key WHERE {condition}",
            tuple(parameters),
        ) or {}
        total_count = int(count_row.get("total") or 0)
        offset = (page - 1) * page_size
        rows = self._fetch_all(
            f"""
            SELECT source_id, archive_id, archive_date, item_code,
                   item_lot_id, process_id, process_name,
                   quantity_decimal, quantity_valid, quantity_error
            FROM v_wms_legacy_archive_exact_key
            WHERE {condition}
            ORDER BY archive_date DESC, source_id DESC
            LIMIT ? OFFSET ?
            """,
            tuple(parameters + [page_size, offset]),
        )
        pagination = {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "has_more": offset + len(rows) < total_count,
        }
        capability_status, capability_reasons = self._capability_result_status(
            "LEGACY_ARCHIVE_EXACT_KEY_QUERY",
            "AVAILABLE",
        )
        if not rows:
            answer = (
                f"旧方式アーカイブには{item_code}/{item_lot_id}/{process_id}の"
                "記録が指定範囲内にありません。"
                if language == "ja"
                else (
                    f"Legacy archive không có bản ghi cho {item_code}/"
                    f"{item_lot_id}/{process_id} trong phạm vi đã chọn."
                )
            )
            return self._result(
                "wms_legacy_archive_exact_key_not_found",
                [],
                answer,
                required_terms=(item_code, item_lot_id, process_id),
                status=capability_status,
                reason_codes=capability_reasons,
                domain=DATASET_LEGACY,
                pagination=pagination,
            )
        lines = [
            f"{row.get('archive_date') or '?'} / {row.get('archive_id') or '?'}: "
            f"{self._quantity(row.get('quantity_decimal'), language)}"
            for row in rows
        ]
        if language == "ja":
            answer = (
                f"資材{item_code}、資材ロット{item_lot_id}、工程{process_id}の"
                "旧方式アーカイブ記録: " + "; ".join(lines)
                + "。これは旧セマンティック期間の記録一覧であり、現行在庫との比較、"
                "推移や増減の判定ではありません。"
            )
        else:
            answer = (
                f"Legacy archive khớp mã vật tư {item_code}, lot vật tư "
                f"{item_lot_id}, công đoạn {process_id}: " + "; ".join(lines)
                + ". Đây là dữ liệu thuộc semantic epoch cũ, không so sánh với current "
                "và không kết luận xu hướng, delta hoặc tăng/giảm."
            )
        return self._result(
            "wms_legacy_archive_exact_key",
            rows,
            answer,
            required_terms=(item_code, item_lot_id, process_id),
            status=capability_status,
            reason_codes=capability_reasons,
            domain=DATASET_LEGACY,
            pagination=pagination,
        )

    def _snapshot_presence(
        self,
        item_code: str,
        item_lot_id: str,
        process_id: str,
        language: str,
    ) -> MesWmsDatabaseResult:
        capability = self._capability("LEGACY_ARCHIVE_EXACT_KEY_QUERY")
        if capability.get("status") == "SUPPRESSED":
            reason_code = str(
                capability.get("reason_code") or REASON_DATASET_NOT_OBSERVED
            )
            row = {
                "item_code": item_code,
                "item_lot_id": item_lot_id,
                "process_id": process_id,
                "legacy_archive_exact_key_present": "NOT_EVALUATED",
                "current_exact_lot_presence": "NOT_EVALUATED",
                "comparison_eligible": False,
            }
            answer = (
                "旧方式アーカイブは今回のエクスポートで確認できないため、"
                "完全一致キーの有無を評価しません。現行在庫もロットを業務キーと"
                "していないため評価せず、差分や増減は判定できません。"
                if language == "ja"
                else (
                    "Legacy archive chưa được quan sát trong export hiện tại nên không "
                    "đánh giá presence của exact-key. Current cũng không dùng lot làm "
                    "khóa nghiệp vụ; không kết luận delta hoặc tăng/giảm."
                )
            )
            return self._result(
                "wms_cross_era_presence_unobserved",
                [row],
                answer,
                required_terms=(item_code, item_lot_id, process_id),
                status="SUPPRESSED",
                reason_codes=(REASON_CROSS_ERA_UNCOMPARABLE, reason_code),
                domain=DATASET_LEGACY,
                evidence_domains=(DATASET_LEGACY, DATASET_CURRENT),
            )
        archive = self._fetch_one(
            """
            SELECT 1 AS present
            FROM v_wms_legacy_archive_exact_key
            WHERE item_code = ? AND item_lot_id = ? AND process_id = ?
            LIMIT 1
            """,
            (item_code, item_lot_id, process_id),
        )
        row = {
            "item_code": item_code,
            "item_lot_id": item_lot_id,
            "process_id": process_id,
            "legacy_archive_exact_key_present": bool(archive),
            "current_exact_lot_presence": "NOT_EVALUATED",
            "comparison_eligible": False,
        }
        if language == "ja":
            answer = (
                f"旧方式アーカイブでは資材{item_code}、ロット{item_lot_id}、工程"
                f"{process_id}の完全一致キーは"
                f"{'確認できます' if archive else '確認できません'}。現行在庫はロットを"
                "業務キーとしていないため評価しません。期間・工程コード体系・粒度が"
                "異なるため、在庫消滅、入出庫、差分または増減とは解釈できません。"
            )
        else:
            answer = (
                f"Legacy archive {'có' if archive else 'không có'} khóa chính xác "
                f"{item_code}/{item_lot_id}/{process_id}. Current không còn dùng lot làm "
                "khóa nghiệp vụ nên không đánh giá presence theo lot. Hai nguồn khác thời "
                "kỳ, namespace và grain; không được diễn giải là hết tồn, nhập/xuất, "
                "delta hoặc tăng/giảm."
            )
        return self._result(
            "wms_cross_era_presence_diagnostic",
            [row],
            answer,
            required_terms=(item_code, item_lot_id, process_id),
            status="SUPPRESSED",
            reason_codes=(REASON_CROSS_ERA_UNCOMPARABLE,),
            domain=DATASET_LEGACY,
            evidence_domains=(DATASET_LEGACY, DATASET_CURRENT),
        )

    def _transaction_audit_by_id(
        self, trans_id: str, language: str
    ) -> MesWmsDatabaseResult:
        suppressed = self._dataset_suppressed_result(
            capability="RAW_TRANSACTION_AUDIT_QUERY",
            intent="wms_raw_transaction_audit_unobserved",
            domain=DATASET_AUDIT,
            language=language,
            required_terms=(trans_id,),
        )
        if suppressed is not None:
            return suppressed
        count_row = self._fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM v_wms_raw_transaction_audit
            WHERE trans_id = ?
            """,
            (trans_id,),
        ) or {}
        total_count = int(count_row.get("total") or 0)
        rows = self._fetch_all(
            """
            SELECT trans_id, trans_code, trans_name, trans_date,
                   process_id, process_name, item_code, item_lot_id,
                   header_quantity_decimal, header_quantity_valid,
                   detail_quantity_decimal, detail_quantity_valid,
                   raw_trans_status, raw_deleted
            FROM v_wms_raw_transaction_audit
            WHERE trans_id = ?
            ORDER BY item_lot_id, trans_id
            LIMIT ?
            """,
            (trans_id, self.DEFAULT_LIST_LIMIT),
        )
        return self._transaction_audit_result(
            rows,
            trans_id,
            language,
            total_count=total_count,
        )

    def _transaction_audit_by_key(
        self,
        item_code: str,
        item_lot_id: str,
        process_id: str,
        language: str,
    ) -> MesWmsDatabaseResult:
        suppressed = self._dataset_suppressed_result(
            capability="RAW_TRANSACTION_AUDIT_QUERY",
            intent="wms_raw_transaction_audit_unobserved",
            domain=DATASET_AUDIT,
            language=language,
            required_terms=(item_code, item_lot_id, process_id),
        )
        if suppressed is not None:
            return suppressed
        count_row = self._fetch_one(
            """
            SELECT COUNT(*) AS total
            FROM v_wms_raw_transaction_audit
            WHERE item_code = ? AND item_lot_id = ? AND process_id = ?
            """,
            (item_code, item_lot_id, process_id),
        ) or {}
        total_count = int(count_row.get("total") or 0)
        rows = self._fetch_all(
            """
            SELECT trans_id, trans_code, trans_name, trans_date,
                   process_id, process_name, item_code, item_lot_id,
                   header_quantity_decimal, header_quantity_valid,
                   detail_quantity_decimal, detail_quantity_valid,
                   raw_trans_status, raw_deleted
            FROM v_wms_raw_transaction_audit
            WHERE item_code = ? AND item_lot_id = ? AND process_id = ?
            ORDER BY trans_date DESC, trans_id
            LIMIT ?
            """,
            (item_code, item_lot_id, process_id, self.DEFAULT_LIST_LIMIT),
        )
        return self._transaction_audit_result(
            rows,
            f"{item_code}/{item_lot_id}/{process_id}",
            language,
            total_count=total_count,
        )

    def _transaction_audit_result(
        self,
        rows: list[dict[str, Any]],
        key_label: str,
        language: str,
        *,
        total_count: int,
    ) -> MesWmsDatabaseResult:
        pagination = self._page_metadata(total_count, len(rows))
        if not rows:
            answer = (
                f"Raw transaction audit không có bản ghi cho {key_label}."
                if language != "ja"
                else f"Raw transaction auditに{key_label}の記録がありません。"
            )
            return self._result(
                "wms_raw_transaction_audit_not_found",
                [],
                answer,
                required_terms=(key_label,),
                status="AVAILABLE",
                domain=DATASET_AUDIT,
                pagination=pagination,
            )
        lines = []
        for row in rows:
            trans_name = str(row.get("trans_name") or "")
            label = (
                trans_name
                if trans_name and "?" not in trans_name and "�" not in trans_name
                else (
                    "source label unreadable"
                    if language == "ja"
                    else "nhãn nguồn không đọc được"
                )
            )
            header_quantity = (
                self._quantity(row.get("header_quantity_decimal"), language)
                if row.get("header_quantity_valid")
                else ("不明" if language == "ja" else "chưa rõ")
            )
            detail_quantity = (
                self._quantity(row.get("detail_quantity_decimal"), language)
                if row.get("detail_quantity_valid")
                else ("不明" if language == "ja" else "chưa rõ")
            )
            lines.append(
                f"{row.get('trans_id')}: {row.get('trans_code')} "
                f"({label}), {row.get('trans_date') or '?'}, "
                f"header_qty={header_quantity}, detail_qty={detail_quantity}, "
                f"raw_status={row.get('raw_trans_status') or '?'}, "
                f"raw_deleted={row.get('raw_deleted') or '?'}"
            )
        if language == "ja":
            truncated = (
                f" 全{total_count}件中{len(rows)}件を表示しています。"
                if pagination["has_more"]
                else ""
            )
            answer = (
                "WMS監査レコード: "
                + "; ".join(lines)
                + "。コード名はソースのラベルです。入出庫方向、完了状態、"
                "差引数量は判定していません。"
                + truncated
            )
        else:
            truncated = (
                f" Đang hiển thị {len(rows)}/{total_count} bản ghi."
                if pagination["has_more"]
                else ""
            )
            answer = (
                "Các bản ghi audit WMS: "
                + "; ".join(lines)
                + ". Tên mã chỉ là nhãn từ nguồn; tôi không suy diễn chiều nhập/xuất, "
                "trạng thái hoàn thành hay số lượng ròng."
                + truncated
            )
        return self._result(
            "wms_raw_transaction_audit",
            rows,
            answer,
            required_terms=(str(rows[0].get("trans_id") or key_label),),
            status="PARTIAL",
            reason_codes=(REASON_TRANSACTION_SEMANTICS_UNAVAILABLE,),
            domain=DATASET_AUDIT,
            pagination=pagination,
        )

    def _exact_key_required(self, language: str) -> MesWmsDatabaseResult:
        answer = (
            "完全一致で確認するため、資材コード、資材ロットID、工程コードの3項目を"
            "すべて指定してください。"
            if language == "ja"
            else (
                "Để tra cứu chính xác, vui lòng cung cấp đủ mã vật tư, lot vật tư "
                "và mã công đoạn. Tôi không tự suy đoán khóa còn thiếu."
            )
        )
        return self._result(
            "wms_exact_key_required", [], answer, status="PARTIAL"
        )

    def _unsupported_kpi_result(
        self,
        capability: str,
        reason_code: str,
        language: str,
    ) -> MesWmsDatabaseResult:
        labels_vi = {
            "MIN_STOCK": "tồn tối thiểu/thiếu vật tư",
            "EXPIRY": "hạn dùng",
            "WINDOW_TIME": "window-time/thời gian lưu kho",
            "TREND": "xu hướng tồn kho",
            "PRODUCTION_WIP": "WIP đang sản xuất",
            "BOTTLENECK": "bottleneck công đoạn",
        }
        labels_ja = {
            "MIN_STOCK": "最低在庫・不足判定",
            "EXPIRY": "使用期限",
            "WINDOW_TIME": "ウィンドウタイム・保管時間",
            "TREND": "在庫推移",
            "PRODUCTION_WIP": "製造中の仕掛品（WIP）",
            "BOTTLENECK": "工程ボトルネック",
        }
        if language == "ja":
            answer = (
                f"{labels_ja.get(capability, capability)}は、信頼できるデータソースと"
                "業務ルールが未確認のため回答できません。現在のWMS contractでは、"
                "資材コード・工程別の現行残高のみ確認できます。"
            )
        else:
            answer = (
                f"Tôi chưa thể trả lời KPI {labels_vi.get(capability, capability)} vì "
                "chưa có nguồn dữ liệu và quy tắc nghiệp vụ đã được xác minh. WMS "
                "hiện chỉ hỗ trợ current balance theo mã vật tư và công đoạn."
            )
        return self._result(
            f"wms_{capability.lower()}_suppressed",
            [],
            answer,
            status="SUPPRESSED",
            reason_codes=(reason_code,),
        )

    def _completed_movements_suppressed(
        self, language: str
    ) -> MesWmsDatabaseResult:
        answer = (
            "WMSのraw transaction auditにはソースのコードと状態をそのまま表示できますが、"
            "完了済み移動、入出庫方向、差引数量として解釈できません。"
            if language == "ja"
            else (
                "Raw transaction audit WMS chỉ hiển thị mã và trạng thái thô từ nguồn; "
                "chưa có contract để kết luận giao dịch hoàn thành, chiều nhập/xuất "
                "hoặc số lượng ròng."
            )
        )
        return self._result(
            "wms_completed_movements_suppressed",
            [],
            answer,
            status="SUPPRESSED",
            reason_codes=(REASON_COMPLETED_MOVEMENTS_UNAVAILABLE,),
            domain=DATASET_AUDIT,
        )

    def _ambiguity_result(self, language: str) -> MesWmsDatabaseResult:
        answer = (
            "ご質問は、工程倉庫の資材在庫（WMS）と製造中の仕掛品（WIP）のどちらを"
            "意味しますか。現在のWMS contractは工程倉庫の資材残高のみ対応しています。"
            if language == "ja"
            else (
                "Câu hỏi đang trộn hai phạm vi: tồn nguyên vật liệu tại kho công đoạn "
                "(WMS) và hàng dở dang đang gia công (WIP). WMS hiện chỉ trả lời"
                "tồn kho công đoạn; vui lòng xác nhận phạm vi cần xem."
            )
        )
        return self._result(
            "wms_wip_ambiguity",
            [],
            answer,
            status="SUPPRESSED",
            reason_codes=("PRODUCTION_WIP_SOURCE_UNAVAILABLE",),
        )

    def _cross_item_aggregate_suppression(
        self, language: str
    ) -> MesWmsDatabaseResult:
        answer = (
            "単位・換算マスターが確認できないため、異なる資材コードの在庫数量を"
            "合算またはランキングできません。1つの資材コードを指定して工程別に"
            "確認するか、1つの工程を指定して資材ごとの数量を個別に確認してください。"
            if language == "ja"
            else (
                "Chưa có master đơn vị tính và quy đổi đã được xác minh nên tôi không "
                "thể cộng hoặc xếp hạng số lượng tồn giữa các mã vật tư khác nhau. "
                "Bạn có thể hỏi một mã vật tư theo từng công đoạn, hoặc chọn một công "
                "đoạn để xem riêng số lượng của từng mã vật tư."
            )
        )
        return self._result(
            "wms_cross_item_aggregate_suppressed",
            [],
            answer,
            status="SUPPRESSED",
            reason_codes=("UOM_MASTER_UNAVAILABLE",),
        )

    def _clarify_scope(self, language: str) -> MesWmsDatabaseResult:
        answer = (
            "工程コードまたは資材コードを指定してください。例: 「工程P-01の在庫」または"
            "「資材コードITEM-01の工程別在庫」。"
            if language == "ja"
            else (
                "Vui lòng nêu mã công đoạn hoặc mã vật tư cần xem, ví dụ: "
                '"tồn kho công đoạn P-01" hoặc "mã vật tư ITEM-01 tồn ở công đoạn nào".'
            )
        )
        return self._result("wms_scope_clarification", [], answer, status="PARTIAL")

    def _unavailable_result(self, language: str) -> MesWmsDatabaseResult:
        answer = (
            "WMSスナップショットがまだ利用できないため、在庫数量を確認できません。"
            if language == "ja"
            else "WMS snapshot chưa sẵn sàng nên tôi chưa thể xác minh số liệu tồn kho."
        )
        answer = self._with_freshness(
            answer,
            source_as_of="",
            source_timezone="unverified",
        )
        return MesWmsDatabaseResult(
            intent="wms_unavailable",
            rows=[],
            imported_at="",
            source_as_of="",
            fallback_answer=answer,
            status="SUPPRESSED",
            reason_codes=(REASON_SNAPSHOT_UNAVAILABLE,),
            domain="SUPPRESSED",
        )

    def _incompatible_result(self, language: str) -> MesWmsDatabaseResult:
        answer = (
            "WMSスナップショットが現在のデータ契約と互換性がないため、"
            "在庫数量を回答しません。"
            if language == "ja"
            else (
                "WMS snapshot không tương thích với data contract hiện hành nên tôi "
                "không trả số liệu tồn kho."
            )
        )
        answer = self._with_freshness(
            answer,
            source_as_of="",
            source_timezone="unverified",
        )
        return MesWmsDatabaseResult(
            intent="wms_incompatible",
            rows=[],
            imported_at="",
            source_as_of="",
            fallback_answer=answer,
            status="SUPPRESSED",
            reason_codes=(
                REASON_SNAPSHOT_INCOMPATIBLE,
                REASON_SOURCE_AS_OF_UNCONFIRMED,
            ),
            domain="SUPPRESSED",
        )

    def _invalid_current_quantity(
        self, item_code: str, process_id: str | None, language: str
    ) -> MesWmsDatabaseResult | None:
        where = "item_code = ?"
        parameters: tuple[Any, ...] = (item_code,)
        if process_id:
            where += " AND process_id = ?"
            parameters = (item_code, process_id)
        rows = self._fetch_all(
            f"""
            SELECT item_code, process_id, quantity_error, time_update
            FROM wms_current_balances
            WHERE {where} AND quantity_valid = 0
            ORDER BY process_id
            LIMIT ?
            """,
            parameters + (self.DEFAULT_LIST_LIMIT,),
        )
        if not rows:
            return None
        scope = f" tại công đoạn {process_id}" if process_id else ""
        if language == "ja":
            process = f"、工程{process_id}" if process_id else ""
            answer = (
                f"現行WMSに資材{item_code}{process}の行はありますが、数量は検証に"
                "失敗しているため在庫数量を回答できません。存在しないとは判定しません。"
            )
        else:
            answer = (
                f"Current WMS có bản ghi cho mã vật tư {item_code}{scope}, nhưng quantity "
                "không hợp lệ nên tôi chưa thể xác minh số lượng; không kết luận là không có tồn kho."
            )
        return self._result(
            "wms_current_quantity_invalid",
            rows,
            answer,
            required_terms=(item_code,) + ((process_id,) if process_id else ()),
            status="PARTIAL",
            reason_codes=("QUANTITY_EVIDENCE_INCOMPLETE",),
            domain=DATASET_CURRENT,
        )

    def _not_found(
        self, kind: str, value: str, language: str
    ) -> MesWmsDatabaseResult:
        if language == "ja":
            answer = f"MKHC WMSスナップショットで{value}の在庫情報が見つかりません。"
        else:
            label = {
                "item": "mã vật tư",
                "process": "công đoạn",
                "item_process": "mã vật tư/công đoạn",
            }.get(kind, "đối tượng")
            answer = f"Không tìm thấy tồn kho cho {label} {value} trong WMS MKHC snapshot."
        return self._result(
            f"wms_{kind}_not_found",
            [],
            answer,
            required_terms=(value,),
            status="AVAILABLE",
        )

    @classmethod
    def _extract_code_after(
        cls,
        question: str,
        label_pattern: str,
        *,
        stop_label_pattern: str | None = None,
    ) -> str | None:
        allow_spaces = stop_label_pattern is not None
        code_chars = r"[A-Za-z0-9_Đđ .-]+" if allow_spaces else r"[A-Za-z0-9_Đđ.-]+"
        match = re.search(
            rf"{label_pattern}{cls.LABEL_CODE_CONNECTOR}({code_chars})",
            question or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        candidate = match.group(1)
        if stop_label_pattern:
            # Bound a space-tolerant capture at the next label in the ORIGINAL
            # text: Vietnamese/Japanese label words carry diacritics or kana
            # outside the code charset, so the capture itself already stopped
            # mid-label and cannot be searched for the boundary.
            remainder = (question or "")[match.start(1):]
            stop = re.search(stop_label_pattern, remainder, flags=re.IGNORECASE)
            if stop:
                candidate = remainder[: stop.start()].rstrip()
        candidate = candidate.strip(".,;:?!。、 ")
        if not candidate:
            return None
        normalized = normalize_wms_text(candidate)
        if normalized in cls.NON_CODE_TOKENS:
            return None
        if cls._looks_like_measurement(candidate):
            return None
        return candidate if re.search(r"\d|[_-]", candidate) else None

    @classmethod
    def _looks_like_measurement(cls, candidate: str) -> bool:
        """Reject quantities, units and bare years captured after a label.

        WMS identifiers are never a bare number of four digits or fewer, and
        never a number glued to a unit of measure, so questions such as
        "vật tư 10kg" or "vật tư 2026" must not be read as an item code.
        """
        collapsed = candidate.replace(" ", "")
        if re.fullmatch(r"\d{1,4}", collapsed):
            return True
        measurement = re.fullmatch(
            r"(\d+(?:[.,]\d+)?)([A-Za-zĐđ]+)", collapsed
        )
        if measurement is None:
            return False
        return normalize_wms_text(measurement.group(2)) in cls.MEASUREMENT_UNITS

    @classmethod
    def _row_description(cls, row: dict[str, Any], language: str) -> str:
        process_id = row.get("process_id") or "?"
        process_name = row.get("process_name") or (
            "名称未マッピング" if language == "ja" else "chưa ánh xạ tên"
        )
        item_code = row.get("item_code") or "?"
        quantity = cls._quantity(row.get("quantity_decimal"), language)
        if language == "ja":
            return f"工程{process_id}（{process_name}）/ {item_code}: {quantity}"
        return f"{process_id} ({process_name}) / {item_code}: {quantity}"

    @staticmethod
    def _quantity(value: Any, language: str = "vi") -> str:
        if value is None:
            return "不明" if language == "ja" else "chưa rõ"
        number = Decimal(str(value))
        if language == "ja":
            if number == number.to_integral():
                return f"{int(number):,}"
            whole, fraction = format(number.normalize(), "f").split(".", 1)
            return f"{int(whole):,}.{fraction}"
        if number == number.to_integral():
            return f"{int(number):,}".replace(",", ".")
        return format(number.normalize(), "f").replace(".", ",")

    @staticmethod
    def _number(value: int) -> str:
        return f"{int(value):,}".replace(",", ".")

    @staticmethod
    def _percent(value: float) -> str:
        return f"{value:.1f}".replace(".", ",") + "%"
