"""Client for querying MKAC PCB production data from the MES API."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class MesApiError(RuntimeError):
    """Raised when MES data cannot be retrieved or validated."""


@dataclass(frozen=True)
class MesLotError:
    lot_id: str
    product_id: str
    total_error_qty: int


class MesClient:
    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        timeout: float = 15.0,
        verify: bool | str = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_url = api_url
        self.token = token
        self.timeout = timeout
        self.verify = verify
        self.transport = transport

    @classmethod
    def from_env(cls) -> "MesClient | None":
        api_url = os.getenv("MES_API_URL", "").strip()
        token = os.getenv("MES_API_TOKEN", "").strip()
        if not api_url or not token:
            return None

        ca_cert = os.getenv("MES_CA_CERT", "").strip()
        if ca_cert:
            verify: bool | str = str(Path(ca_cert))
        else:
            verify = os.getenv("MES_VERIFY_TLS", "true").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        return cls(
            api_url=api_url,
            token=token,
            timeout=float(os.getenv("MES_API_TIMEOUT", "15")),
            verify=verify,
        )

    async def get_lots_with_highest_error(self) -> list[MesLotError]:
        rows = await self._fetch_lot_errors()
        highest = max(row.total_error_qty for row in rows)
        return [row for row in rows if row.total_error_qty == highest]

    async def _fetch_lot_errors(self) -> list[MesLotError]:
        payload = {
            "ServiceName": "mes_data",
            "ActionName": "DEMO_GET_TOTAL_ERROR",
            "Condition": {"Schema_Data": os.getenv("MES_API_SCHEMA_DATA", "MES_DATA")},
        }
        headers = {"Authorization": f"Bearer {self.token}"}

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise MesApiError("MES API không phản hồi trong thời gian cho phép.") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise MesApiError(f"MES API trả về HTTP {status}.") from exc
        except httpx.HTTPError as exc:
            raise MesApiError("Không thể kết nối tới MES API.") from exc

        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise MesApiError("MES API trả về dữ liệu không phải JSON.") from exc

        if body.get("code") != 200 or not isinstance(body.get("data"), list):
            raise MesApiError("MES API trả về cấu trúc dữ liệu không hợp lệ.")

        rows = [self._parse_row(item) for item in body["data"]]
        if not rows:
            raise MesApiError("MES API không trả về dữ liệu Lot.")
        return rows

    @staticmethod
    def _parse_row(item: Any) -> MesLotError:
        if not isinstance(item, dict):
            raise MesApiError("MES API có bản ghi Lot không hợp lệ.")

        lot_id = str(item.get("Lot_Id", "")).strip()
        product_id = str(item.get("Product_Id", "")).strip()
        raw_quantity = str(item.get("Total_Error_Qty", "")).strip()
        if not lot_id or not product_id or not raw_quantity:
            raise MesApiError("MES API thiếu thông tin Lot, mã hàng hoặc số lỗi.")

        try:
            quantity = int(raw_quantity.replace(",", ""))
        except ValueError as exc:
            raise MesApiError("MES API trả về số lượng lỗi không hợp lệ.") from exc

        return MesLotError(
            lot_id=lot_id,
            product_id=product_id,
            total_error_qty=quantity,
        )
