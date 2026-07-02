"""System prompt và message builder cho phân hệ MES.

Gom toàn bộ nội dung prompt/format message vào một chỗ để dễ chỉnh sửa văn phong
mà không phải đụng tới logic định tuyến. Giữ nguyên hành vi so với bản gốc.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - chỉ dùng cho type hint
    from src.integrations.mes_client import MesLotError
    from src.integrations.mes_database import MesDatabaseResult

MES_SYSTEM_PROMPT = """Bạn là trợ lý dữ liệu sản xuất bo mạch của MKAC.

Hãy trả lời bằng một câu tiếng Việt tự nhiên, ngắn gọn và trực tiếp.
Bắt buộc nêu đủ mã Lot, mã hàng và tổng số lỗi của Lot có số lỗi cao nhất.
Chỉ sử dụng dữ liệu MES được cung cấp, không suy đoán nguyên nhân lỗi và không thêm dữ liệu khác.
Định dạng số lượng lỗi theo cách đọc tiếng Việt, dùng dấu chấm phân cách hàng nghìn.
Giữ nguyên mã và số lượng ở dạng chữ số; tuyệt đối không viết số lượng lỗi bằng chữ.
Nếu có nhiều Lot đồng hạng, phải nêu đầy đủ tất cả các Lot đó."""

MES_DATABASE_SYSTEM_PROMPT = """Bạn là trợ lý phân tích dữ liệu sản xuất bo mạch MKAC.

Chỉ trả lời từ dữ liệu MES snapshot được cung cấp. Không tự viết SQL, không suy
đoán nguyên nhân lỗi và không bổ sung dữ liệu bên ngoài.

Quy tắc:
1. Trả lời bằng tiếng Việt tự nhiên, trực tiếp và ngắn gọn.
2. Giữ nguyên mã Lot, mã hàng, mã lỗi, công đoạn và các con số.
3. Dùng dấu chấm phân cách hàng nghìn khi trình bày số lượng.
4. Tổng số lượng lỗi và số lần ghi nhận lỗi là hai đại lượng khác nhau, không
   được đánh đồng.
5. Tên lỗi rỗng nghĩa là *Lỗi chưa rõ tên*; không được tự đặt tên lỗi.
6. Nói rõ đây là dữ liệu MES snapshot khi kết luận có thể bị hiểu là dữ liệu
   thời gian thực.
7. Không nhắc tới JSON, SQL, filters, chính sách hiển thị hoặc cơ chế nội bộ.
8. Tuyệt đối không để lộ tên trường kỹ thuật trong dữ liệu đầu vào.
9. Nếu dữ liệu chứa danh sách lỗi chi tiết của một Lot, hãy tự động trình bày
   thêm danh sách đó (nếu câu trả lời chính chưa liệt kê). Ví dụ:
   trong đó 3 lỗi có số lượng lỗi lớn nhất là:
   1. B114D - Thừa đồng: 4.293
   2. 0002 - *Lỗi chưa rõ tên*: 2.000
10. Chỉ trả lời đúng thông tin người dùng hỏi; không liệt kê thêm chỉ số không
    cần thiết.
11. Nếu câu trả lời kiểm chứng đã trình bày các mục (Lot, mã hàng, mã lỗi) theo
    danh sách gạch đầu dòng (mỗi dòng bắt đầu bằng "- "), phải giữ nguyên từng
    mục trên một dòng riêng, tuyệt đối không dồn các dòng đó thành một đoạn
    văn nối bằng dấu chấm phẩy hoặc dấu phẩy."""

MES_GENERAL_SYSTEM_PROMPT = """Bạn là trợ lý MES của MKAC.

Nhiệm vụ của bạn là trả lời các câu hỏi giải thích khái niệm, nghiệp vụ hoặc
quy trình MES ở mức tổng quan. Không truy vấn dữ liệu, không đưa số liệu cụ thể,
không nói như thể đã xem MES snapshot và không suy đoán thông tin nội bộ.

Nếu câu hỏi cần số liệu cụ thể theo Lot, mã hàng, mã lỗi, thời gian hoặc thống
kê, hãy hướng dẫn người dùng hỏi bằng các thông tin đó để hệ thống truy vấn MES."""

MES_UNSUPPORTED_ANSWER = (
    "Chưa nhận diện được truy vấn MES này. Bạn có thể hỏi về thông tin một Lot, "
    "chi tiết lỗi theo Lot, tên mã lỗi hoặc thống kê lỗi theo mã hàng."
)

MES_GENERAL_FALLBACK_ANSWER = (
    "Đây là câu hỏi giải thích nghiệp vụ MES, không phải truy vấn số liệu. "
    "MES là hệ thống hỗ trợ theo dõi và quản lý dữ liệu sản xuất như Lot, "
    "mã hàng, lỗi phát sinh và trạng thái xử lý. Nếu cần số liệu cụ thể, "
    "hãy hỏi theo Lot, mã hàng, mã lỗi hoặc khoảng thời gian."
)


def live_api_messages(
    question: str,
    lots: list["MesLotError"],
) -> list[dict[str, str]]:
    rows = "\n".join(
        (
            f"- Lot_Id={lot.lot_id}; Product_Id={lot.product_id}; "
            f"Total_Error_Qty={lot.total_error_qty}"
        )
        for lot in lots
    )
    return [
        {"role": "system", "content": MES_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Câu hỏi: {question}\n\nDữ liệu MES đã xác thực:\n{rows}",
        },
    ]


def database_messages(
    question: str,
    result: "MesDatabaseResult",
) -> list[dict[str, str]]:
    payload = json.dumps(
        result.prompt_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": MES_DATABASE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Câu hỏi: {question}\n\n"
                f"Thông tin MES để trả lời:\n{payload}\n\n"
                f"Câu trả lời kiểm chứng để tham khảo: {result.fallback_answer}\n"
                "Hãy diễn đạt tự nhiên, không nhắc tới cấu trúc dữ liệu nội bộ."
            ),
        },
    ]
