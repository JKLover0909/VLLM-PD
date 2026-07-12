---
name: datetime-nlp-resolver
description: "Xử lý và chuẩn hóa các mốc thời gian tự nhiên (tiếng Việt và tiếng Nhật) trong câu hỏi thành dạng ISO 8601 UTC/Local để sử dụng cho Calendar API."
---

# Date-Time NLP Resolver (VI/JA)

## 1. Nguyên lý hoạt động
Khi người dùng yêu cầu đặt lịch họp bằng ngôn ngữ tự nhiên (ví dụ: "9 giờ sáng mai", "chiều thứ Sáu tuần tới"), mô hình LLM có xu hướng tự sinh ngày/giờ không chính xác hoặc không đồng bộ với thời gian thực của server. 

Quy trình xử lý an toàn:
1. **LLM trích xuất cấu trúc thô:** LLM nhận diện và trả về thông tin thời gian dưới dạng JSON có cấu trúc (chứa các mốc tương đối như ngày, giờ, khoảng thời gian).
2. **Backend tính toán logic thời gian:** Backend lấy thời gian thực của server làm mốc gốc (`now`), kết hợp múi giờ của hệ thống (`Asia/Ho_Chi_Minh`) để tính ra chính xác ngày và giờ theo chuẩn ISO 8601.

## 2. Quy tắc chuẩn hóa thời gian (Time Normalization Rules)

### Xử lý ngày tương đối:
*   "hôm nay" (kyou/今日): Ngày hiện tại của hệ thống.
*   "ngày mai" (ashita/明日): Ngày hiện tại + 1.
*   "ngày kia" (asatte/明後日): Ngày hiện tại + 2.
*   "thứ [X]" (kinyoubi/金曜日): Tìm ngày thứ X tiếp theo trong tuần. Nếu ngày hiện tại trùng với thứ X nhưng thời gian yêu cầu đã trôi qua, tự động cộng thêm 7 ngày (chuyển sang tuần sau).

### Xử lý giờ tương đối:
*   "sáng" (gozen/午前): Mặc định từ 8:00 đến 11:30. Nếu chỉ nói "9 giờ sáng" -> 09:00.
*   "chiều" (gogo/午後): Cộng thêm 12 giờ nếu giờ từ 1:00 đến 6:00. Ví dụ: "2 giờ chiều" -> 14:00.
*   "cuối giờ chiều" (yugata/夕方): Mặc định đặt mốc 17:00.
*   Nếu không nêu rõ giờ cụ thể (chỉ nói "ngày mai họp phòng ban"), mặc định lấy mốc 09:00 hoặc 14:00 và yêu cầu xác nhận.

## 3. Quản lý múi giờ và thời lượng (Timezone & Duration)
*   Múi giờ mặc định cho hệ thống là `Asia/Ho_Chi_Minh` (UTC+7).
*   Thời lượng mặc định cho cuộc họp là 60 phút nếu người dùng không chỉ định cụ thể.
*   Trước khi ghi nhận lịch họp, backend phải kiểm tra điều kiện: `start_time > current_time` (không cho phép đặt lịch họp trong quá khứ). Nếu không hợp lệ, trả về thông báo lỗi và yêu cầu điều chỉnh lại giờ họp.
