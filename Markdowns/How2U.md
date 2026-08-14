# Hướng dẫn sử dụng hệ thống trợ lý AI Meibook (VLLM-PD)

Tài liệu này tóm tắt cách vận hành và khai thác các tính năng chính của **Meibook**, hệ thống chatbot trợ lý AI nội bộ cho MKAC.

> Lưu ý Production: WMS đã có code trong repository nhưng chỉ hiển thị khi backend xác nhận snapshot WMS khả dụng. Nếu Production chưa import snapshot và chưa bật cấu hình WMS, tab WMS sẽ được ẩn và mọi truy vấn WMS phải fail-closed.

---

## 1. Đăng nhập và xác thực người dùng

1. **Truy cập ứng dụng**
   - Production: `http://localhost:8001`
   - Dev: `http://localhost:8002`

2. **Nhập mã nhân viên (`employee_id`)**
   - Nhân viên nội bộ: nhập mã nhân viên hợp lệ theo danh bạ đã import.
   - Tài khoản demo/guest: nhập `000000` nếu cần kiểm tra luồng khách.

3. **Chọn ngôn ngữ UI**
   - `VI`: tiếng Việt.
   - `JA`: tiếng Nhật.

---

## 2. Các chế độ hoạt động

Thanh chọn mode nằm trong giao diện chat. Production mặc định có ba luồng chính:

### 2.1. Hành chính nhân sự (`mode=mkac`)

- Tra cứu nội quy lao động, quy định công tác, bảo hiểm, chế độ đãi ngộ, giờ làm thêm, khám sức khỏe và thông tin nhân sự MKAC.
- Câu hỏi nhân sự có cấu trúc được ưu tiên trả lời từ SQLite, ví dụ headcount, danh sách phòng ban, trưởng/phó phòng.
- Câu hỏi tài liệu dùng RAG trên nguồn MKAC và hiển thị trích dẫn khi có nguồn phù hợp.
- Không có nguồn đủ mạnh thì hệ thống phải nói rõ không tìm thấy thông tin, không tự suy diễn chính sách.

### 2.2. Sản xuất MES (`mode=mes`)

- Tra cứu Lot, mã hàng, lỗi sản xuất và thống kê lỗi từ snapshot MES đã import.
- Ưu tiên route deterministic/SQL template để giữ số liệu kiểm chứng được.
- SQL Agent chỉ là fallback có guardrail: read-only, view allowlist, row limit và timeout.
- Không dùng MES để suy đoán tồn kho WMS khi WMS chưa được bật riêng.

### 2.3. Nghiên cứu tài liệu (`mode=research`)

- Tra cứu chuyên sâu tài liệu nội bộ theo chủ đề Research.
- Người dùng chọn topic hoặc phạm vi tài liệu trước khi hỏi.
- Câu trả lời phải bám nguồn, có citation và có thể mở preview trang nguồn nếu tài liệu đã được xử lý ảnh trang.

### 2.4. WMS (`mode=wms`, tùy cấu hình)

- WMS chỉ xuất hiện khi backend báo `wmsStatus.available=true`.
- Khi chưa có snapshot Production hoặc `MES_WMS_DATABASE_ENABLED=false`, UI sẽ ẩn tab WMS và backend trả lời fail-closed.
- Khi được bật, WMS phải dùng snapshot SQLite riêng `mes_wms.sqlite`, không reuse MES database hoặc MES SQL Agent.

---

## 3. Báo cáo cấp điều hành

Một số câu hỏi dạng báo cáo có thể kích hoạt report artifact thay vì câu trả lời text thuần.

### 3.1. Báo cáo HR

Ví dụ:

- “Lập báo cáo nhân sự”
- “Tạo báo cáo cơ cấu headcount các phòng ban”
- “人事レポートを作成してください。”

Kết quả thường gồm tổng hợp headcount, phân bổ phòng ban và thẻ artifact HTML trong giao diện.

### 3.2. Báo cáo MES

Ví dụ:

- “Lập báo cáo tổng hợp lỗi MES”
- “Lập báo cáo về 5 lỗi nhiều nhất”
- “MES不良の概要レポートを作成してください。”

Kết quả phải dựa trên snapshot MES và nêu rõ phạm vi dữ liệu, giới hạn hoặc lỗi nếu có.

### 3.3. Báo cáo WMS

Ví dụ:

- “Lập báo cáo tổng quan tồn kho WMS”
- “Báo cáo tình hình tồn kho các công đoạn WMS”
- “工程在庫レポートを作成してください。”

Chỉ dùng khi WMS đã được bật và snapshot tương thích. Nếu WMS đang disabled/unavailable, hệ thống phải từ chối rõ ràng thay vì suy đoán tồn kho từ MES hoặc LLM.

---

## 4. Tiện ích giao diện

### 4.1. Lịch sử prompt

- Trong ô nhập câu hỏi, dùng `ArrowUp` để gọi lại câu hỏi trước đó.
- Dùng `ArrowDown` để đi về câu hỏi mới hơn hoặc khôi phục nội dung đang gõ.

### 4.2. Gửi và dừng câu hỏi

- `Enter`: gửi câu hỏi.
- `Shift + Enter`: xuống dòng.
- Nút Stop: hủy request streaming đang chạy nếu cần.

### 4.3. Ngôn ngữ và giao diện

- Nút `VI | JA`: đổi ngôn ngữ hiển thị và ngôn ngữ phản hồi.
- Nút sáng/tối: đổi theme light/dark.

### 4.4. Citation và preview nguồn

- Với câu trả lời có nguồn, bấm citation/source để xem thông tin nguồn.
- Nếu có preview trang, hệ thống mở ảnh trang đã xử lý từ tài liệu gốc.

---

## 5. Câu hỏi mẫu nhanh

| Phân hệ | Mẫu câu hỏi | Mục đích |
|---|---|---|
| HR | “Phòng ban nào đông nhân sự nhất?” | Tra cứu headcount theo phòng ban |
| HR | “Lập báo cáo nhân sự” | Tạo báo cáo HR cấp điều hành |
| MES | “Thông tin chi tiết của Lot 000866-05-000” | Tra cứu thông tin Lot |
| MES | “Lập báo cáo tổng hợp lỗi MES” | Tạo báo cáo lỗi sản xuất |
| Research | “Quy định về bảo mật thông tin trong IT” | Tra cứu tài liệu theo topic |
| WMS | “Tồn kho công đoạn NCDRIL_FAC1 trong WMS” | Chỉ dùng khi WMS đã được bật và snapshot khả dụng |

---

## 6. Khi hệ thống không trả lời được

- HR/MKAC: nếu không có nguồn nội bộ phù hợp, hệ thống phải nói không tìm thấy thông tin.
- MES: nếu intent hoặc dữ liệu snapshot không hỗ trợ, hệ thống không được bịa số liệu.
- Research: nếu topic không có evidence, câu trả lời phải nêu thiếu nguồn.
- WMS: nếu disabled/unavailable, hệ thống phải fail-closed và không suy đoán từ nguồn khác.
