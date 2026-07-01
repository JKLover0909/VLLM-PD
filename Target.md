Ưu Tiên Cao
Tách rõ 3 chế độ hơn nữa
Hiện đã có Hành chính nhân sự, MES, Nghiên cứu, nhưng backend vẫn nên có routing rõ ràng hơn theo từng mode. Đặc biệt MES nên đi qua SQL/MES agent riêng, không lẫn với RAG tài liệu.

Thêm memory ngắn hạn chuẩn
Hiện đã bắt đầu có conversation_context, nhưng nên chuẩn hóa để câu như “gửi thông tin này qua email” hiểu được “thông tin này” là câu trả lời trước đó. Đây rất quan trọng cho trải nghiệm demo.

Cải thiện SQL Agent cho MES
Nên cho model nhìn schema MES dạng ngắn gọn, có whitelist bảng/cột, rồi sinh SQL read-only. Khi đó hỏi kiểu:
“Trong Lot lỗi nhiều nhất, 3 lỗi phổ biến nhất là gì?”
hệ thống tự suy luận query thay vì phải chuẩn bị từng intent cố định.

Lưu và chọn tài liệu nghiên cứu mẫu
Thay vì chỉ một session demo, nên có danh sách “Tài liệu mẫu” để chọn. Khi bấm vào thì clone hoặc bind vào session research hiện tại. Demo sẽ mượt hơn nhiều.

Ưu Tiên Trung Bình
5. Hiển thị trạng thái nguồn tốt hơn
   Phần Độ tương đồng nên có tooltip giải thích ngắn: “Mức độ liên quan giữa câu hỏi và đoạn trích”. Người dùng sẽ đỡ hiểu nhầm là độ chính xác.
Source preview nâng cấp
Nếu xem PDF preview được rồi, nên thêm highlight đoạn trích trên ảnh hoặc ít nhất scroll/đánh dấu vùng trang liên quan. Đây là điểm tạo cảm giác hệ thống “có bằng chứng”.

Quản lý lỗi/timeout thân thiện
Với MES API, Gmail API, LiteLLM, Qdrant, nên có message riêng thay vì lỗi chung. Ví dụ: “MES hiện không phản hồi”, “Không gửi được email do Gmail token hết hạn”.

Health dashboard nhỏ
Một panel admin hoặc endpoint readiness kiểm tra: FastAPI, Qdrant, LiteLLM, MES DB, Gmail, model cloud/local. Trước demo chạy một phát biết cái gì chết.

Ưu Tiên Sau
9. Chuẩn hóa i18n
   Hiện đã có VN/JP ở frontend, nhưng nên đưa toàn bộ label vào file riêng như i18n.js để dễ bảo trì, tránh main.jsx ngày càng phình.
Bảo mật
   Nếu dùng ngoài demo: cần auth thật cho session, thu hẹp CORS, bảo vệ MES/Gmail action bằng xác nhận trước khi gửi, và không để các API nội bộ public.
Nếu chọn một việc làm tiếp ngay, mình đề xuất: hoàn thiện memory ngắn hạn + Gmail action theo ngữ cảnh. Vì nó làm demo rất tự nhiên: hỏi MES xong nói “gửi kết quả này cho A”, hệ thống hiểu và gửi đúng nội dung.

###### 

Giảm vai trò của LLM trong MES
MES đang có nhiều câu có thể trả lời deterministic từ SQLite/API, không cần gọi Qwen3 để “diễn đạt lại”. Với demo, nên ưu tiên:
query SQL cố định
format câu trả lời bằng template
chỉ gọi LLM khi câu hỏi thật sự tự do/phức tạp
Đây là hướng giảm latency rõ nhất. MES có thể từ ~30s xuống vài trăm ms đến vài giây.

Dùng Qwen Coder cho SQL/MES, không dùng Qwen3 chat
Test vừa rồi:
local-qwen-coder: ~0.4-0.7s
Qwen3 chat: ~2.7s cho prompt ngắn, nhưng qua app lên ~30s do pipeline + output dài
Vì vậy MES SQL Agent nên dùng Qwen Coder cho planning, còn câu trả lời nên template hóa.

Tắt hoặc giảm dịch tự động khi không cần
Tiếng Nhật hiện phải đi qua lớp translate, nên request JP chậm hơn. Có thể tối ưu bằng:
cache bản dịch prompt/câu hỏi phổ biến
chỉ dịch câu hỏi sang tiếng Việt, còn output MES dùng template Nhật trực tiếp
với MES, tạo sẵn formatter vi/ja thay vì gọi translation model

Giảm max_tokens cho Qwen3
Nếu câu trả lời MKAC/MES chỉ cần ngắn, không nên để token budget rộng. Ví dụ:
MKAC: 512 hoặc 768
MES: 256 hoặc 384
chỉ tăng khi research hoặc câu hỏi dài
Điều này giảm nguy cơ model suy luận lan man và giảm thời gian.

Tăng strict prompt + hậu xử lý
Qwen3 hay sinh reasoning/echo prompt. Mình đã thêm hậu xử lý, nhưng tốt hơn là prompt cũng nên ép:
“Chỉ trả lời kết luận cuối cùng”
“Không giải thích quá trình đọc tài liệu”
“Tối đa 3 câu”
Cái này không làm nhanh bằng deterministic, nhưng giúp giảm output dài.

Streaming thật thay vì gom rồi trả
Hiện sau khi thêm clean reasoning, streaming RAG đang gom output rồi làm sạch, nên người dùng phải đợi xong mới thấy chữ. Có thể cải thiện UX bằng:
stream trạng thái trước: “Đang tìm tài liệu…”
“Đang gọi model local…”
stream sau khi lọc token nếu thiết kế bộ lọc incremental
Không làm model nhanh hơn, nhưng cảm giác chờ tốt hơn.

Cache câu hỏi phổ biến
Với demo, các câu kiểu:
“MKAC có bao nhiêu phòng ban?”
“Lot nào nhiều lỗi nhất?”
“3 lỗi nhiều nhất trong lot nhiều lỗi nhất?”
nên cache kết quả trong RAM/SQLite. Câu trả lời gần như tức thì.

Dùng model nhỏ hơn cho tác vụ đơn giản
Có thể thêm một model local nhỏ hơn để:
dịch ngắn
phân loại intent
rewrite câu hỏi
format câu trả lời
Qwen3-14B dùng cho mọi thứ hơi nặng.


MES chuyển tối đa sang deterministic/template, không gọi LLM nếu query đã rõ.
Cache các câu MES/MKAC phổ biến.
Giảm max_tokens theo mode.
Tạo formatter tiếng Nhật trực tiếp cho MES để bỏ bước dịch.
Tối ưu streaming trạng thái để người dùng thấy hệ thống đang làm gì.