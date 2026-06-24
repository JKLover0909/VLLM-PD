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