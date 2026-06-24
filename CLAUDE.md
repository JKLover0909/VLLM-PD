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