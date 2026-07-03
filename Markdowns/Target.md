Bạn là một Senior Full-stack Architect kiêm UX/UI Product Designer. Hãy rà soát toàn bộ source code của hệ thống này như một buổi technical audit chuyên nghiệp.

Bối cảnh hệ thống:
Đây là một hệ thống chatbot nội bộ giống ChatGPT, dùng để hỏi đáp về nhân sự và hệ thống MES trong doanh nghiệp. Người dùng chính có thể là nhân viên, quản lý, HR, kỹ thuật sản xuất hoặc quản trị hệ thống. Hệ thống cần dễ dùng, phản hồi nhanh, bảo mật dữ liệu nội bộ và có khả năng mở rộng.

Nhiệm vụ của bạn:

1. Rà soát UX/UI

* Đánh giá giao diện hiện tại có giống một chatbot chuyên nghiệp, hiện đại, dễ dùng hay chưa.
* Kiểm tra luồng sử dụng: đăng nhập, chọn chủ đề hỏi đáp, nhập câu hỏi, xem câu trả lời, xem nguồn tham chiếu, lịch sử hội thoại, phản hồi đúng/sai.
* Đề xuất cải thiện layout, màu sắc, typography, spacing, trạng thái loading, empty state, error state.
* Đề xuất cách hiển thị câu trả lời có cấu trúc: bullet, bảng, citation, file/link nguồn, mức độ tin cậy.
* Đề xuất giao diện riêng cho từng nhóm use case: hỏi đáp nhân sự, hỏi đáp MES, tra cứu quy trình, tra cứu tài liệu, cảnh báo lỗi.
* Kiểm tra responsive trên desktop, tablet, mobile.
* Chỉ ra các điểm gây khó hiểu, thừa thao tác hoặc chưa thân thiện với người dùng nội bộ.

2. Rà soát frontend code

* Kiểm tra cấu trúc component, state management, routing, API calling, error handling.
* Đánh giá khả năng tái sử dụng component.
* Đề xuất cách tổ chức lại component nếu hiện tại đang rối.
* Kiểm tra performance phía frontend: re-render không cần thiết, bundle size, lazy loading, caching, debounce input, streaming response.
* Đề xuất cải thiện accessibility nếu cần.

3. Rà soát backend architecture

* Phân tích kiến trúc backend hiện tại: API layer, service layer, database, authentication, authorization, logging, vector search/RAG, LLM integration.
* Kiểm tra backend đã tách module rõ chưa: user management, chat, document ingestion, retrieval, HR domain, MES domain, audit log.
* Đánh giá khả năng mở rộng khi nhiều người dùng cùng hỏi.
* Kiểm tra các điểm nghẽn: truy vấn database, gọi LLM, embedding, vector search, xử lý file, session memory.
* Đề xuất refactor backend theo kiến trúc rõ ràng, dễ maintain.

4. Tối ưu hệ thống hỏi đáp/RAG

* Kiểm tra pipeline từ upload tài liệu → chunking → embedding → lưu vector DB → retrieval → reranking → prompt → sinh câu trả lời.
* Đánh giá chất lượng chunking, metadata, phân quyền theo tài liệu, lọc theo phòng ban/hệ thống.
* Đề xuất cách cải thiện retrieval để trả lời chính xác hơn cho tài liệu HR và MES.
* Đề xuất cơ chế citation để người dùng biết câu trả lời lấy từ đâu.
* Đề xuất cách xử lý khi không tìm thấy thông tin thay vì trả lời bừa.
* Đề xuất prompt system phù hợp cho chatbot nội bộ doanh nghiệp.
* Đề xuất cơ chế feedback người dùng để cải thiện chất lượng câu trả lời.

5. Bảo mật và phân quyền

* Kiểm tra authentication, authorization, role-based access control.
* Đánh giá rủi ro lộ dữ liệu nhân sự, dữ liệu sản xuất, tài liệu nội bộ.
* Đề xuất phân quyền theo vai trò: nhân viên, HR, quản lý, kỹ thuật MES, admin.
* Kiểm tra API có bị lộ thông tin nhạy cảm không.
* Đề xuất audit log cho câu hỏi, câu trả lời, tài liệu được truy xuất.
* Đề xuất chống prompt injection, data leakage, unauthorized document access.

6. Tối ưu hiệu năng backend

* Kiểm tra tốc độ phản hồi API, streaming response, timeout, retry, queue/background jobs.
* Đề xuất caching cho các truy vấn phổ biến.
* Đề xuất tối ưu vector search, database index, connection pooling.
* Đề xuất async processing cho embedding, upload tài liệu, phân tích file.
* Đề xuất cơ chế rate limit, circuit breaker, monitoring.
* Đề xuất cách giảm latency khi gọi LLM.

7. DevOps và vận hành

* Kiểm tra file .env, config, Docker, deployment, logging, monitoring.
* Đề xuất cấu trúc môi trường dev/staging/production.
* Đề xuất health check, metrics, error tracking.
* Đề xuất backup database/vector DB.
* Đề xuất CI/CD nếu phù hợp.

8. Output mong muốn
   Hãy trả lời theo cấu trúc sau:

A. Tổng quan hệ thống hiện tại

* Mô tả ngắn hệ thống đang được tổ chức như thế nào.
* Điểm mạnh hiện tại.
* Vấn đề lớn nhất cần ưu tiên.

B. Danh sách vấn đề phát hiện
Với mỗi vấn đề, trình bày theo format:

* Vấn đề:
* Mức độ nghiêm trọng: Critical / High / Medium / Low
* Vị trí file/thư mục liên quan:
* Tác động:
* Cách cải thiện:
* Gợi ý code hoặc kiến trúc nếu cần:

C. Đề xuất cải thiện UX/UI

* Các cải thiện nhanh có thể làm ngay.
* Các cải thiện quan trọng cho sản phẩm nội bộ.
* Gợi ý layout/trải nghiệm chatbot tốt hơn.

D. Đề xuất cải thiện backend

* Refactor architecture.
* Tối ưu API.
* Tối ưu RAG.
* Tối ưu database/vector DB.
* Tối ưu bảo mật.

E. Roadmap ưu tiên triển khai
Chia thành 3 giai đoạn:

* Giai đoạn 1: Sửa lỗi nghiêm trọng và cải thiện nhanh.
* Giai đoạn 2: Tối ưu kiến trúc, hiệu năng, UX chính.
* Giai đoạn 3: Nâng cấp nâng cao như monitoring, feedback loop, analytics, multi-agent, advanced permission.

F. Checklist hành động
Tạo checklist rõ ràng để developer có thể làm theo từng bước.

Yêu cầu quan trọng:

* Hãy đọc kỹ toàn bộ repository trước khi kết luận.
* Không chỉ nhận xét chung chung, phải chỉ rõ file, module, component hoặc API nào cần sửa.
* Nếu có thể, hãy đề xuất code cụ thể.
* Ưu tiên các cải thiện thực tế, có thể triển khai trong sản phẩm nội bộ doanh nghiệp.
* Không thay đổi code ngay lập tức nếu chưa được yêu cầu. Trước tiên hãy audit, phân tích và đề xuất kế hoạch cải thiện.
