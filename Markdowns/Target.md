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

Phân tích hiệu năng hệ thống hỏi đáp — Meibook
Lưu ý đo lường: môi trường không có tool load-test/profiling, nên phần "cách đo" là hướng dẫn để bạn chạy, không phải số liệu tôi tự đo.

Hiện trạng grounded (tóm tắt kỹ thuật)
Thành phần	Cấu hình thực tế
Embedder	BAAI/bge-m3 1024d, local GPU, 1 threading.Lock toàn cục (embedder.py:20)
Chunking	1400 ký tự/chunk, overlap 220 ký tự, char-based (không token-based) (parser.py:53)
Retrieval	Qdrant dense cosine, top_k=5, threshold=0.3, không rerank, không hybrid/BM25 (vector_store.py:199)
Ngữ cảnh hội thoại	Không dùng khi trả lời (stateless)
LLM calls/câu	MES tất định 0–1 · MES SQL agent 2 · RAG 1 · JA +2 (dịch) tuần tự
Cache	Full-response, MES snapshot-versioned TTL 1 ngày; không cache embedding/retrieval
Monitoring	Không có metrics/latency/token logging (chỉ time.monotonic cho floor + TTL)
Floor độ trễ	MIN_QUERY_RESPONSE_SECONDS=2.0 (chủ đích)
Đã sửa trong phiên (không đề xuất lại): streaming thật, hybrid template gate cắt LLM lần 2, MES snapshot cache versioned, dịch source song song.

Phân tích theo từng chiều
1. Tốc độ phản hồi (latency)
[P0] Event loop bị chặn bởi SQLite đồng bộ — authorize_query/employee_directory gọi SQLite không qua asyncio.to_thread (main.py:1071, employee_directory.py:62).
Nguyên nhân: hàm sync chạy thẳng trong coroutine.
Tác động: 1 truy vấn HR chậm đóng băng toàn bộ request đồng thời của người khác.
Đo: bắn 20 request song song, đo p95; bật asyncio debug (PYTHONASYNCIODEBUG=1) xem "slow callback".
Giải pháp: bọc cụm directory trong 1 asyncio.to_thread. Ưu tiên P0 · Độ khó Thấp.
[P1] Đường JA: 2 LLM dịch tuần tự trên critical path (main.py:1071 localize + translate answer).
Tác động: câu JA chịu 3 round-trip LLM nối tiếp.
Giải pháp: cache dịch (câu gợi ý cố định → hit 100%); với suggestion JA đã biết bản VI (cùng file quick_answers.json) → bỏ qua bước dịch câu hỏi bằng cách gửi kèm bản VI. P1 · TB.
[P1] authorize_query chạy TRƯỚC cache-check → cache hit vẫn trả giá blocking DB ([main.py:1040-1075]). P1 · Thấp.
2. Độ chính xác câu trả lời (accuracy)
Điểm tốt: MES có kiểm chứng đầu ra (sql_answer_matches_result, database_answer_has_required_terms) — chống bịa số. SQL agent sandbox chắc.
[P1] Không có eval harness / golden set.
Tác động: mọi thay đổi prompt/model/retrieval không đo được hồi quy chất lượng.
Đo: dựng bộ ~50–100 câu hỏi vàng (MES + HR) với đáp án kỳ vọng; chạy tự động, chấm bằng exact-match (số liệu MES) + LLM-judge (HR).
Giải pháp: script eval + CI gate. P1 · TB.
[P2] RAG không kiểm tra groundedness/faithfulness — câu trả lời tài liệu chỉ dựa retrieval, không verify "câu trả lời có nằm trong nguồn không". P2 · TB.
3. Khả năng hiểu ngữ cảnh (context understanding) ⚠️
[P0/P1] Trả lời stateless — bỏ hoàn toàn lịch sử hội thoại. conversation_context gửi lên nhưng rag_pipeline.query() không nhận; chỉ dùng cho email.
Tác động: câu nối tiếp ("còn mã kia thì sao?", "liệt kê thêm") không hiểu → người dùng phải lặp lại đầy đủ mỗi lần. Đây là thiếu sót AI-Q&A lớn nhất.
Đo: hỏi 1 câu, rồi hỏi follow-up tham chiếu; xem có đúng không.
Giải pháp: (a) truyền conversation_context vào messages LLM (RAG); (b) với MES, thêm bước query rewriting (LLM viết lại câu follow-up thành câu độc lập trước khi vào intent/SQL). P0 nếu người dùng cần multi-turn · TB.
4. Chất lượng truy xuất (retrieval)
[P1] Dense-only, không rerank, top_k=5, threshold=0.3.
Nguyên nhân: pipeline retrieval tối giản.
Tác động: câu hỏi dùng từ khóa/mã cụ thể (mã hàng, số hiệu) mà embedding không bắt tốt → miss; threshold 0.3 có thể lọc nhầm hoặc để lọt nhiễu; top_k=5 cố định không co giãn theo độ khó.
Đo: recall@k trên golden set; log điểm số các hit.
Giải pháp: thêm hybrid (BM25/keyword + dense) hoặc rerank (bge-reranker) top_k=20→5; cân nhắc query expansion. P1 · TB–Cao.
[P2] Chunk char-based 1400 ký tự — cắt theo dòng/ký tự, không theo token/ngữ nghĩa → chunk có thể cắt giữa ý. P2 · TB.
5. Xử lý câu hỏi mơ hồ (ambiguity)
[P1] Không có bước làm rõ / disambiguation. MES: regex miss → SQL agent → nếu fail → MES_UNSUPPORTED_ANSWER cụt. RAG: không đủ chunk qua threshold → câu chung chung/rỗng.
Tác động: người dùng gặp ngõ cụt thay vì được gợi ý cách hỏi lại.
Giải pháp: khi độ tin cậy thấp (SQL agent can_answer=false, hoặc retrieval score thấp), trả câu hỏi làm rõ hoặc gợi ý các câu hỏi gần đúng thay vì lỗi cụt. P1 · TB.
6. Độ ổn định đa người dùng (concurrency)
[P0] (trùng #1) blocking event loop = nghẽn toàn cục dưới tải.
[P1] Embedder 1 lock toàn cục (embedder.py) — upload tài liệu (embed batch lớn) làm mọi query của người khác xếp hàng chờ. P1 · TB–Cao (hàng đợi inference riêng).
[P1] Cache & rate-limit in-memory per-process — chạy nhiều worker/replica sẽ không chia sẻ → hit rate giảm, rate-limit hớ. Giải pháp: Redis dùng chung khi scale. P1 · TB.
7. Chi phí token/API (cost)
Điểm tốt: max_tokens đã giới hạn theo ngữ cảnh (mes_config.py); hybrid gate cắt LLM lần 2; model local qua LiteLLM (chi phí thấp).
[P2] Không hạch toán token — không log usage/câu → không biết đường nào tốn nhất.
Giải pháp: log response.usage (prompt/completion tokens) theo route + model. P2 · Thấp.
[P2] JA tốn gấp đôi (2 lần dịch) — cache dịch (#1) giảm trực tiếp. P2 · Thấp.
8. Caching
Điểm tốt: full-response cache + MES snapshot-versioned (đã làm phiên này).
[P2] Không cache tầng embedding/retrieval — câu hỏi khác chữ nhưng cùng ý vẫn embed + search lại. Giải pháp: cache embedding theo hash text; cache retrieval theo (session, normalized query). P2 · Thấp–TB.
[P2] Floor 2s che lợi ích cache (bạn giữ chủ đích) — chấp nhận, nhưng cache vẫn tiết kiệm token.
9. Logging / Monitoring ⚠️
[P0] Gần như không có observability. Không metrics, không log latency/route/token/cache-hit. Chỉ logger.info rời rạc ở điểm routing.
Tác động: production lỗi/chậm → mù, không biết đường nào, tỉ lệ cache hit, p95, tỉ lệ fallback SQL-agent, tỉ lệ UNSUPPORTED.
Giải pháp: (a) log có cấu trúc mỗi query: {route, model, latency_ms, tokens, cache_hit, answer_scope, fallback}; (b) endpoint /metrics Prometheus (đếm request/route, histogram latency, cache hit); (c) dashboard tối thiểu. P0 (nền tảng cho mọi tối ưu khác) · TB.
10. Trải nghiệm chờ (waiting UX)
Điểm tốt (vừa làm): streaming thật, skeleton, con trỏ nhấp nháy.
[P2] Trạng thái theo bước chưa chi tiết — nên hiện "Đang tra cứu MES… → Đang soạn…" theo pha thực. P2 · Thấp.
[P2] Floor 2s khiến câu nhanh vẫn chờ — cân bằng UX/tốc độ, bạn đã chọn giữ.
✅ Checklist tối ưu hiệu năng theo tầng
Frontend

 Gửi kèm bản VI của câu gợi ý JA (bỏ 1 lần dịch) · [ ] Trạng thái chờ theo pha thực · [ ] (đã có) streaming + skeleton + cursor
Backend (API)

 P0 Bọc mọi SQLite/IO đồng bộ trong asyncio.to_thread · [ ] Đưa authorize_query sau cache-check · [ ] P0 Log có cấu trúc mỗi query (route/latency/tokens/cache) + /metrics · [ ] Redis cho cache/rate-limit khi scale · [ ] Sửa rate-limit tin X-Forwarded-For (bảo mật)
Vector DB (Qdrant)

 Thêm rerank (bge-reranker) hoặc hybrid BM25+dense · [ ] Chỉnh top_k/score_threshold theo recall@k đo được · [ ] Cache embedding query theo hash
RAG pipeline

 Truyền conversation_context vào LLM (multi-turn) · [ ] Query rewriting cho câu follow-up · [ ] Chunk theo token/ngữ nghĩa thay vì ký tự · [ ] Groundedness check cho câu trả lời tài liệu · [ ] Bước disambiguation khi độ tin cậy thấp
Prompt

 Đo & cắt prompt thừa (few-shot dài?) · [ ] Tách system prompt ổn định để tận dụng prompt caching của provider nếu hỗ trợ
Model

 Hạch toán usage token theo route · [ ] Cân nhắc model nhỏ hơn cho bước intent/rewrite · [ ] Warm model tránh cold-start
Hạ tầng

 Eval harness + golden set trong CI (chống hồi quy chất lượng) · [ ] Load test (p50/p95 dưới N user đồng thời) · [ ] Hàng đợi inference riêng cho embedder (tách upload vs query) · [ ] Alert khi p95/error-rate/fallback vượt ngưỡng
3 việc tôi khuyến nghị làm trước (ROI cao nhất):

Observability (P0, #9) — không đo được thì mọi tối ưu khác là mò.
Bọc to_thread cho employee_directory (P0, #1) — sửa nghẽn đồng thời, effort thấp.
Multi-turn context (#3) — thiếu sót AI-Q&A lớn nhất, ảnh hưởng trực tiếp cảm nhận "thông minh".