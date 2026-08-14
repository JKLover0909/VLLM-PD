# AGENTS.md

Tài liệu này là hướng dẫn chính thức (Onboarding Guide) dành cho bất kỳ AI Agent nào tiếp nhận, bảo trì hoặc phát triển thêm tính năng cho dự án **Meibook (VLLM-PD)**. Bắt buộc phải đọc kỹ trước khi thực hiện thay đổi codebase.

---

## 1. Project Overview
**Meibook (VLLM-PD)** là một hệ thống Chatbot AI nội bộ đa năng được phát triển cho doanh nghiệp (MKAC). Hệ thống hỗ trợ song ngữ (Tiếng Việt và Tiếng Nhật) với 3 luồng nghiệp vụ chính:
1. **Hỏi đáp Hành chính nhân sự (HR/MKAC):** Tra cứu quy định, nội quy, thông tin nhân sự. Kết hợp tìm kiếm cấu trúc qua SQLite và RAG trên Qdrant.
2. **Quản lý Sản xuất (MES):** Tra cứu thông tin mã Lot, mã hàng, thống kê lỗi từ hệ thống MES.
3. **Nghiên cứu tài liệu (Research):** Cho phép người dùng chọn các nhóm tài liệu có sẵn (NotebookLM-like) để đặt câu hỏi chuyên sâu, trích dẫn nguồn cụ thể.

## 2. System Architecture
Kiến trúc Client-Server, triển khai bằng Docker Compose (`docker-compose.web.yml`).
*   **Frontend:** ReactJS + Vite + CSS thuần. SPA render tin nhắn qua Server-Sent Events (SSE).
*   **Backend:** FastAPI (Python). Xử lý định tuyến, auth, session, rate limit, gọi module RAG/MES.
*   **Vector Database:** Qdrant. Lưu trữ embedding (`mkac_knowledge`, `docjp_knowledge`).
*   **LLM Gateway:** LiteLLM Proxy. Điều phối các truy vấn tới mô hình Local (Qwen) qua Ollama hoặc Cloud (OpenAI, Azure) dựa trên fallback chain.
*   **Database (Structured):** SQLite. Dùng để lưu trữ dữ liệu nhân sự và snapshot dữ liệu sản xuất.
*   **Ingestion Pipeline:** Sử dụng `Docling` và `PyMuPDF` để OCR/parse, cắt chunk có overlap, sinh ảnh preview, dùng `BAAI/bge-m3` để tạo embedding (1024-dim).

## 3. Important Directory Structure
*   `frontend/src/`: Giao diện (React). `main.jsx` chứa logic chính, `styles.css` chứa CSS.
*   `src/api/`: Khởi tạo FastAPI (`main.py`), định nghĩa API schemas (`schemas.py`), config (`config.py`), topics (`research_topics.py`).
*   `src/rag/`: Chứa core logic RAG (`rag_pipeline.py`, `vector_store.py`, `parser.py`, `prompts.py`, `media_paths.py`).
*   `src/integrations/`: Xử lý nghiệp vụ MES (`mes_query_service.py`, `mes_sql_agent.py`) và tích hợp ngoài (Gmail).
*   `src/auth/`: Quản lý hỏi đáp cấu trúc cho HR.
*   `src/i18n/`: Dịch thuật tự động qua `translation.py`.
*   `scripts/`: Các file thực thi (import DB, index Qdrant, evaluation).
*   `tests/`: Thư mục Pytest.
*   `config/`: Chứa file manifest (`docjp_manifest.json`, `research_topics.json`).

## 4. Main Workflows

### 4.1 HR / MKAC Q&A
Nhận diện Intent xem có phải câu hỏi cấu trúc không (lấy từ SQLite). Nếu không, query collection `mkac_knowledge` trong Qdrant -> Rerank -> Gắn vào prompt -> LLM. Hỗ trợ tự dịch câu hỏi tiếng Nhật sang tiếng Việt trước khi retrieval.

### 4.2 MES Q&A
Parse thông số (mã Lot, Mã hàng, Thời gian). Ưu tiên deterministic/rule-based SQL cho MES vì dễ kiểm chứng, ổn định và ít hallucination hơn LLM-generated SQL. Tuy nhiên agent vẫn phải kiểm tra schema, dữ liệu đầu vào, điều kiện lọc và kết quả trả về trước khi kết luận. Nếu phức tạp, gọi `mes_sql_agent` (LLM-based) làm fallback.

### 4.2b WMS Q&A

WMS (`mode=wms`) code đã được merge nhưng Production WMS snapshot và SQL Agent mặc định **disabled**. UI WMS chỉ hiện khi backend xác nhận snapshot khả dụng (`wmsStatus.available`). Không giả định WMS đã hoạt động trên Production nếu chưa import snapshot và bật cấu hình tương ứng.

Khi enable WMS trên Production, phải tuân thủ các ràng buộc đã thiết lập ở Dev: read-only, allowlisted views, row limit, timeout; chỉ fail-closed trước LLM khi snapshot disabled/unavailable/incompatible/query-error; đánh dấu rõ kết quả LLM-generated SQL là suy luận độ tin cậy thấp (`SQL_AGENT_ANSWER_UNVERIFIED`); không reuse MES database hoặc MES SQL Agent cho WMS. Ba rủi ro đã biết: cộng số lượng giữa các UOM chưa có master quy đổi, so sánh cross-era current/legacy vốn khác grain, và suy diễn WIP/bottleneck/trend/min-stock vốn không tồn tại trong snapshot.

### 4.3 Research / NotebookLM-like Document QA
Người dùng chọn 1 chủ đề trên UI. Backend query vào Qdrant (`docjp_knowledge`) kết hợp filter `metadata.category`. Không dùng double-translation đối với tài liệu tiếng Nhật để tránh nhiễu RAG.

### 4.4 Source Preview / Citation Flow
Nhấn vào nguồn trên UI, API `/sources/preview` đọc file ảnh PNG (được generate lúc index) từ ổ cứng (`docjp_processed` hoặc `mkac_processed`). Hàm `resolve_processed_image_path` trong `src/rag/media_paths.py` xử lý mapping đường dẫn từ Host (`/home/...`) vào Docker (`/app/...`).

## 5. Host Python và cách chạy dự án

Dự án dùng Docker bind-mount. Runtime Docker dùng Python riêng trong image và không dùng Conda host. Mọi lệnh Python chạy trực tiếp trên host (test, lint, script) phải đi qua `scripts/meibook-python`; cấm dùng bare `python`, `python3`, `pip`, `pytest`, `pyflakes` hoặc Conda `base`.

*   **Main host env:** `meibook`, Python 3.10; bootstrap từ `environment.host.yml` và `requirements.host.in`.
*   **Dev host env:** `meibook-dev`, Python 3.10; chỉ dùng trong checkout `/home/jkl/Code/VLLM-PD-dev`.
*   **CPU host mặc định:** wrapper đặt embedding/OCR CPU để không cạnh tranh GPU Production. Chỉ đặt `MEIBOOK_ALLOW_GPU=1` cho tác vụ GPU đã được người dùng phê duyệt.
*   **Restart app Production:** `docker compose -f docker-compose.web.yml restart app`
*   **Build Frontend:** `cd frontend && npm run build` (cần chạy sau khi sửa UI).
*   **Index tài liệu Research:** `MAX_DOCUMENT_PAGES=200 scripts/meibook-python scripts/index_docjp_documents.py --reindex`

## 6. How to Test, Lint, Build and Verify
*   **Backend Test:** `scripts/meibook-python -m pytest tests/ -q --ignore=tests/test_mes_integration.py --ignore=tests/test_mkac_pipeline.py`
*   **Backend Linter:** `scripts/meibook-python -m pyflakes src/api/*.py src/rag/*.py src/integrations/*.py`
*   **Frontend Lint:** `cd frontend && npm run lint`
*   **Xem log API/Preview:** `docker logs meibook-web --tail 50`

## 7. Development Conventions
*   **Coding Style:** Tên biến, tên hàm, code comment và commit message phải tuân theo convention hiện có của repo (ưu tiên tiếng Anh kỹ thuật ngắn gọn).
*   **Schemas:** Cẩn trọng khi đổi `schemas.py` vì có thể làm frontend React hỏng payload.
*   **UI/UX:** Cần giữ dark mode, responsive, dùng CSS thuần trong `styles.css`. Giao diện phải chuyên nghiệp, rõ luồng người dùng.

## 8. Agent Working Rules
*   **Ngôn ngữ giao tiếp:** Luôn trao đổi, giải thích và báo cáo kết quả với người dùng bằng tiếng Việt, trừ khi người dùng yêu cầu tiếng Anh.
*   **Làm thật, không đoán mò:** Luôn đọc log, phân tích nguyên nhân gốc trước khi sửa code.
*   **Test-driven:** Sửa code xong phải chạy test. Nếu test fail, trước tiên phải xác định nguyên nhân (code sai, test lỗi thời, thiếu dependency). Chỉ cập nhật test khi logic/spec mới đã được xác nhận hợp lệ; nếu không, phải sửa code.
*   **Tác động dây chuyền:** Khi sửa API, frontend, hoặc DB, phải kiểm tra các module bị ảnh hưởng.
*   **Tối ưu RAG:** Ưu tiên chất lượng câu trả lời, bám sát nguồn tài liệu, trích dẫn rõ ràng, giảm hallucination.
*   **Chủ động rà soát:** Đề xuất cải tiến khi thấy code chưa tối ưu, nhưng vẫn phải ưu tiên hoàn thành task chính trước.

## 9. Data, Security and Privacy Rules
*   **Tuyệt đối không:** `git push`, `git reset --hard`, xóa file lớn, drop database, clear Qdrant collection, hoặc reindex toàn bộ nếu người dùng chưa yêu cầu rõ.
*   **Không tự ý commit** nếu người dùng chưa yêu cầu.
*   **Thao tác nguy hiểm:** Phải giải thích trước và chờ người dùng xác nhận.
*   **Bảo mật:** Không hard-code API key, token, password. Không log dữ liệu nhạy cảm (mã nhân viên, dữ liệu HR/MES thật). Không đưa dữ liệu thật vào test/prompt nếu chưa ẩn danh.
*   **Anti-hallucination:** Với câu hỏi không có trong nguồn, chatbot phải nói rõ "không tìm thấy thông tin", cấm tự suy diễn quy định công ty. Không để LLM tự bịa số liệu MES.

## 10. Debugging Guide
*   **Preview image not found:** Kiểm tra log `docker logs meibook-web`. Lỗi thường do đường dẫn mount Host/Docker lệch, hoặc PDF chưa được sinh ảnh lúc index.
*   **Giao diện web không cập nhật:** Quên chạy `npm run build` hoặc chưa Hard Reload (`Ctrl + Shift + R`) trình duyệt.
*   **ERR_NGROK_334:** URL ngrok bị chiếm bởi máy khác. Khởi động ngrok với config rỗng (`ngrok http 8001 --config=none`) hoặc dùng cloudflared.
*   **Test Token Budget Fail:** Thường do sửa cấu hình `MAX_TOKENS` trong `.env` / `rag_pipeline.py` nhưng quên cập nhật số expected trong `test_token_budgets.py`.

## 11. Definition of Done
A task is only considered done when:
1. The requested change is implemented within scope.
2. Related frontend/backend/API/database impacts are checked.
3. Relevant test, lint, build, or smoke check has been run, or the reason for not running is clearly stated.
4. Changed files are summarized.
5. Verification result and remaining risks are reported to the user.

## 12. Known Weak Points / Future Improvements
*   `main.jsx` đang rất lớn (>3000 dòng), cần chia nhỏ thành các React Components.
*   Chức năng user tự upload tài liệu cá nhân đang tạm ẩn, cần khôi phục qua collection `docmind_documents`.
*   Truy vấn tiếng Việt trên kho tài liệu tiếng Nhật (Research) có thể được tối ưu thêm bằng HyDE (Hypothetical Document Embeddings).
*   Chưa có cảnh báo MES tự động khi tỷ lệ lỗi vượt ngưỡng.
