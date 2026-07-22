# Bản đồ độ bao phủ kiểm thử (TestPrompt Coverage Mapping)

Tài liệu này lập bản đồ đối chiếu các trường hợp kiểm thử thủ công trong 3 tệp TestPrompt với các bộ kiểm thử tự động (pytest) và runner kịch bản trong dự án Meibook.

---

## 1. TestPrompt.md (MES & HR RAG/Database)

Bộ test prompt này được chia làm hai hình thức chạy chính:
1. **Live API Runner:** Gọi trực tiếp vào API Server đang chạy qua script `scripts/run_mes_hr_testprompt.py`.
2. **Pytest (Unit & Integration):** Chạy cục bộ bằng dữ liệu giả lập (mock/fixture SQLite) để xác thực logic mã nguồn nhanh chóng mà không cần kết nối LLM.

### Ánh xạ chi tiết case

| Nhóm case | ID prompt | Runner tự động | Tệp Pytest liên quan | Ghi chú |
|---|---|---|---|---|
| **A. MES — Số liệu thật** | `1` – `14` | `run_mes_hr_testprompt.py` | `tests/test_mes_database.py`, `tests/test_query_routing.py` | Xác thực qua SQL views cục bộ `v_error_details`. |
| **B. MES — Biên, phủ định** | `15` – `22` | `run_mes_hr_testprompt.py` | `tests/test_mes_database.py`, `tests/test_mes_time_sql_routing.py` | Kiểm thử các câu hỏi không có dữ liệu, sai tháng (tháng 13), hoặc so sánh rỗng. |
| **C. MES — Ngoài phạm vi** | `23` – `27` | `run_mes_hr_testprompt.py` | `tests/test_mes_integration.py` | Trả lời "không có thông tin", từ chối bịa đặt tên công nhân/chi phí sửa lỗi. |
| **D. MES — Định dạng/Yêu cầu** | `28` – `31` | `run_mes_hr_testprompt.py` | `tests/test_mes_sql_agent.py` | Xác thực trả bảng Markdown hoặc trả đúng 1 số duy nhất. |
| **E. MES — An toàn/Injection** | `32` – `35` | `run_mes_hr_testprompt.py` | `tests/test_mes_sql_agent.py` | Ngăn chặn và từ chối các câu lệnh phá hủy như `DROP TABLE`. |
| **F. MES — Tiếng Nhật** | `36` – `39` | `run_mes_hr_testprompt.py` | `tests/test_translation_service.py` | Xác minh bypass dịch thuật cho câu hỏi cấu trúc tiếng Nhật. |
| **G. MES — Multi-turn** | `40` – `42` | `run_mes_hr_testprompt.py` | `tests/test_mes_database.py` | Tự động giữ `session_id` và kế thừa ngữ cảnh câu trước. |
| **H. HR — Tên riêng thật** | `43` – `52` | `run_mes_hr_testprompt.py` | `tests/test_query_routing.py`, `tests/test_employee_directory.py` | Tra cứu phòng ban, chức vụ của nhân viên từ danh bạ SQLite. |
| **I. HR — Phủ định, mơ hồ** | `53` – `56` | `run_mes_hr_testprompt.py` | `tests/test_employee_directory.py` | Xử lý tên gọi tắt ("Anh Phi" -> Nguyễn Trọng Phi) hoặc người không có. |
| **J. HR — Nội dung tài liệu** | `57` – `68` | `run_mes_hr_testprompt.py` | `tests/test_mkac_pipeline.py` | RAG trên tài liệu công ty (nội quy, phụ cấp, xe cộ, quyết định). |
| **K. HR — Ngoài phạm vi** | `69` – `72` | `run_mes_hr_testprompt.py` | `tests/test_query_routing.py` | Từ chối trả thông tin nhạy cảm không có (như lương cơ bản, số điện thoại). |
| **L. HR — Multi-turn** | `73` – `75` | `run_mes_hr_testprompt.py` | `tests/test_query_routing.py` | Giữ ngữ cảnh phòng ban / mức phụ cấp qua nhiều lượt hỏi. |
| **M. HR — Tiếng Nhật** | `76` – `79` | `run_mes_hr_testprompt.py` | `tests/test_translation_service.py` | Dịch ngược câu hỏi tiếng Nhật -> tiếng Việt RAG -> trả kết quả tiếng Nhật. |
| **N. HR — An toàn/Injection** | `80` – `83` | `run_mes_hr_testprompt.py` | `tests/test_query_routing.py` | Chặn yêu cầu sửa thông tin hoặc gửi toàn bộ lương ra ngoài. |
| **O. Đa domain** | `84` – `86` | `run_mes_hr_testprompt.py` | `tests/test_mes_integration.py` | Hỏi sai mode hoặc hỏi ghép chéo (MES + HR). |
| **P. Diễn đạt khác** | `87` – `90` | `run_mes_hr_testprompt.py` | `tests/test_query_routing.py` | Câu hỏi không dấu, sai chính tả nhẹ. |
| **Q. Tiếng Nhật đầy đủ** | `JA-001` - `JA-090` | `run_mes_hr_testprompt.py` | `tests/test_translation_service.py` | Mirror toàn bộ 90 câu hỏi trên bằng tiếng Nhật. |

*Lưu ý an toàn: Runner `run_mes_hr_testprompt.py` mặc định **tự động bỏ qua (skip)** case `80` và `JA-080` (yêu cầu gửi email thật ra ngoài) trừ khi có cờ `--run-unsafe`.*

---

## 2. TestPrompt_Research.md (Research/DocJP RAG)

Môi trường Research phụ thuộc dữ liệu RAG động từ các file Markdown tiếng Nhật nên pytest tập trung kiểm tra tầng hạ tầng (registry/cache), còn semantic chất lượng câu trả lời được kiểm thử qua runner API thật.

*   **API Runner:** `scripts/run_research_testprompt.py` (chạy nhanh) và `scripts/run_research_evaluation.py` (xuất báo cáo Markdown đầy đủ tại `logs/research_evaluation_report.md`).
*   **Pytest unit/integration:**
    *   `tests/test_research_topics.py`: Xác thực cấu hình, danh sách nhóm tài liệu (INFO, LEGAL, ACC, GA) và mapping ngôn ngữ.
    *   `tests/test_research_query_cache.py`: Xác thực hoạt động của bộ nhớ đệm (Q&A cache) dựa theo ngữ cảnh nhóm.
    *   `tests/test_research_routing_safety.py`: Đảm bảo an toàn định tuyến, không truy cập chéo session hoặc lộ dữ liệu.
    *   `tests/test_research_upload_preview.py`: Đảm bảo file PDF/Markdown upload được chunking và tạo ảnh xem trước (preview) chính xác.

---

## 3. TestPrompt_ReportAgent.md (Báo cáo lỗi sản xuất MES)

Report Agent được tích hợp sâu trong pytest để kiểm thử logic tạo bảng, tính toán và bảo mật HTML trước khi render.

*   **API Runner:** Chạy thủ công trên giao diện web hoặc API test client.
*   **Pytest unit/integration:**
    *   `tests/test_report_agent.py`: 
        *   `test_report_intent_requires_report_and_mes_context`: Kiểm tra định tuyến (Nhóm A).
        *   `test_report_period_parses_month_day_and_range`: Kiểm tra phân tích kỳ báo cáo ngày/tháng/khoảng ngày (Nhóm B).
        *   `test_report_agent_generates_verified_report_and_excludes_test_data`: Kiểm tra tính chính xác của KPI và loại bỏ dữ liệu dummy (Nhóm C).
        *   `test_report_html_escapes_data`: Chống HTML injection (Nhóm E).
    *   `tests/test_report_api.py`:
        *   `test_handle_report_query_refuses_unsupported_without_running_agent`: Kiểm tra cơ chế fail-closed cho các yêu cáo nằm ngoài mẫu chuẩn (Nhóm D).
        *   `test_streaming_report_emits_complete_agent_protocol`: Kiểm tra vòng đời SSE và payload của artifact (Nhóm F).
    *   `tests/test_report_artifact_store.py`: Kiểm tra vòng đời lưu trữ, TTL và ghi đè cache của báo cáo HTML (Nhóm E).

---

## Lệnh kiểm tra nhanh dự án (Chỉ đọc)

```bash
# 1. Chạy toàn bộ unit/integration test cục bộ không cần LLM
scripts/meibook-python -m pytest tests/ -q --ignore=tests/test_mes_integration.py --ignore=tests/test_mkac_pipeline.py

# 2. Chạy live test chế độ Research (3 câu mẫu)
scripts/meibook-python scripts/run_research_testprompt.py --limit 3 --delay 4.1 --base-url http://localhost:8002

# 3. Chạy live test chế độ MES/HR (5 câu mẫu)
scripts/meibook-python scripts/run_mes_hr_testprompt.py --limit 5 --delay 4.1 --base-url http://localhost:8002
```
