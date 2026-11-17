# QA Agent QMS 5M1E

## Phạm vi

`mode="qa"` là một luồng hỏi đáp độc lập về quản lý chất lượng. Meibook cung cấp giao diện, employee gate và gateway; Agent QMS 5M1E nhận câu hỏi, truy vấn QMS và trả lời. Meibook không kết nối trực tiếp Oracle, không chạy SQL Agent QA và không dùng dữ liệu MES/WMS hoặc SQLite MSM/MSA mock để thay thế QMS.

Agent QMS trả lời bằng tiếng Việt. UI có thể hiển thị tiếng Nhật nhưng không tự dịch câu hỏi người dùng nhập hoặc câu trả lời; riêng prompt mẫu/gợi ý có nhãn tiếng Nhật được map trước với câu tiếng Việt gửi tới agent. Câu hỏi đầu tiên của cuộc trò chuyện cần đưa đủ mã máy (vd. `B302-01`) hoặc mã phiếu kiểm tra (vd. `MMSHB B0121-01`), khoảng thời gian và điều kiện liên quan; các câu sau có thể hỏi nối tiếp (vd. "Còn tháng trước thì sao?") theo mục [Hỏi nối tiếp](#hỏi-nối-tiếp-agent-api-11). Các view QMS hiện không có trường mã hàng, nên không hỏi theo mã hàng.

## Luồng xử lý

1. Frontend gửi một request tới `/query/stream` với `mode="qa"`, `conversation_context=[]`, câu hỏi tối đa 2.000 Unicode code points và `qa_reset_context=true` khi màn hình QA đang trống (cuộc trò chuyện mới, sau khi xóa hoặc sau khi reload).
2. Backend xác minh employee gate và từ chối guest `000000`, sau đó chuyển thẳng tới QA handler trước cache, dịch thuật, RAG, MES/WMS, Gmail/Calendar và generic router.
3. Adapter gửi duy nhất một `POST /v1/jobs` kèm `context`/`history` của thread hiện tại; sau đó poll `GET /v1/jobs/{job_id}` trong thời hạn giới hạn. Nếu trạng thái nhận POST không rõ, không tự gửi lại POST.
4. SSE gửi trạng thái chờ/xử lý và khi hoàn tất metadata, `sources=[]`, một event `replace` chứa toàn bộ câu trả lời, rồi `done`.
5. Tham chiếu QA được lọc và hiển thị riêng từ `qa_metadata`. Chúng không phải RAG source, không mở URL và không dùng PDF preview. Raw SQL không hiển thị.
6. Câu trả lời QA render Markdown kèm bảng pipe GFM (`| cột | cột |`) qua `frontend/src/markdown-tables.js`. Chỉ bật bảng; strikethrough, autolink và footnote không bật, ảnh Markdown bị bỏ và raw HTML vẫn hiển thị dạng text. Các mode khác giữ cách render cũ. Bảng chỉ xuất hiện khi Agent QMS tự viết bảng đúng cú pháp trong `answer`.
7. Prompt mẫu và gợi ý tiếp theo của QA nằm trong `frontend/src/qa-prompts.js`, dùng mã máy/mã phiếu lấy từ bộ eval và test của Agent QMS. Với UI tiếng Nhật, nhãn được dịch sang tiếng Nhật nhưng khi bấm frontend gửi câu tiếng Việt tương ứng vì Agent v1 chỉ nhận tiếng Việt. Mỗi lần bấm gửi một câu hỏi live mới (không có câu trả lời soạn sẵn); câu hỏi vẫn đi qua thread hỏi nối tiếp như câu người dùng tự gõ. Gợi ý chỉ hiện dưới câu trả lời QA đã hoàn tất (có `qa_metadata`), bỏ qua câu vừa hỏi và không hiện khi agent đang hỏi lại để làm rõ. Khi dữ liệu QMS được làm mới, cần kiểm tra lại các mã này.

Dừng chờ hoặc đóng kết nối chỉ hủy việc Meibook poll/stream; Agent QMS có thể vẫn tiếp tục chạy job vì API hiện không có thao tác cancel/idempotency.

## Hỏi nối tiếp (Agent API 1.1)

Agent QMS là stateless: mỗi job trả về `context` (object) và `history` (≤20 lượt `{role, content}`) mô tả cuộc trò chuyện, và client phải gửi lại nguyên văn ở câu kế tiếp. Meibook giữ phần này **phía server** trong `src/api/qa_handlers.py` (`QaContextStore`); trình duyệt không nhận và không gửi `context`/`history`.

- Khóa thread là cặp (session ID đã chuẩn hóa UUID, employee ID đã qua employee gate), nên session của người khác không nối tiếp được thread này.
- Store nằm trong bộ nhớ của process app: hết hạn sau 60 phút không dùng, tối đa 500 thread (LRU). Restart app hoặc chạy nhiều worker uvicorn làm mất/tách state; khi đó câu kế tiếp chỉ đơn giản bắt đầu thread mới.
- Câu đầu tiên của thread gửi `context={}` và `history=[]`. Adapter kiểm tra lại state trước khi gửi và khi nhận (`context` ≤50 khóa và ≤16 KB JSON; `history` chỉ giữ role `user`/`assistant`, cắt nội dung về 4.000 ký tự, giữ 20 lượt cuối); state sai định dạng bị bỏ thay vì làm lỗi request.
- Job `done` (kể cả khi agent hỏi lại để làm rõ) thay state đã lưu. Job lỗi, mất job hoặc timeout giữ state cũ để người dùng gửi lại. Agent trả 422 khi đang gửi kèm state được chuẩn hóa thành `QA_CONTEXT_RESET` và state bị xóa. Câu trả lời bị output guardrail thay thế cũng xóa state, để không hỏi tiếp dựa trên nội dung đã bị chặn. `qa_reset_context=true` xóa state trước khi gửi.
- `qa_metadata` bổ sung `context_supported`, `context_applied` (câu này có gửi kèm state không rỗng), `clarification` (agent đang hỏi lại) và `effective_question` (câu hỏi đầy đủ agent đã hiểu). UI hiện nhãn "Agent cần thêm thông tin" khi `clarification=true` và hiện `effective_question` phía trên câu trả lời khi `context_applied=true` và câu này khác câu người dùng gõ.
- Hỏi chủ đề khác nên tạo cuộc trò chuyện mới; reload trang làm UI trống nên câu kế tiếp tự bắt đầu thread mới.

## Cấu hình

Các giá trị dưới đây là placeholder; không lưu API key thật trong repository, tài liệu, log, frontend hoặc ví dụ cấu hình:

```env
QA_AGENT_ENABLED=false
QA_AGENT_URL=
QA_AGENT_API_KEY=
QA_AGENT_REQUEST_TIMEOUT_SECONDS=5
QA_AGENT_TIMEOUT_SECONDS=180
QA_AGENT_POLL_INTERVAL_SECONDS=1.5
QA_AGENT_MAX_INFLIGHT=2
QA_AGENT_HEALTH_TTL_SECONDS=30
```

Chỉ cấu hình QA trong môi trường Dev đã được phê duyệt, bằng file local ignored. `QA_AGENT_URL` phải trỏ tới API jobs gốc đã được quản trị viên xác nhận. Trạng thái bật nhưng thiếu URL/key phải báo misconfigured; không suy ra endpoint mặc định và không fallback sang model hoặc nguồn dữ liệu khác.

Bật/tắt cần sửa cấu hình Dev local, rồi kiểm tra cơ chế nạp env và recreate đúng app Dev nếu Compose truyền cấu hình lúc tạo container. Trước mọi thao tác Docker phải xác minh checkout/branch, `docker-compose.dev.yml`, Compose project, container `-dev`, ports và bind mounts theo `AGENTS.md`. Không dùng Production Compose hoặc container không có hậu tố `-dev` cho thử nghiệm này.

## Quyền truy cập và dữ liệu

- QA yêu cầu employee ID hợp lệ và từ chối guest. Employee ID chỉ là access gate hiện tại, không phải xác thực mạnh hay RBAC; chỉ bật pilot cho đối tượng đã được phê duyệt.
- Backend không gửi employee ID, tên nhân viên, session ID, `conversation_context` của trình duyệt hay dữ liệu MKAC sang Agent QMS. Request upstream chỉ gồm câu hỏi, mã đối chiếu opaque và `context`/`history` do chính Agent QMS trả về ở các lượt trước của cùng thread.
- Không đưa PII hoặc dữ liệu QMS nhạy cảm vào prompt/test nếu chưa được phép. Upstream có thể ghi log câu hỏi.
- Không ghi câu hỏi, câu trả lời, SQL, payload nghiệp vụ, API key hoặc authorization header vào log của Meibook.

## Trạng thái và lỗi

Tab QA chỉ ẩn khi `QA_AGENT_ENABLED=false`. Nếu tính năng đã bật nhưng chưa cấu hình, offline hoặc bận, tab vẫn hiển thị để giải thích tình trạng; UI không tự chuyển sang mode khác.

Mã lỗi QA được chuẩn hóa gồm `QA_DISABLED`, `QA_MISCONFIGURED`, `QA_OFFLINE`, `QA_BUSY`, `QA_JOB_LOST`, `QA_AGENT_ERROR`, `QA_TIMEOUT` và `QA_CONTEXT_RESET` (agent từ chối state hỏi nối tiếp; Meibook đã đặt lại thread và người dùng cần gửi lại câu hỏi đầy đủ). Người dùng có thể gửi lại có chủ đích sau khi xử lý lỗi, nhưng gateway không tự retry POST. `QA_TIMEOUT` có nghĩa Meibook ngừng chờ; nó không chứng minh job upstream đã bị hủy.

Health `/health` chỉ thể hiện liveness/config snapshot của API QMS, không chứng minh API key, LLM, Oracle hay query path QA hoạt động. Không dùng trạng thái `ONLINE` như bằng chứng kết quả nghiệp vụ được xác minh hoặc dữ liệu mới nhất.

## Diễn giải câu trả lời

Câu trả lời từ Agent QMS là nội dung tham khảo/suy luận của agent, chưa được Meibook xác minh theo hợp đồng nghiệp vụ. Citations hỗ trợ tra cứu chéo, không chứng minh độ chính xác, đầy đủ hoặc freshness. Hãy đối chiếu với hệ thống QMS và quy trình có thẩm quyền trước khi quyết định chất lượng hoặc hành động khắc phục. Meibook không trả raw SQL.

## Kiểm thử và go-live

Kiểm thử mặc định dùng mock transport và unit/API tests; không tạo job thật. Live test chỉ được thực hiện sau khi người vận hành chủ động bật Dev, xác minh kết nối từ container Dev, cung cấp credentials qua kênh an toàn và phê duyệt câu hỏi mẫu không nhạy cảm. Không tự khởi động dịch vụ QMS, sửa firewall, thực hiện Production smoke test, hoặc đổi cấu hình Production.

Trước khi cân nhắc Production cần duyệt riêng quyền truy cập/RBAC, secret mới, TLS/mạng, timeout/buffering proxy, tính đúng đắn của dữ liệu và phương án vận hành. Merge/deploy Production là một tác vụ riêng cần yêu cầu rõ ràng.
