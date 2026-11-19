# progress.md — Nhật ký bàn giao giữa các phiên

> Quy tắc: entry mới nhất ở trên cùng; giữ khoảng 10 entry gần nhất (entry cũ hơn
> có thể xóa vì đã nằm trong Git history). Chỉ ghi kết quả của lệnh đã thật sự chạy.
> **Không** ghi secret, token, mã nhân viên, tên người hay số liệu HR/MES thật.
> Trạng thái tính năng nằm ở `feature_list.json`; quy trình phiên nằm ở `CLAUDE.md`
> → "Session Harness".

## Current state

_Cập nhật: 2026-11-17 (tất cả alias dùng Ollama local `qwen3.5:9b`; Production chưa đổi)_

- **Checkout:** `/home/jkl/Code/VLLM-PD-dev`, nhánh `dev` @ `810f720`. `main` đi trước
  `dev` 1 commit (`7fec558`); `dev` không có commit nào chưa vào `main`.
- **Working tree:** nhiều thay đổi chưa commit từ các phiên trước (trước phiên
  2026-10-31: 40 file sửa, 27 file untracked) cộng file harness của phiên này. Chưa
  review nội dung; không revert/format/stage file mình không sửa.
- **Baseline kiểm thử (2026-10-31, Dev, trước khi thêm harness):**
  pytest suite mặc định 909 passed; `npm --prefix frontend test` 44/44 pass;
  pyflakes trên `src/` sạch; `git diff --check` sạch.
- **Đang làm dở:** `guard.cross_turn` (test pass, chờ review), `mode.qa` (chưa live
  test với QMS :8600), `llm.routing` (code Dev chuẩn bị remote Qwen target; chưa có
  tất cả alias dùng Ollama local `qwen3.5:9b`; Production chưa đổi).
- **Cần người dùng quyết định:**
  1. Review + commit nhóm thay đổi chưa commit (nên tách commit theo nhóm bên dưới).
  2. Có merge `7fec558` từ `main` về `dev` không.
  3. Ưu tiên sửa `hygiene.test_fixture_pii` (fixture test dùng tên thật).

### Nhóm thay đổi chưa commit (theo tên file, chưa review nội dung)

| Nhóm | File chính |
| --- | --- |
| QA mode | `src/integrations/qa_agent_service.py`, `src/api/qa_handlers.py`, `src/api/schemas.py`, `frontend/src/qa-*.js`, `Markdowns/QA_AGENT.md`, `tests/test_qa_*.py`, `frontend/test/qa-*.test.js` |
| Guards | `src/api/prompt_guard.py`, `query_guard.py`, `scope_guard.py`, `src/api/main.py`, `tests/test_prompt_*guard.py`, `tests/test_query_*guard*.py` |
| Frontend modules | `frontend/src/markdown-tables.js`, `query-stream.js`, `mode-tabs.js`, `main.jsx`, `components/*.jsx`, `styles.css`, `package*.json` |
| MSM mock DB | `scripts/create_msm_mock_database.py`, `database/schema/msm_msa.sql`, `database/MSM_MSA_SQL_CONTEXT.md`, `tests/test_create_msm_mock_database.py` |
| Model routing | `litellm_config.yaml`, `tests/test_litellm_routing_config.py`, `tests/test_model_routes.py` |
| Backend khác | `src/actions/*.py`, `src/i18n/translation.py`, `src/integrations/mes_*.py`, `cctvai_hardware_service.py`, `src/rag/*.py`, `src/api/helpers.py`, `sse.py` |
| Infra / docs | `docker-compose*.yml`, `.env*.example`, `README.md`, `AGENTS.md`, `Markdowns/*.md`, `database/README.md` |
| Harness (2026-10-31) | `CLAUDE.md`, `AGENTS.md` (2 chỗ), `feature_list.json`, `progress.md`, `scripts/harness-check.sh`, `.claude/settings.json`, `.claude/hooks/*.sh` |

## Session log

### 2026-11-18 — Intro robot chạy mỗi lần tải trang (frontend, Dev)

- **Yêu cầu:** hiệu ứng khởi đầu chỉ thấy 1 lần/ngày (lưu `meibook-intro-seen` = ngày trong localStorage); người dùng muốn lần nào vào/reload cũng thấy.
- **Sửa:** `src/intro.js` bỏ `INTRO_STORAGE_KEY`, `markIntroSeen`, logic theo ngày; `shouldPlayIntro({ search })` trả `true` trừ khi
  URL có `?intro=0` (tắt cho 1 lần tải: kiosk/demo/test). `IntroSplash.jsx` không đọc/ghi storage nữa. Giữ nguyên: bỏ qua được bằng
  phím/chuột/chạm/cuộn, không chặn app, reduced-motion dùng bản ngắn ≤ 0,6 s, bản đầy đủ 2,6 s. Cờ cũ còn trong trình duyệt bị bỏ qua.
- **Test:** `test/intro.test.js` viết lại 3 test hành vi (mỗi lần tải, không phụ thuộc storage, `?intro=0`) — 2 đỏ trước khi sửa.
  `npm --prefix frontend test` 92/92. Playwright mock 15/15 (lần đầu + reload ×2 đều hiện và tự kết thúc, không lưu gì, cờ cũ không
  chặn, phím tắt < 1 s, `?intro=0` tắt rồi lần sau hiện lại, reduced-motion hiện lại khi reload). Bộ cũ chuyển sang `?intro=0`:
  robot 27/27, tương tác 30/30, motion 18/18, focus 10/10, theme 10/10, composer 20/20. Build Dev; bundle 8002 khớp, không còn
  `meibook-intro-seen`; `/health` 200. Không commit.

### 2026-11-18 — Nút đính kèm đè chữ trong ô nhập Research (frontend, Dev)

- **Lỗi:** ở Research/upload, `.composer-attach` (nút ghim giấy) đè lên chữ placeholder. Đo trong trình duyệt: nút rộng 44 px
  (kích thước `.icon-button` chung) đặt ở `left: 9px` → kết thúc x=506, trong khi chữ bắt đầu ở x=501 (`padding-left: 48px` cố định) →
  chồng 5 px. Trên phone nút 48 px / chữ cách 4 px.
- **Sửa:** ba biến `--attach-size/--attach-inset/--attach-gap` trên `.composer:has(.composer-attach)`; nút và `padding-left` của
  textarea cùng tính từ ba biến này (36 px desktop, 44 px phone), selector `.composer .composer-attach` đủ mạnh để thắng
  `.icon-button` (và override trong media query). Bỏ hai giá trị cứng `48px`/`58px` đã lệch nhau.
- **Xác minh:** Playwright mock, không proxy, Research + upload (1 file giả): 20/20 (VI sáng desktop, JA tối desktop, phone, tablet:
  không chồng, khoảng cách ≥ 8 px, không tràn, không lỗi); các bộ cũ robot 27/27, tương tác 30/30, motion 18/18, focus 10/10,
  theme 10/10; `npm --prefix frontend test` 92/92. Build Dev; asset 8002 khớp; `/health` 200. Không commit.

### 2026-11-18 — Theme "Meiko Light" cho môi trường nhà máy (frontend, Dev)

- **Quyết định người dùng:** màu chủ đạo xanh Meiko + vàng an toàn; mặc định nền sáng.
- **Màu:** token `--aurora-*` (giữ tên để không phải đổi rule phụ thuộc, kể cả `intro.css`) chuyển sang xanh công nghiệp
  `#0284c7/#0369a1/#075985`, vàng `#f59e0b` chỉ cho điểm nhấn nhỏ (icon chip thứ 2 với chữ `#451a03` 7:1, vạch dưới nhãn khi hover,
  con trỏ stream, vạch quét). Nền `#f7f9fc`, chữ slate `#0f172a`. ~170 chỗ indigo/violet còn lại trong `styles.css` + `intro.css`
  đổi sang thang xanh tương ứng; giữ màu ngữ nghĩa (WMS xanh lá, lỗi đỏ, accent topic tím/OpenAI dot). Tương phản: nút chữ trắng
  ≥ 5.9:1, ink 7.2:1, muted 4.5:1+. Bản trước: `/tmp/meibook-ui-audit/{styles,intro}.before-meiko-light.css`.
- **Mặc định nền sáng:** `src/theme-preference.js` + `test/theme-preference.test.js` (3 test, đỏ trước khi có module).
  `"system"` cũ được lưu ngầm → chuyển sang `light` **một lần** (cờ `meibook-theme-light-default`); lựa chọn sáng/tối/hệ thống
  sau đó được tôn trọng. Script inline trong `index.html` khớp cùng logic để không nháy nền tối; theme-color mới.
- **Phiên khác:** working tree có `IntroSplash.jsx`, `intro.css`, `intro.js`, `test/intro.test.js` (không do phiên này tạo, chỉ đổi
  màu trong `intro.css`). Script Playwright được cập nhật để đánh dấu intro đã xem trước khi kiểm thử.
- **Xác minh:** `npm --prefix frontend test` 92/92. Playwright mock, không proxy: theme 10/10 (OS tối → sáng; "system" cũ → sáng một
  lần; tối chủ động giữ; "system" sau migrate tôn trọng; nút trắng trên xanh; không còn indigo/violet trong style render; phone không
  tràn), robot 27/27, tương tác 30/30, motion 18/18, focus 10/10. Build Dev; asset 8002 khớp; `/health` 200. Không commit.

### 2026-11-18 — Intro "thức dậy" 3D lần mở đầu + đồng hồ chờ thật (frontend, Dev)

- **Research:** `/tmp/meibook-ui-research/research.md` (nguồn NN/g, web.dev, MDN…). "Quy tắc 8 giây / cá vàng" là số liệu
  không có nguồn; điều có thật là ấn tượng đầu ~50 ms và mốc NN/g 0,1 s / 1 s / 10 s. Three.js/R3F/Spline/Rive tốn 130–500+ KB
  gzip cho vài giây hiệu ứng nên không dùng; chọn CSS 3D transform + SVG robot sẵn có (0 KB thư viện, chạy offline).
- **Thêm:** `frontend/src/intro.js` (+ `test/intro.test.js`, 6 test; mutation kiểm tra test đỏ khi bỏ logic once-per-day):
  phát 1 lần/ngày (localStorage `meibook-intro-seen`; storage lỗi → bỏ qua), biến thể `full`/`lite`/`reduced` theo
  reduced-motion, save-data, CPU ≤ 2 nhân; `?intro=1` để demo; `formatElapsed`. `components/IntroSplash.jsx` + `intro.css`:
  overlay `aria-hidden`, nền aurora xoay, 3 vòng quỹ đạo 3D + 18 hạt hội tụ, robot "thức dậy" (mắt mở), chữ MEIBOOK; dòng
  trạng thái đổi sang "Sẵn sàng" chỉ khi `/health` thật trả về; tự tắt ~2,6 s, bấm phím/chạm/cuộn để bỏ qua. App render và
  dùng được ngay bên dưới (không chặn). Reduced motion: tĩnh, 0,6 s.
- **"Sống mỗi giây", trung thực:** `MessageList` thêm đồng hồ đếm thời gian chờ thật (giây/秒) trong bóng chờ, `aria-hidden`
  để không làm phiền trình đọc màn hình; biến mất khi câu trả lời về.
- **`main.jsx`:** chỉ thêm import + 1 dòng `<IntroSplash>` (tránh đụng thay đổi dở của phiên khác).
- **Xác minh:** `npm --prefix frontend test` 85/85. Playwright mock, không proxy (`/tmp/meibook-ui-audit/intro-check.mjs`):
  21/21 (hiện lần đầu, 3D rings/hạt, aria-hidden, app render bên dưới, "sẵn sàng" sau /health, tự tắt, cùng ngày không hiện
  lại, bỏ qua bằng phím/chuột, phone không tràn, reduced-motion, `?intro=1`, đồng hồ đếm thật, không lỗi). Ảnh sáng/tối/phone
  ở `/tmp/meibook-ui-audit/intro/`. Build Dev (JS +~1,4 KB gzip); asset 8002 khớp; `/health` 200. Không commit.
- **Chưa làm (đề xuất):** nhãn giai đoạn thật từ backend qua SSE ("Đang tìm tài liệu", "Đang viết"), timeline từng bước
  có thời lượng, nút Hủy rõ khi chờ > 10 s.
- **Hoàn thiện (cùng ngày):** thêm sàn lưới phối cảnh 3D trôi về phía người xem; vòng quỹ đạo vẽ bằng mask trong suốt
  thật; dòng trạng thái + thanh tiến độ theo `/health` thật (đang kết nối → shimmer, online → đầy, offline → đỏ "Chưa kết
  nối được máy chủ", không bao giờ báo "sẵn sàng" giả); khi thoát, robot intro bay vào đúng vị trí robot thật
  (`handoffTransform`, 3 test). Sửa lỗi thật: `runViewTransition` để rơi promise bị từ chối "Transition was skipped" thành
  `pageerror` khi transition bị bỏ qua (test đỏ trước, xanh sau).
- **Kiểm thử lặp:** `npm --prefix frontend test` 89/89. Ma trận Playwright mock `/tmp/meibook-ui-audit/intro-matrix.mjs`
  (6 cỡ màn hình × sáng/tối × VI/JA, 3 trạng thái server, 6 cách bỏ qua, biến thể full/lite/reduced, reload nhanh,
  dùng app sau khi bỏ qua) chạy 1 + 3 + 2 vòng: 289 → 867/867 → 578/578, không lỗi. Bộ cũ khi intro đã xem: robot 27/27,
  motion 18/18, tương tác 30/30, focus 10/10. Build Dev; asset 8002 khớp; `/health` 200. Không commit.

### 2026-11-18 — Theme "Aurora": màu chủ đạo indigo→violet→cyan + font hiện đại (frontend, Dev)

- **Màu:** remap toàn bộ teal thương hiệu (~130 chỗ: hex + rgba) sang thang indigo/violet trong `styles.css`; giữ nguyên màu ngữ
  nghĩa (WMS ready xanh lá, lỗi đỏ, cảnh báo vàng). Bản trước khi đổi: `/tmp/meibook-ui-audit/styles.before-aurora.css`.
- **Lớp "Aurora" cuối `styles.css`:** token `--aurora-*`, nền body gradient mờ, header kính mờ + đường kẻ gradient, tiêu đề chữ
  gradient, thẻ tác vụ viền gradient khi hover + icon chip gradient (3 tổ hợp luân phiên), preview và ô nhập viền gradient khi
  focus, nút chính gradient, bóng thoại người dùng gradient nhạt, avatar trợ lý gradient. Sáng/tối, reduced-motion, forced-colors.
- **Font:** Merriweather (serif) → Be Vietnam Pro (VI) + Noto Sans JP (JA) qua Google Fonts trong `index.html`; theme-color mới.
- **Tương phản:** gradient sau chữ trắng chỉ dùng điểm dừng ≥ 4.5:1 (#4f46e5 6.3, #7c3aed 5.7, #0e7490 5.4, #be185d 6.0);
  chữ tiêu đề gradient sáng/tối, link/ink 7.4–9.5. Sửa specificity tiêu đề (rule `.empty-copy h1` cũ thắng).
- **Xác minh:** `npm --prefix frontend test` 79/79; Playwright mock không proxy: robot/bóng thoại 27/27, tương tác 30/30,
  motion 18/18, focus 10/10; ảnh sáng/tối/phone không tràn, không lỗi. Build Dev; asset 8002 khớp; `/health` 200. Không commit.
- **Lưu ý:** font tải từ Google Fonts (máy offline/chặn mạng sẽ dùng system-ui). Logo MEIKO giữ nguyên màu thương hiệu riêng.

### 2026-11-18 — Viền focus thừa trong ô chọn chế độ (frontend, Dev)

- **Nguyên nhân:** rule toàn cục `select:focus-visible { outline: 3px ... }` vẽ viền quanh chính thẻ `<select>`; thẻ này bắt đầu sau
  icon nên cạnh trái nằm giữa icon và chữ, chồng lên viền focus của `.mode-selector-wrap`.
- **Sửa:** `.mode-selector:focus, .mode-selector:focus-visible { outline: none; }`; viền focus do wrapper đảm nhận (bao icon + chữ).
- **Xác minh:** Playwright mock 10/10 (sáng/tối × chuột/bàn phím: select không còn outline, wrapper outline 2px bao icon);
  tương tác 30/30; `npm --prefix frontend test` 79/79. Build Dev; asset 8002 khớp; `/health` 200. Không commit.

### 2026-11-18 — Bỏ thanh trạng thái robot trống (frontend, Dev)

- **Vấn đề:** sau khi robot đi khắp màn hình, thanh `agent-presence.compact` dưới header (~75 px) gần như luôn trống:
  ô neo không có robot, dòng trạng thái rỗng ở `ready`, chỉ còn nút ⏸.
- **Thay đổi:** bỏ thanh. `AgentPresence` chỉ còn sân khấu chào; thêm `AgentStatus` (live region `sr-only`, vẫn thông báo
  trạng thái cho trình đọc màn hình) và `MotionToggle` (nút trên header cạnh nút giao diện). Robot khi câu trả lời bị cuộn khuất
  về góc phải vùng hội thoại (`corner`, thay `bar`). Vạch quét chuyển lên mép trên ô nhập. Grid workspace bớt 1 hàng.
  Bỏ CSS compact cũ; rule cũ `border:0` làm nút header mất viền → giới hạn lại cho `.agent-more-prompts`.
- **Xác minh:** `npm --prefix frontend test` 79/79 (test `corner` đỏ trước khi sửa). Playwright mock, không proxy: robot/bóng
  thoại 27/27 (thêm: không còn thanh, tin nhắn bắt đầu ngay dưới header, nút ở header, live region còn, robot về góc khi cuộn),
  motion 18/18, tương tác 30/30. Build vào `frontend/dist` Dev; asset 8002 khớp; `/health` 200. Không restart, không commit.

### 2026-11-17 — Robot đi khắp màn hình + bóng thoại truyện tranh (frontend, Dev)

- **Thêm:** `frontend/src/robot-path.js` (chọn đích theo ưu tiên: lỗi hiển thị → request đang chạy → preview → thẻ được
  hover/focus → đang gõ → tuần tra nhàn rỗi → chỗ nghỉ; tính vị trí + clamp + thời gian di chuyển) với
  `frontend/test/robot-path.test.js` (9 test); `components/RoamingRobot.jsx` (một robot duy nhất, `aria-hidden`, đặt tuyệt đối
  trong workspace, theo dõi bằng ResizeObserver/MutationObserver/scroll; bong bóng suy nghĩ "…", "?" , "!" theo trạng thái thật).
  `AgentPresence` chỉ còn ô neo trống (`data-robot-anchor`); `TaskLauncher` có `data-task-index`; `MessageList` có `data-latest`.
- **Bóng thoại:** tin nhắn người dùng bên phải, trợ lý bên trái; viền mực 2px, đổ bóng cứng, đuôi trỏ về người nói; trạng thái chờ là
  bóng viền đứt; bóng "pop" khi xuất hiện; bảng/code giữ bóng full width. Sáng/tối, phone, forced-colors.
- **Quyết định:** focus ô nhập không đủ để robot rời câu trả lời (app tự focus sau mỗi câu); chỉ khi người dùng gõ (`composing`).
  Tắt chuyển động / reduced-motion → robot đứng yên ở chỗ neo, không tuần tra, không di chuyển.
- **Xác minh:** `npm --prefix frontend test` 78/78. Playwright mock, không proxy: robot/bóng thoại 23/23 (hover thẻ, preview, working
  → câu trả lời, gõ → ô nhập, lỗi → banner lỗi, reduced/motion-off đứng yên, phone không tràn, không pageerror); motion 18/18;
  tương tác 30/30. Build vào `frontend/dist` Dev; asset 8002 khớp; `/health` 200. Không restart, không commit.

### 2026-11-17 — Giao diện "sống" ngoài con robot (frontend, Dev)

- **Thêm:** `frontend/src/motion.js` (helper thuần: `spotlightVars`, `canAnimate`, `taskTransitionName`, `runViewTransition`
  luôn áp dụng update đúng 1 lần kể cả khi API thiếu/lỗi) + `frontend/test/motion.test.js` (7 test, đỏ trước khi có module).
- **Hiệu ứng, đều gắn với trạng thái thật** (`data-agent-state` trên workspace = `resolveAgentPresence`): nền ambient đổi màu/tốc độ
  theo ready/working/answering/error/asking; thẻ tác vụ có đèn theo con trỏ + nghiêng ≤4° + icon nhún; chọn thẻ ↔ preview morph
  bằng View Transitions (`flushSync`); tin nhắn trượt vào; khung nhập "thở" và thanh robot có vạch quét chỉ khi đang xử lý;
  số nguồn bật lên khi đổi. Tắt hết bằng nút chuyển động robot, `prefers-reduced-motion` và forced-colors.
  Sửa `task-rise` fill-mode `both`→`backwards` (giữ `transform` làm mất tilt).
- **Xác minh:** `npm --prefix frontend test` 69/69. Playwright mock, không proxy: motion 18/18 (spotlight, tilt, view transition
  2 chiều, working→ambient/breathe/scan, dừng khi xong, toggle tắt drift, reduced-motion không tilt/drift/transition, phone không
  tràn, không pageerror) + tương tác cũ 30/30. Build vào `frontend/dist` Dev; asset 8002 khớp; `/health` 200. Không restart.
- **Còn lại:** View Transitions chỉ chạy ở Chromium/Safari mới; Firefox đổi trạng thái tức thì (đã có fallback).

### 2026-11-17 — Sửa fallback RAG và prompt echo trong dịch JA (Dev)

- Sửa hai call site `_no_information_answer(mode, question)` thành `_no_information_answer(mode)`; thêm 4 regression tests cho query/stream, MKAC/Research không có thông tin.
- Đưa hướng dẫn dịch query vào system message, user message chỉ chứa câu hỏi. Chặn prompt echo, retry tối đa một lần rồi giữ nguyên nội dung gốc; bỏ prefix hướng dẫn bị lặp. Nếu bản dịch thêm email context không có ở câu gốc, bỏ bản dịch để tránh định tuyến sai vào Gmail. Đây là phòng vệ cho lỗi đã thấy, không bảo đảm phát hiện mọi sai nghĩa của model.
- Xác minh cuối: backend suite 936 passed / 6 dependency deprecation warnings (loại trừ hai integration suites theo hướng dẫn); pyflakes trên file thay đổi và git diff --check sạch.
- Restart riêng app Dev sau preflight project meibook-dev, mount src Dev và port 8002. Health 200 sau startup. Live edited-task JA trên điện thoại: một SSE request 200, 18.2 s, working→ready, không error banner/pageerror/overflow. Lượt sản phẩm đã pass desktop/phone sau sửa fallback; nhánh thiếu thông tin được ép kiểm tra bằng unit tests, không khẳng định live lượt sản phẩm đi qua nhánh đó.
- Hai lượt lặp trước đó là cache hits, không tính là hai lần suy luận độc lập. Không gửi email/lịch, không sửa Production, chưa commit. Các lỗi được ghi trong entry UI bên dưới là kết quả trước khi sửa.


### 2026-11-17 — Màn hình chọn việc ít chữ hơn (frontend, Dev)

- **Yêu cầu:** ít chữ hơn nhưng vẫn tinh tế, người dùng vẫn hiểu. Đã đối chiếu NN/g (nhãn icon, progressive
  disclosure, gợi ý prompt), WCAG 2.2 (1.1.1, 2.2.2, 2.5.3, 2.5.8, 4.1.3), Material chips, W3C i18n.
- **Thay đổi:** robot + một tiêu đề `h1` duy nhất; bỏ dòng MEIBOOK/mode, hint và nút "Bắt đầu" trùng.
  Nút chuyển động thành icon (tên cố định + `aria-pressed`). Thẻ tác vụ: icon theo nghĩa (`iconKey` trong
  `task-launcher.js`, theo ID registry/nhãn VI/đại lượng QA) đặt trên nhãn ngắn, `title` = câu hỏi đầy đủ, số cột theo
  số thẻ. Dòng phụ: "Tự nhập" · "Gợi ý khác (n)" · nút ⓘ. Preview: ô sửa trực tiếp, Enter = thực hiện, Esc = quay lại;
  bỏ nút "Chỉnh câu hỏi" (`editSelectedTask` trong `main.jsx`). `word-break: auto-phrase` cho JA.
- **Xác minh:** `npm --prefix frontend test` 62/62. Playwright trên build tạm, server tĩnh không proxy, mọi API mock:
  30/30 (desktop + phone: chọn thẻ không gửi, Enter gửi đúng 1 request, Esc, ⓘ, tự nhập, không tràn, không lỗi).
  Số khối chữ vùng làm việc MKAC giảm 17 → 9. Build vào `frontend/dist` Dev; asset 8002 khớp; `/health` 200.
- **Live smoke 8002 (guest, JA, model thật qwen3.5:9b):** 6/8 lượt OK (MKAC 部署/残業規定, MES Lot, WMS prepared + query,
  tự nhập câu hỏi nghỉ phép 18 s / 5 nguồn / LiteLLM 200). UI đúng ở mọi lượt (chọn thẻ không gửi, đúng 1 `/query/stream`,
  robot working→ready/error, không tràn, không lỗi console). 2 lượt lỗi do **backend**, chưa sửa:
  1. `src/rag/rag_pipeline.py:318,517` gọi `_no_information_answer(mode, question)` nhưng hàm chỉ nhận `mode` (từ `050becb`)
     → TypeError khi MKAC rơi nhánh `general` (vd. "主な製品", sau khi web search lỗi).
  2. Lớp dịch JA→VI (`translate_query`) trả về nguyên prompt hướng dẫn thay vì câu dịch; prompt có chữ "Gửi email..." nên
     `parse_email_send_command` báo thiếu địa chỉ email → lỗi (câu "MKACの残業申請は誰の承認が必要ですか？").
  CCTVAI ẩn với guest; QA OFFLINE (ConnectTimeout) nên không thử. Script: `/tmp/meibook-ui-audit/live*.mjs`.

### 2026-11-17 — Chuẩn bị remote Qwen routing, TLS/VPN gated

- **Yêu cầu:** mọi tác vụ LLM thường dùng một Qwen3.5-9B Q4_K_M remote; Azure
  `gpt-5.4-mini` chỉ fallback tuần tự khi primary lỗi. Người dùng chốt telemetry
  hardware không được gửi tới remote/cloud và yêu cầu TLS/VPN trước dữ liệu thật.
- **Thay đổi trong Dev:** LiteLLM aliases 4 role đều trỏ đến một provider/env
  primary; bỏ alias cloud/OpenAI thứ hai và alias hardware. Compose bỏ Ollama env,
  host-gateway/proxy dependency/service khỏi topology web; ví dụ env để endpoint/key
  trống. `openai` từ client legacy map về primary; bỏ provider options per caller;
  hardware renderer deterministic và history hardware bị lọc ở API/frontend. Cập nhật
  README, AGENTS, Architecture/Deploy/MesEXPLAIN và model labels. Production không sửa.
- **Xác minh:** 196 tests pass qua `scripts/meibook-python`; Dev Compose config -q
  và git diff --check sạch tại lần chạy. Trước đó YAML syntax check tool Ruby/Node
  thiếu dependency; parse độc lập chưa làm được. Không gọi endpoint remote, không đọc
  `.env.docker`, không restart/rebuild container.
- **Blocker:** chưa có TLS/VPN endpoint được cung cấp/xác minh và key vận hành chưa
  cấu hình. Vì vậy không thể smoke-test wire thật hoặc kết luận model primary hoạt động.
  Ollama service/model trên host vẫn đang active; Dev runtime cũng chưa chuyển.
  Production chưa thay code/config hoặc restart. Không dừng Ollama, không xóa model.
- **Còn phải làm:** kiểm thử LiteLLM 1.83 với upstream giả để chứng minh
  `chat_template_kwargs.enable_thinking` passthrough và strip khỏi Azure, test ảnh,
  JSON/tools/stream/fallback; sửa tài liệu con còn stale và review `git diff` vì file
  Dev có thay đổi chưa commit từ trước. Sau TLS/VPN: user-approved Dev smoke → xin duyệt
  riêng Production rollout và Ollama stop. Không commit.

### 2026-11-17 — Robot hiện diện + sân khấu thích ứng (frontend, Dev)

- **Yêu cầu:** giao diện "sống" theo phương án C (robot dẫn dắt, lựa chọn là chính), chưa có giọng nói.
- **Thêm:** `frontend/src/agent-presence.js` (suy ra trạng thái từ busy/pending/status/timeline/stopped/error/QA clarification,
  không bịa tiến độ), `components/AgentRobot.jsx` (robot SVG 2.5D, mắt/đầu theo con trỏ, chạm và focus; rAF, dừng khi tab ẩn
  hoặc reduced-motion), `components/AgentPresence.jsx` (sân khấu lớn khi chưa có hội thoại, thanh gọn khi đã có; nút tắt chuyển động;
  nhãn trạng thái `role="status"`), `frontend/test/agent-presence.test.js` (6 test, fail trước khi có module).
- **Sửa:** `main.jsx` (thay icon empty state bằng robot, tối đa 4 gợi ý + "Xem thêm", vẫn dùng `sendMessage`/`handleQuickAnswerClick`);
  `styles.css` (khối "Agent presence", sáng/tối, phone, reduced-motion). Không đổi backend/API/SSE.
- **Xác minh:** `npm --prefix frontend test` 50/50 (baseline 44/44). Playwright trên build tạm + server tĩnh không proxy, mọi API mock:
  43/43 (sáng/tối, desktop/phone, ánh mắt theo con trỏ, working→complete, không tràn ngang, không lỗi console, reduced-motion dừng robot).
  Build vào `frontend/dist` Dev; asset 8002 khớp bản build; `/health` 200. Không restart Docker.
- **Sự cố cần biết:** một lần thăm dò Vite dev server không mock đã đi qua proxy mặc định `vite.config.js` tới cổng **8001 (Production)**,
  gọi các GET khởi động và `POST /sessions` (chỉ ghi log, không ghi Qdrant/DB). `vite.config.js` proxy về 8001 là rủi ro cho kiểm thử Dev.
- **Còn lại:** robot chưa có thao tác Có/Không mới (không có contract backend); chưa kiểm thử với model thật trên 8002.

### 2026-11-16 — Giữ bản 9B Q4_K_M để dành VRAM cho BGE/OCR

- **Quyết định mới của người dùng:** mọi vai trò local dùng `qwen3.5:9b`
  (Q4_K_M, file ~6.6 GB), không chọn bản Q8_0 vừa tải. Không xóa model nào.
- **Đã sửa trên Dev:** `litellm_config.yaml` (5 alias cùng tag Q4_K_M),
  `tests/test_litellm_routing_config.py`, `feature_list.json`, `progress.md`.
- **Xác minh:** 25 targeted tests passed; preflight xác nhận project `meibook-dev`,
  container `meibook-litellm-dev`, port `127.0.0.1:4001`, mount config từ Dev.
  Restart chỉ LiteLLM Dev; cả 5 alias smoke pass với fallback tắt, không có reasoning,
  header upstream `:11435`, attempted fallbacks 0. Web 8002 healthy.
  `ollama ps`: chỉ Q4_K_M đang loaded, 100% GPU, ~5.7 GB, context 8192.
  Tổng GPU 9808/16303 MiB (còn ~6.3 GiB trống tại thời điểm kiểm tra).
- **Còn lại:** docs/env examples chưa cập nhật (agent isolation guard chặn, không bypass).
  Production chưa rollout: file config vẫn Q8_0 từ bước trước, container chưa restart.
  Không chạy Production Compose hoặc xóa model. KHÔNG xóa `qwen3.5:9b` như kế hoạch cũ.

### 2026-11-16 — Hợp nhất mọi alias local về một model Ollama (trước quyết định Q4)

- **Mục tiêu:** bỏ Swift-27B ở máy LAN (`192.168.10.188:8998`, kiểm tra ngày
  2026-11-16 thấy connection refused), chạy mọi vai trò local trên một model host.
  `qwen3.5:14b` không tồn tại trên Ollama registry nên dùng `qwen3.5:9b-q8_0`.
- **Thay đổi (cả Dev và Production checkout, chưa commit):** `litellm_config.yaml`
  (5 alias local → `ollama_chat/qwen3.5:9b-q8_0`, `think: false`, bỏ `CHAT_LAN_*`);
  `tests/test_litellm_routing_config.py`, `tests/test_translation_service.py`;
  bỏ `CHAT_LAN_API_BASE` khỏi `docker-compose*.yml`; `.env*.example`; tài liệu
  (`AGENTS.md`, `README.md`, `Markdowns/{ARCHITECTURE,DEPLOY,MesEXPLAIN}.md`).
- **Đã kiểm tra:** pytest routing/model routes/translation 25 passed ở cả 2 checkout;
  `docker compose -f docker-compose.dev.yml config -q` OK; `git diff --check` sạch.
  Model chạy 100% GPU, khoảng 9 GB, context 8192; GPU tổng ~12.9/16.3 GB.
  Recreate `meibook-litellm-dev`: 5 alias trả lời với `fallbacks: []`, header
  `x-litellm-model-api-base` = `:11435`, 0 fallback. Smoke `/query/stream` trên 8002
  (guest, UI tiếng Nhật, mkac) trả lời đầy đủ trong khoảng 23 s, 14/14 request LiteLLM 200.
- **Chưa làm (cần xác nhận):** recreate `meibook-litellm` Production; sau đó mới
  `ollama rm qwen3.5:9b`. Trước khi Production chuyển, Ollama (`MAX_LOADED_MODELS=1`)
  sẽ phải đổi qua lại giữa 2 model khi có request coder của Production.
- **Rủi ro:** Dev và Production dùng chung một slot Ollama và GPU, nên request phải
  xếp hàng. Nếu Ollama tắt, chat/coder fallback sang Azure; hardware summary
  fail-closed. `.env.docker` thật vẫn còn key `CHAT_LAN_*` (không còn được dùng).

### 2026-10-31 — Thiết lập session harness

- **Mục tiêu:** harness cho các phiên agent; không đổi business logic.
- **Thay đổi:**
  - `CLAUDE.md`: thêm mục "Session Harness" (quy trình bắt đầu/kết thúc phiên,
    validation matrix, mô tả hook).
  - `AGENTS.md`: sửa 2 thông tin lỗi thời (`docker-compose.dev.yml` đã được track;
    frontend đã có `npm test`, vẫn chưa có lint).
  - Mới: `feature_list.json`, `progress.md`, `scripts/harness-check.sh`,
    `.claude/hooks/{session-context,guard-environment,guard-files,post-edit-check}.sh`
    và thư viện dùng chung `.claude/hooks/lib-paths.sh`.
  - `.claude/settings.json`: đăng ký thêm SessionStart, PreToolUse, PostToolUse hook.
- **Đã kiểm tra:** baseline ở "Current state"; kết quả kiểm thử hook và
  `scripts/harness-check.sh full` ghi ở dòng dưới.
- **Kết quả xác minh harness (2026-10-31):**
  - `scripts/harness-check.sh full`: 6/6 PASS (whitespace, pyflakes `src/`, frontend
    `node --test`, JSON, pytest 909 passed, `docker compose -f docker-compose.dev.yml
    config -q`). Thêm file có trailing whitespace thì `quick` trả exit 1; mode sai trả exit 2.
  - `guard-environment.sh`: 100/100 case payload khớp (34 allow, 27 deny, 39 ask);
    bộ chạy test được kiểm chéo bằng case cố tình sai. Hook chạy khoảng 10 ms.
  - `guard-files.sh`: 35/35 case khớp (14 allow, 11 deny, 10 ask).
  - `post-edit-check.sh`: báo đúng JSON lỗi, import thừa, trailing whitespace; im lặng
    với file sạch, file ngoài repo, file secret và dữ liệu runtime.
  - `session-context.sh`: 21 dòng trong Dev; không in gì ngoài 2 checkout Meibook.
  - Hook chỉ có hiệu lực từ phiên Claude Code mới (hoặc sau khi reload settings);
    phiên thiết lập không được hook mới bảo vệ.
  - Bộ case test hook nằm ngoài repo (`/tmp/meibook-harness-test/`), chưa thành
    test chính thức.
- **Phát hiện HR (không sửa trong phiên này):** các tên placeholder người dùng
  nghi ngờ không tồn tại trong DB nhân sự; ngược lại phần lớn tên trong fixture
  `tests/test_query_routing.py` trùng nhân viên thật → `hygiene.test_fixture_pii`.
- **Rủi ro:** hook dựa trên regex, chỉ là lưới an toàn; `.claude/` được track nên
  hook sẽ chạy cả ở checkout Production sau khi merge; Codex/agent khác không chạy
  Claude Code hook.
- **Bước tiếp theo:** review và commit theo nhóm; sửa fixture PII; smoke test
  `guard.cross_turn` trên cổng Dev 8002 khi được phép.
