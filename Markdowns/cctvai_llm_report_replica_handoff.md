# CCTVAI Reporting Replica — Tài liệu bàn giao (kết nối & schema)

Tài liệu dành cho team triển khai LLM/công cụ báo cáo đọc dữ liệu CCTVAI. Đây là
bản tóm tắt phần cần dùng — chi tiết hạ tầng/vận hành xem
`docs/features/cctvai/database/cctvai_llm_report_replica.md` (nội bộ team ict).

> **Ghi chú bản sửa 2026-09-03:** các mục đánh dấu ✅ đã được kiểm chứng bằng query
> thật trên replica. Bản bàn giao gốc có 4 điểm sai đã được sửa: port trong ví dụ
> psql, thiếu schema prefix trong query mẫu, thiếu `del_flag`/`LOWER()` khi join, và
> số bảng thực tế. Xem mục 8.

## 1. Đây là gì

Một Postgres **chỉ đọc**, đồng bộ gần như tức thời (thường dưới vài giây) từ
database CCTVAI thật, dành riêng cho việc query/báo cáo — tách khỏi hệ thống đang
vận hành camera/AI thật để không ảnh hưởng production.

**Không phải dữ liệu tức thời tuyệt đối** — đồng bộ bất đồng bộ (asynchronous),
không dùng để ra quyết định nghiệp vụ cần dữ liệu real-time chính xác tới từng
mili-giây.

## 2. Thông tin kết nối

|          | prod             |
| -------- | ---------------- |
| host     | _(hỏi team ict)_ |
| port     | 55434            |
| database | `cctvai`         |
| user     | `cctvai_llm_ro`  |
| password | _(hỏi team ict — không commit vào repo)_ |

Ví dụ kết nối:

```
psql -h <host> -p 55434 -U cctvai_llm_ro -d cctvai
```

**Ràng buộc kết nối** — role này áp dụng sẵn ✅ (đã kiểm chứng):

- Chỉ đọc (`default_transaction_read_only = on`) — mọi câu `INSERT`/`UPDATE`/`DELETE` sẽ bị từ chối.
- `statement_timeout = 30s` — query chạy quá 30 giây sẽ tự bị huỷ.
- Quyền cấp ở **mức cột**, không phải mức bảng — xem mục 4.

## 3. Schema — các bảng đọc được

Tất cả bảng nghiệp vụ nằm trong schema `cctvai`.

> ⚠️ **`search_path` mặc định của role là `"$user", public` — KHÔNG chứa `cctvai`.**
> Schema `public` rỗng, không có bảng nào. Vì vậy query không có prefix sẽ lỗi
> `relation "event_snapshots" does not exist`.
>
> Bắt buộc chọn 1 trong 2 cách:
>
> ```sql
> -- Cách 1: set đầu session (khuyến nghị cho tool/LLM)
> SET search_path TO cctvai, public;
>
> -- Cách 2: prefix đầy đủ trong mọi query
> SELECT ... FROM cctvai.event_snapshots ...
> ```
>
> Hoặc set ngay trong connection string: `options=-csearch_path%3Dcctvai,public`

Replica đồng bộ 8 bảng nghiệp vụ, cộng thêm `schema_migrations` (bảng nội bộ của
migration tool — bỏ qua khi làm báo cáo). Tổng cộng `\dt cctvai.*` trả về 9 dòng.

| Bảng                               | Nội dung                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cctvai_lines`                     | Tuyến/khu vực camera (`line_id`, `line_name`)                                                                                                                                                                                                                                                                                                                          |
| `cctvai_cameras`                   | Master camera — **xem mục 4, một số cột bị chặn**                                                                                                                                                                                                                                                                                                                      |
| `cctvai_violation_types`           | Danh mục loại vi phạm AI (`violation_code`, `violation_name`, `severity`)                                                                                                                                                                                                                                                                                              |
| `cctvai_camera_violation_mappings` | Cặp (camera, loại vi phạm) được phép phát hiện                                                                                                                                                                                                                                                                                                                         |
| `cctvai_line_responsibles`         | `user_id` người phụ trách 1 line — **chỉ là số, không có tên/email, xem mục 5**                                                                                                                                                                                                                                                                                        |
| `event_snapshots`                  | **Bảng trung tâm** — 1 dòng = 1 sự kiện AI phát hiện. Cột đáng chú ý: `camera_id`, `violation_type` (mã tham chiếu tới 2 bảng trên, không phải FK cứng), `detected_time`/`end_time` (**epoch milliseconds**, không phải timestamp — quy đổi bằng `to_timestamp(detected_time/1000.0)`), `record_status` (`recording`/`recorded`), `confidence_rate` (0–1, có thể NULL) |
| `cctvai_notification_outbox`       | Hàng đợi nội bộ gửi notification — không cần dùng cho báo cáo                                                                                                                                                                                                                                                                                                          |
| `cctvai_speaker_outbox`            | Hàng đợi nội bộ điều khiển loa/đèn còi — không cần dùng cho báo cáo                                                                                                                                                                                                                                                                                                    |
| `schema_migrations`                | Bảng nội bộ migration (`version`, `dirty`) — **không dùng cho báo cáo**                                                                                                                                                                                                                                                                                                |

Chi tiết đầy đủ từng cột: `docs/features/cctvai/database/cctvai_database_design.md`
(tài liệu as-built của schema gốc — đúng cho cả replica vì cấu trúc bảng giống hệt
primary).

## 4. Cột bị chặn trên `cctvai_cameras`

**Không dùng `SELECT *` trên bảng này — sẽ bị từ chối** ✅ (đã kiểm chứng: lỗi
`permission denied for table cctvai_cameras`). Phải chỉ định rõ cột:

```sql
SELECT id, camera_id, camera_name, line_ref_id, protocol, description,
       is_active, del_flag, create_date, edit_date, camera_name_translations
FROM cctvai.cctvai_cameras;
```

Các cột `username`, `password`, `endpoint`, `main_cam_url`, `sub_cam_url`,
`speaker_url`, `speaker_topic`, `light_topic` **không đọc được** — đây là thông tin
đăng nhập thiết bị camera vật lý (một số lưu plaintext), cố ý loại khỏi quyền đọc.

**Danh sách cột đọc được của từng bảng** ✅ (đã kiểm chứng qua
`information_schema.column_privileges`):

| Bảng                               | Cột đọc được                                                                                                                                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cctvai_lines`                     | `id, line_id, line_name, description, is_active, del_flag, create_date, edit_date`                                                                                                                              |
| `cctvai_cameras`                   | `id, camera_id, camera_name, line_ref_id, protocol, description, is_active, del_flag, create_date, edit_date, camera_name_translations` — **11/19 cột**                                                          |
| `cctvai_violation_types`           | `id, violation_code, violation_name, severity, description, del_flag, create_date, edit_date`                                                                                                                    |
| `cctvai_camera_violation_mappings` | `camera_ref_id, violation_ref_id, create_date, edit_date`                                                                                                                                                       |
| `cctvai_line_responsibles`         | `line_ref_id, user_id, create_date, edit_date`                                                                                                                                                                  |
| `event_snapshots`                  | `id, camera_id, violation_type, detected_time, end_time, thumbnail_path, thumbnail_metadata, video_path, confidence_rate, created_at, record_status, updated_at`                                                 |
| `cctvai_notification_outbox`       | tất cả cột                                                                                                                                                                                                      |
| `cctvai_speaker_outbox`            | tất cả cột — **lưu ý `speaker_topic`/`light_topic` ĐỌC ĐƯỢC ở bảng này** (chỉ bị chặn trên `cctvai_cameras`)                                                                                                     |

## 5. Không có dữ liệu người dùng (email/tên/phòng ban)

Replica **không** có schema `identity` (nơi lưu user/phòng ban) — chỉ có schema
`cctvai` và `public` (rỗng) ✅. Cột `cctvai_line_responsibles.user_id` chỉ là số ID,
**không tự resolve ra được tên/email**. Nếu cần thông tin người phụ trách, phải gọi
API riêng của `identity-service` (không có trong phạm vi tài liệu này — hỏi team ict
nếu cần).

## 6. Ba cạm bẫy khi sinh SQL (bắt buộc đọc trước khi làm báo cáo)

### 6.1 ⚠️ `del_flag` KHÔNG phải filter tuỳ chọn — nó là cách de-dup duy nhất

Đây là cạm bẫy nguy hiểm nhất của schema này, và **trái ngược trực giác thông thường
về soft delete**.

`camera_id` và `violation_code` **KHÔNG unique** trên toàn bảng. Cùng một mã tồn tại
nhiều dòng, phân biệt bằng `del_flag` ✅:

| `camera_id` | Số dòng | Active | Đã xoá |
| ----------- | ------- | ------ | ------ |
| `cam008`    | 4       | 1      | 3      |
| `cam007`    | 3       | 1      | 2      |
| `cam001`…`cam015` | 2 | 1      | 1      |

Tương tự `violation_code = 'glove_not_changed'` có 2 dòng: một `severity='alert'`
(active) và một `severity='normal'` (đã xoá).

**Chỉ khi lọc `del_flag = FALSE` thì mã mới trở thành khóa duy nhất** ✅:

| Bảng                     | Tổng dòng | Active | `DISTINCT` mã khi active |
| ------------------------ | --------- | ------ | ------------------------ |
| `cctvai_cameras`         | 30        | 15     | 15 ← unique              |
| `cctvai_violation_types` | 17        | 15     | 15 ← unique              |
| `cctvai_lines`           | 3         | 2      | 2 ← unique               |

**Hệ quả: quên `del_flag` thì JOIN bị nhân bản dòng (fan-out), số liệu phồng lên.**
Đo thực tế trên replica ✅ — tổng sự kiện thật là **8.305**:

| Cách viết JOIN | Số dòng trả về | Sai lệch |
| -------------- | -------------- | -------- |
| JOIN cả 3 bảng master, **không** lọc `del_flag` | **11.986** | **+44 %** |
| JOIN riêng `violation_types`, không lọc | 8.782 | +5,7 % |
| JOIN có `AND NOT c.del_flag` / `AND NOT v.del_flag` | **8.305** | ✅ đúng |

Query sai vẫn **chạy thành công, không báo lỗi** — chỉ trả về số lớn hơn sự thật.
Đây là dạng lỗi LLM không thể tự phát hiện.

**Quy tắc bắt buộc:** mọi JOIN tới `cctvai_cameras`, `cctvai_violation_types`,
`cctvai_lines` phải kèm `AND NOT <alias>.del_flag` **ngay trong điều kiện JOIN**
(không phải ở `WHERE`, vì `WHERE` sẽ biến `LEFT JOIN` thành `INNER JOIN`).

### 6.2 Join soft-ref — `event_snapshots` không có FK cứng

`event_snapshots.camera_id` → `cctvai_cameras.camera_id` và
`event_snapshots.violation_type` → `cctvai_violation_types.violation_code` là **tham
chiếu mềm, không có ràng buộc DB**. Thực tế trên replica ✅:

- **26 sự kiện** không khớp camera active nào (camera lạ như `cam002-deepstream`,
  `cam003-test`, hoặc camera đã tháo hẳn).
- Trong đó **1 sự kiện** chỉ khớp được camera đã soft-delete.
- **3 sự kiện** có `violation_type` không khớp danh mục.

→ Dùng `INNER JOIN` sẽ **âm thầm bỏ mất** các dòng này. Báo cáo tổng số sự kiện phải
dùng `LEFT JOIN` + `COALESCE`, và **đếm bằng `count(e.id)` chứ không phải `count(*)`**
để không đếm dòng thừa nếu lỡ fan-out.

Khi chỉ cần kiểm tra tồn tại mà không cần cột từ master, **`EXISTS` an toàn hơn
`JOIN`** vì miễn nhiễm hoàn toàn với fan-out.

Về `LOWER()`: xlsx schema khuyến nghị `LOWER(e.camera_id) = LOWER(c.camera_id)`. Kiểm
chứng cho thấy dữ liệu hiện tại **không có lệch hoa/thường** (kết quả giống hệt khi bỏ
`LOWER()`). Vẫn nên giữ để phòng dữ liệu tương lai, nhưng lưu ý nó **vô hiệu hoá
index** — với 8k dòng thì không sao, cần cân nhắc lại khi dữ liệu lớn.

### 6.3 `detected_time` là epoch milliseconds, không phải timestamp

Luôn quy đổi `to_timestamp(detected_time / 1000.0)`. Cột `end_time` **NULL khi sự
kiện đang diễn ra** — hiện có 166 sự kiện `end_time IS NULL` ✅. Đừng tính thời lượng
mà không lọc NULL.

Khoảng dữ liệu hiện có ✅: **2024-07-23 → hiện tại** (replica bám sát real-time).

## 7. Ví dụ query báo cáo (đã sửa đúng)

Mọi query dưới đây đã kiểm chứng chạy đúng trên replica ✅. Chú ý `AND NOT ….del_flag`
nằm **trong** điều kiện JOIN, và `count(e.id)` thay cho `count(*)`.

```sql
SET search_path TO cctvai, public;

-- Số sự kiện theo mức độ nghiêm trọng, 7 ngày gần nhất.
SELECT COALESCE(vt.severity, '(không rõ)') AS severity, count(e.id) AS total
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_violation_types vt
       ON LOWER(vt.violation_code) = LOWER(e.violation_type)
      AND NOT vt.del_flag
WHERE to_timestamp(e.detected_time / 1000.0) >= now() - interval '7 days'
GROUP BY 1
ORDER BY total DESC;

-- Chi tiết sự kiện gần nhất kèm tên line/camera/loại vi phạm.
-- LEFT JOIN + del_flag trong ON: giữ đủ 8.305 sự kiện, kể cả 26 sự kiện
-- của camera không còn trong master.
SELECT
  COALESCE(c.camera_name, e.camera_id || ' (không có trong master)') AS camera_name,
  l.line_name,
  vt.violation_name,
  vt.severity,
  to_timestamp(e.detected_time / 1000.0) AS detected_at,
  e.record_status,
  e.confidence_rate
FROM cctvai.event_snapshots e
LEFT JOIN cctvai.cctvai_cameras c
       ON LOWER(c.camera_id) = LOWER(e.camera_id) AND NOT c.del_flag
LEFT JOIN cctvai.cctvai_lines l
       ON l.id = c.line_ref_id AND NOT l.del_flag
LEFT JOIN cctvai.cctvai_violation_types vt
       ON LOWER(vt.violation_code) = LOWER(e.violation_type) AND NOT vt.del_flag
ORDER BY e.detected_time DESC
LIMIT 20;

-- Danh sách camera ĐANG hoạt động — 15 dòng.
SELECT c.camera_id, c.camera_name, l.line_name, c.protocol, c.is_active
FROM cctvai.cctvai_cameras c
LEFT JOIN cctvai.cctvai_lines l ON l.id = c.line_ref_id AND NOT l.del_flag
WHERE NOT c.del_flag
ORDER BY l.line_name, c.camera_name;

-- Sự kiện "mồ côi" — camera_id không khớp camera active nào (26 dòng).
-- Dùng EXISTS: miễn nhiễm fan-out.
SELECT e.camera_id, count(e.id) AS events
FROM cctvai.event_snapshots e
WHERE NOT EXISTS (
  SELECT 1 FROM cctvai.cctvai_cameras c
  WHERE LOWER(c.camera_id) = LOWER(e.camera_id) AND NOT c.del_flag
)
GROUP BY 1 ORDER BY events DESC;
```

**Phân bố sự kiện theo loại vi phạm** ✅ (top 5, đã lọc `del_flag`):

| `violation_code`  | Tên               | `severity` | Sự kiện |
| ----------------- | ----------------- | ---------- | ------- |
| `touched_pcb`     | Tay chạm vào bo   | alert      | 4.606   |
| `no_mask`         | Thiếu khẩu trang  | alert      | 1.149   |
| `hand_not_washed` | Chưa rửa tay      | alert      | 709     |
| `no_coat`         | Chưa mặc đồ phòng sạch | alert | 622     |
| `glove_not_changed` | Chưa thay găng tay | alert   | 477     |

## 8. Các điểm đã sửa so với bản bàn giao gốc

| # | Bản gốc                                          | Thực tế (đã kiểm chứng 2026-09-03)                                        |
| - | ------------------------------------------------ | ------------------------------------------------------------------------- |
| 1 | Ví dụ psql dùng `-p 55432`                       | Port đúng là `55434` (bảng thông tin kết nối đúng, ví dụ sai)              |
| 2 | Query mẫu viết `FROM event_snapshots` không prefix | `search_path` không chứa `cctvai` → query gốc **lỗi**. Phải `SET search_path` hoặc prefix |
| 3 | Query mẫu dùng `INNER JOIN`, không lọc `del_flag`  | **Fan-out +44 %** (11.986 dòng thay vì 8.305) vì `camera_id`/`violation_code` không unique khi chưa lọc `del_flag` — xem mục 6.1 |
| 4 | "Không có bảng nào khác ngoài 8 bảng"             | Có thêm `schema_migrations` (9 bảng tổng cộng)                             |
| 5 | Không nói quyền cấp ở mức cột                     | Grant theo **cột**, nên `SELECT *` fail trên `cctvai_cameras`              |
| 6 | Ngụ ý `speaker_topic`/`light_topic` bị chặn hoàn toàn | Chỉ chặn trên `cctvai_cameras`; **đọc được** trên `cctvai_speaker_outbox`  |

## 9. Quy mô dữ liệu (snapshot 2026-09-03) ✅

| Bảng                               | Số dòng |
| ---------------------------------- | ------- |
| `event_snapshots`                  | 8.303   |
| `cctvai_notification_outbox`       | 8.265   |
| `cctvai_speaker_outbox`            | 3.631   |
| `cctvai_camera_violation_mappings` | 239     |
| `cctvai_cameras`                   | 30      |
| `cctvai_violation_types`           | 17      |
| `cctvai_lines`                     | 3       |
| `cctvai_line_responsibles`         | 3       |

Tổng dữ liệu rất nhỏ (~12k dòng nghiệp vụ). Mọi query báo cáo đều nằm thoải mái
trong `statement_timeout = 30s`.
