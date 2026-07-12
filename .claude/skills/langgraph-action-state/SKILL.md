---
name: langgraph-action-state
description: "Thiết kế đồ thị trạng thái Agent theo mô hình Draft-Confirm-Execute. Dùng khi xây dựng các hành động có tác động ra môi trường ngoài (như gửi email, đặt lịch họp) để đảm bảo an toàn và sự kiểm soát của người dùng."
---

# LangGraph Action-Draft State Machine

## 1. Kiến trúc đồ thị trạng thái
Tách rời quá trình quyết định của LLM và quá trình thực thi hành động thật trên hệ thống. Mọi hành động ghi (write actions) phải đi qua trạng thái chờ duyệt (pending draft).

```
[User Request] 
      │
      ▼
[LLM Intent/Field Extraction Node] ───> [Validate & Create Draft Node]
                                                  │
                                                  ▼
                                       [Save Draft in SQLite]
                                                  │
                                                  ▼
                                        [Return Draft to UI]
                                                  │
                                            (Chờ user duyệt)
                                                  │
                                                  ▼
[Execute Action Node (Calendar/Gmail)] <─── [Confirm API Request]
```

## 2. Schema thiết kế cho Action Store (actions.sqlite)
Lưu trạng thái các hành động trong database để tránh mất dữ liệu khi restart container hoặc reload giao diện:
```sql
CREATE TABLE IF NOT EXISTS pending_actions (
    action_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    action_type TEXT NOT NULL,          -- 'calendar.create_event', 'gmail.send'
    payload TEXT NOT NULL,              -- JSON chứa chi tiết cuộc họp/email
    status TEXT NOT NULL,               -- 'pending', 'confirmed', 'cancelled'
    version INTEGER DEFAULT 1,          -- Chống race condition khi edit ở nhiều tab
    expires_at TEXT NOT NULL,           -- Tự động vô hiệu hóa sau 15-30 phút
    created_at TEXT NOT NULL
);
```

## 3. Luồng tương tác Confirmation trên UI
*   **SSE Event:** Server trả về một payload dạng JSON chứa `action_id`, `action_type` và `payload` của draft thông qua Server-Sent Events.
*   **UI Card Rendering:** Giao diện React nhận diện event và render một form xác nhận (Confirmation Card) trực tiếp trong danh sách tin nhắn.
*   **API Edits:** Cung cấp các endpoint:
    *   `PATCH /actions/{action_id}`: Người dùng sửa trực tiếp tiêu đề, giờ họp, danh sách email người tham dự từ giao diện.
    *   `POST /actions/{action_id}/confirm`: Thực thi hành động thật qua API của Google, đổi trạng thái sang `confirmed`.
    *   `DELETE /actions/{action_id}`: Hủy hành động nháp, đổi trạng thái sang `cancelled`.
*   **Idempotency & Safety:** Khi confirm, client phải gửi kèm `expected_version` của draft. Nếu version trên DB khác version gửi lên, báo lỗi để tránh race condition. Nút bấm trên UI phải disable ngay sau khi nhấn để tránh gửi trùng (double-submit).
