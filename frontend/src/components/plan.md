# Kế hoạch Refactor `main.jsx`

Hiện tại file `main.jsx` dài hơn 3500 dòng, chứa logic của App, State management (React Hooks), I18n translation, và toàn bộ giao diện cho cả 3 module (HR, MES, Research).

## Các bước chia nhỏ:
1. **Chia nhỏ UI Components** (Không đụng vào State logic ở App component vội để tránh lỗi hồi quy)
   - `components/EmployeeLogin.jsx`: Form nhập mã nhân viên (Lines 2750-2814)
   - `components/ResearchSidebar.jsx`: Thanh bên (Sidebar) cho module Research (Lines 2337-2598)
   - `components/MessageList.jsx`: Vùng hiển thị tin nhắn (Lines 2628-2746)
   - `components/ChatInput.jsx`: Ô nhập liệu và thanh trạng thái (Lines 2953-3178)
   - `components/SourcePreviewDialog.jsx`: Popup xem ảnh nguồn RAG (Lines 3317-3375)

2. **Cách tiếp cận:**
   - Sẽ tách từng component một.
   - Các Component con sẽ là dạng "Dumb Component" hoặc nhận state/props từ `App`.
   - Giữ nguyên CSS class (từ `styles.css`) và không đổi bất kỳ thẻ HTML nào để tránh phá hỏng kết quả của bài audit Mobile UI/UX trước đó.

