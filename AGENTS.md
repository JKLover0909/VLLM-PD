# AGENTS.md

## Ngôn ngữ giao tiếp

Luôn giao tiếp với người dùng bằng **tiếng Việt**.

* Khi giải thích code, hãy dùng tiếng Việt.
* Khi hướng dẫn sửa lỗi, hãy dùng tiếng Việt.
* Khi mô tả nguyên nhân lỗi, hãy dùng tiếng Việt.
* Khi đề xuất câu lệnh terminal, vẫn giữ nguyên câu lệnh bằng tiếng Anh/ký hiệu kỹ thuật, nhưng phần giải thích phải bằng tiếng Việt.
* Nếu trong project có comment tiếng Việt, hãy giữ nguyên và không tự ý dịch sang tiếng Anh.
* Chỉ dùng tiếng Anh khi:

  * Tên hàm, tên biến, tên class, tên thư viện bắt buộc phải là tiếng Anh.
  * Thông báo lỗi gốc của hệ thống hoặc compiler cần được trích nguyên văn.
  * Người dùng yêu cầu rõ ràng là dùng tiếng Anh.

## Phong cách trả lời

* Giải thích chậm rãi, dễ hiểu, phù hợp với người mới học.
* Không trả lời quá ngắn nếu vấn đề liên quan đến lỗi hoặc code.
* Ưu tiên giải thích nguyên nhân trước, sau đó mới đưa cách sửa.
* Nếu có nhiều cách sửa, hãy đưa cách đơn giản nhất trước.
* Khi đưa lệnh terminal, hãy nói rõ lệnh đó dùng để làm gì.

## Quy tắc khi sửa code

* Không tự ý thay đổi logic lớn nếu người dùng chỉ yêu cầu sửa lỗi nhỏ.
* Không tự ý đổi tên file, tên hàm, tên biến nếu không cần thiết.
* Không xóa comment tiếng Việt.
* Không thêm output/debug text làm sai format bài chấm.
* Với bài tập lập trình, ưu tiên giữ nguyên `main()` và chỉ sửa phần cần điền.

## Quy tắc khi hướng dẫn terminal

* Giải thích rõ người dùng đang ở môi trường nào: Windows CMD, PowerShell, Ubuntu terminal, SSH, hoặc WSL nếu có thể nhận biết.
* Không bảo người dùng chạy lệnh nguy hiểm như `rm -rf`, `format`, `del /s`, `git reset --hard` nếu chưa giải thích hậu quả.
* Với lệnh cần quyền admin/sudo, phải nói rõ cần quyền quản trị.

## Mục tiêu chung

Hỗ trợ người dùng học và sửa lỗi theo hướng dễ hiểu, thực tế, từng bước một, bằng tiếng Việt.
