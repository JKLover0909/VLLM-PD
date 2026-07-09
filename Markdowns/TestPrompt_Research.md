# Bộ Test Prompt Research / DocJP RAG

Tài liệu này là bộ test hỏi đáp cho chế độ **Nghiên cứu tài liệu** của Meibook khi người dùng đã chọn sẵn một trong 4 nhóm tài liệu trên giao diện.

Nguồn khảo sát: `documents/Research/DocJP_md` với 78 file Markdown đã tách theo prefix:

- `【情報】`: Công nghệ thông tin & Bảo mật.
- `【法務】`: Pháp chế & Quản lý rủi ro.
- `【経理】`: Kế toán.
- `【総務】`: Hành chính tổng hợp.

Bộ test không kiểm tra intent routing giữa các nhóm, vì giao diện đã route theo topic. Mỗi test case tập trung kiểm tra retrieval trong đúng nhóm, tính chính xác của câu trả lời, khả năng xử lý điều kiện/ngoại lệ, tình huống thực tế, câu hỏi thiếu thông tin và hallucination guard.

## Tổng quan số lượng

| Nhóm | Số test case |
|---|---:|
| Công nghệ thông tin & Bảo mật | 44 |
| Pháp chế & Quản lý rủi ro | 44 |
| Kế toán | 17 |
| Hành chính tổng hợp | 30 |
| **Tổng** | **135** |

## Quy ước chấm nhanh

- Pass tốt: trả lời đúng nội dung, đúng nguồn, nêu điều kiện/ngoại lệ, không thêm dữ liệu ngoài tài liệu.
- Pass một phần: đúng ý chính nhưng thiếu điều kiện/ngoại lệ hoặc thiếu cấu trúc.
- Fail: trả lời sai nguồn, bịa nội dung, tự suy luận ngoài tài liệu, hoặc không nói rõ khi tài liệu thiếu thông tin.

### TC-INFO-001: Danh sách phần mềm bị cấm

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】（４） 使用禁止と判断されたソフト.md`

- **Câu hỏi kiểm thử:**
  Những phần mềm nào bị cấm sử dụng trong công ty?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu các nhóm/phần mềm bị cấm được tài liệu liệt kê: Winny và các phần mềm chia sẻ file tương tự như Share, WinMX, Winnyp, うたたね, Marie, 新月, AsgumoWeb, RinGOch, Ansem, Speranza, Sparrow, Zigumo, Cabos, Perfect Dark; +Lhaca Deluxe 1.20; CADLUS/P板.COM free software; Baidu IME; GOM Player; DocuWorks 8.0.3; QuickTime for Windows; Skype cá nhân; Splashtop Personal; PDFCrack. Nên nói rõ một số phiên bản/dịch vụ doanh nghiệp như Skype for Business/Microsoft Teams không thuộc đối tượng cấm theo đoạn tương ứng.

- **Bằng chứng cần tìm trong tài liệu:**
  Heading 使用禁止ソフト一覧; các dòng Winny, +Lhaca, Baidu IME, GOM Player, DocuWorks 8.0.3, QuickTime for Windows, Skype, PDFCrack.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `使用禁止ソフト`, `Winny`, `PDFCrack`

### TC-INFO-002: Skype cá nhân và doanh nghiệp

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】（４） 使用禁止と判断されたソフト.md`

- **Câu hỏi kiểm thử:**
  Skype có bị cấm không, còn Skype for Business thì sao?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần phân biệt: Skype/Skype for Windows dành cho cá nhân bị cấm sử dụng nội bộ. Skype for Business là sản phẩm đổi tên từ Lync, khác với Skype cá nhân, có thể giới hạn liên lạc bên ngoài nên không thuộc đối tượng cấm. Dòng dịch vụ doanh nghiệp đã chuyển tên Lync -> Skype for Business -> Microsoft Teams.

- **Bằng chứng cần tìm trong tài liệu:**
  Đoạn Skype（個人向け）; “Skype for Business は Lync の名称変更製品”; “個人向けサービスは社内利用禁止”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Skype`, `Skype for Business`, `Lync`

### TC-INFO-003: Cloud service bị cấm

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】使用禁止と判断したクラウドサービス.md`

- **Câu hỏi kiểm thử:**
  Các dịch vụ cloud/file transfer nào bị cấm dùng?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần liệt kê các dịch vụ tài liệu nêu là cấm hoặc không được phép: ギガファイル便, データ便, FilePost, firestorage, Dropbox, UP300.net, マイポケット, GoogleDrive và Slack. Lý do chính là không quản lý được file/log, trái hướng tăng cường governance/ESG hoặc không phải ứng dụng được phép, thiếu theo dõi log/backup/monitoring.

- **Bằng chứng cần tìm trong tài liệu:**
  Heading 使用禁止と判断したクラウドサービス一覧; các dòng ギガファイル便, Dropbox, GoogleDrive, Slack.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `使用禁止クラウドサービス`, `Dropbox`, `Slack`

### TC-INFO-004: Gửi video điện thoại cho PC

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】使用禁止と判断したクラウドサービス.md`

- **Câu hỏi kiểm thử:**
  Tôi quay video bằng điện thoại và muốn chuyển vào PC, có dùng Gigafile được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không nên dùng Gigafile vì tài liệu xếp ギガファイル便 vào dịch vụ bị cấm. Phương án thay thế trong tài liệu là gửi cho chính mình qua LINE WORKS rồi đăng nhập LINE WORKS trên PC để lưu. Nếu là gửi/nhận với đối tác, nên dùng dịch vụ được quản lý như Box/SharePoint phía đối tác hoặc HENNGE Secure Transfer khi phía Meiko cần cung cấp dịch vụ.

- **Bằng chứng cần tìm trong tài liệu:**
  Đoạn ギガファイル便; “スマホで撮影した動画をPCに送りたい → Lineworksで自分宛に送付”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ギガファイル便`, `Lineworks`, `HENNGE Secure Transfer`

### TC-INFO-005: Xử lý email đáng ngờ

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】迷惑メールが届いた場合の対応.md`

- **Câu hỏi kiểm thử:**
  Khi nhận được email có vẻ là phishing thì tôi phải làm gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần trả lời theo 5 điểm: không trả lời email; không mở file đính kèm; không bấm URL; không nhập tài khoản, thông tin cá nhân, mật khẩu hoặc thẻ; liên hệ bộ phận Hệ thống thông tin để họ bổ sung điều kiện lọc hoặc blacklist sender. Nếu có thể, cung cấp tiêu đề email và thời gian nhận để hỗ trợ điều tra.

- **Bằng chứng cần tìm trong tài liệu:**
  Các mục ①返信しない, ②添付ファイルは開かない, ③URLはクリックしない, ④情報を入力しない, ⑤情報システム部門に連絡.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `迷惑メール`, `返信しない`, `情報システム部門`

### TC-INFO-006: Các loại spam mail

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】迷惑メールが届いた場合の対応.md`

- **Câu hỏi kiểm thử:**
  Tài liệu phân loại những kiểu email rác nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu các loại chính: email nhằm lây nhiễm virus; phishing để đánh cắp thông tin cá nhân; lừa đảo như架空請求/クリック詐欺; quảng cáo/quảng bá không mong muốn; chain mail hoặc email tin giả; các loại khác như giả mạo sender hoặc email rỗng để xác nhận địa chỉ tồn tại.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần 迷惑メールとは; danh sách 1）～6）.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `迷惑メールとは`, `フィッシング`, `チェーンメール`

### TC-INFO-007: AI nào được dùng

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`

- **Câu hỏi kiểm thử:**
  Công ty cho phép dùng loại AI tạo sinh nào từ 2026-01-06?

- **Câu trả lời chuẩn kỳ vọng:**
  Từ 2026-01-06, tài liệu cho phép dùng Copilot do Microsoft cung cấp: Microsoft 365 Copilot Chat bản doanh nghiệp miễn phí qua trình duyệt bằng tài khoản công ty, và Microsoft 365 Copilot bản doanh nghiệp trả phí tích hợp Word/Excel nếu có license và được申請 bằng IT機器申請. Các AI tương tự như ChatGPT, DeepSeek và AI tạo hình/video/âm thanh về nguyên tắc không được dùng, trừ khi hỏi thông tin hệ thống khi có nhu cầu nghiệp vụ.

- **Bằng chứng cần tìm trong tài liệu:**
  Mục １．生成AI利用; “Microsoft社提供のCopilotのみ利用可”; “ChatGPT、DeepSeek等は利用不可”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `生成AI`, `Copilot`, `ChatGPT`

### TC-INFO-008: Copilot không có shield

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`

- **Câu hỏi kiểm thử:**
  Nếu Copilot trên máy tôi không có biểu tượng cái khiên thì có dùng được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Tài liệu nói Copilot được chọn vì có Microsoft Enterprise Data Protection và có biểu tượng “盾のマーク”. Copilot không có biểu tượng khiên, bao gồm một số chức năng tiêu chuẩn Windows 11, bị hạn chế sử dụng. Người dùng cần liên hệ nếu Copilot họ dùng không có biểu tượng này.

- **Bằng chứng cần tìm trong tài liệu:**
  補足事項; “盾のマーク”; “盾のマークが無い Copilot は利用制限します”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `盾のマーク`, `Copilot`, `利用制限`

### TC-INFO-009: AI miễn phí khác

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`

- **Câu hỏi kiểm thử:**
  Gemini hoặc Claude có được dùng cho tài liệu mật không?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu trả lời chuẩn không được tự suy đoán theo kiến thức ngoài. Tài liệu chỉ nêu Copilot của Microsoft được phép và ChatGPT/DeepSeek cùng các AI tương tự không được dùng; không thấy cho phép Gemini hoặc Claude. Vì vậy phải nói tài liệu hiện tại không cung cấp căn cứ cho phép dùng Gemini/Claude với tài liệu mật, và nên hỏi Information Systems/ISMS trước khi dùng.

- **Bằng chứng cần tìm trong tài liệu:**
  Các dòng “Copilotのみ利用可”, “ChatGPT、DeepSeek 等、類似する生成AI は利用不可”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Gemini`, `Claude`, `類似する生成AI`

### TC-INFO-010: USB được phép ghi

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】(連絡文書)データ書き込み制御について_20250312.md`

- **Câu hỏi kiểm thử:**
  USB hoặc ổ cứng ngoài muốn ghi dữ liệu thì điều kiện là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Chỉ các thiết bị USB/SD/HDD ngoài/CD-DVD/smartphone portable media đã được bộ phận Hệ thống thông tin cho phép trước mới được ghi dữ liệu. Thiết bị chưa được phép sẽ chỉ đọc, không ghi. Bộ phận sử dụng phải quản lý thiết bị được phép bằng checklist Y1071別紙2 hàng tháng qua ISMS推進担当者.

- **Bằng chứng cần tìm trong tài liệu:**
  Mục 制御方針 và 月次管理体制; “事前に許可したUSB等のみ書き込み可能”; “読み取り専用”; “Y1071 別紙2”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `USB`, `書き込み制御`, `Y1071`

### TC-INFO-011: USB cá nhân

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】(連絡文書)データ書き込み制御について_20250312.md`

- **Câu hỏi kiểm thử:**
  Nếu USB đã được cấp phép thì tôi có được cắm vào PC cá nhân hoặc dùng cho việc riêng không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Tài liệu nói kể cả thiết bị đã được cấp phép, việc ghi dữ liệu vẫn được log ai/khí nào/dữ liệu gì, và nghiêm cấm kết nối với PC cá nhân hoặc sử dụng riêng tư. Chỉ được dùng cho mục đích nghiệp vụ.

- **Bằng chứng cần tìm trong tài liệu:**
  Mục 使用における留意事項; “私物PCへの接続や私的利用は厳禁”; “業務目的に限定”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `私物PC`, `私的利用`, `ログ収集`

### TC-INFO-012: Email DLP hoạt động thế nào

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  HENNGE Email DLP làm gì khi gửi mail ra ngoài?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu 4 điểm: email gửi ra ngoài bị giữ 5 phút để có thể hủy hoặc gửi ngay; file đính kèm được chuyển thành URL và gửi cho người nhận; có thể xem số lượt tải và vô hiệu hóa link; trường hợp đối tác yêu cầu encrypted ZIP thì có thể cấu hình domain/email cụ thể, nhưng ZIP không được khuyến nghị vì khó virus scan. Mail nội bộ không áp dụng các hành vi này.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần HENNGE Email DLPとは; các mục ①～④.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Email DLP`, `5分間保留`, `URL化`

### TC-INFO-013: Gửi nhầm file ra ngoài

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Tôi vừa gửi nhầm file đính kèm cho đối tác, còn trong 5 phút thì xử lý sao?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần đăng nhập HENNGE One, mở Email DLP, vào 保留トレイ/マイメール, chọn mail đang 一時保留中 và bấm 削除 để mail không gửi đi. Nếu muốn giữ mail lâu hơn có thể bấm 停止, nếu muốn gửi ngay bấm 送信.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần HENNGE Email DLP利用手順（保留メールを削除する）; “削除”, “停止”, “送信”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `保留トレイ`, `削除`, `一時保留中`

### TC-INFO-014: Login HENNGE One

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Đăng nhập HENNGE One ở đâu và dùng tài khoản nào?

- **Câu trả lời chuẩn kỳ vọng:**
  URL đăng nhập là https://ap.ssso.hdems.com/portal/meiko-elec.com/. Người dùng dùng địa chỉ email làm username và mật khẩu đăng nhập Windows làm password.

- **Bằng chứng cần tìm trong tài liệu:**
  Các đoạn HENNGE Oneのログオン先; ユーザー名：メールアドレス; パスワード：Windowsのログオンパスワード.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `HENNGE One`, `メールアドレス`, `Windows`

### TC-INFO-015: File quá 35MB

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Nếu file đính kèm lớn hơn 35MB thì Email DLP có xử lý được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Email DLP hoạt động sau khi Office365 gửi mail, nhưng Office365 chỉ gửi được mail gồm body và attachment tối đa 35MB. Mail vượt ngưỡng này sẽ không đến Email DLP. Với file lớn nên dùng HENNGE Secure Transfer, mỗi lần upload tối đa 2GB; lớn hơn thì chia nhỏ dưới 2GB/lần.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần HENNGE Secure Transferとの違い; “Office365で送信可能なサイズ 35MB”; “1回あたり2GB”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `35MB`, `Secure Transfer`, `2GB`

### TC-INFO-016: HENNGE Secure Transfer

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】取引先との間でサイズの大きいファイルを受け渡すサービス HENNGE Secure Transfer.md`

- **Câu hỏi kiểm thử:**
  HENNGE Secure Transfer dùng để làm gì và truy cập ở đâu?

- **Câu trả lời chuẩn kỳ vọng:**
  Dùng để gửi/nhận dữ liệu có dung lượng email không gửi được. Dịch vụ hỗ trợ chuyển tối đa 2GB mỗi lần, file upload tự xóa sau 2 tuần. Truy cập bằng cách đăng nhập HENNGE One tại https://ap.ssso.hdems.com/portal/meiko-elec.com/ rồi bấm icon HENNGE Secure Transfer; username là email, password là Windows login password.

- **Bằng chứng cần tìm trong tài liệu:**
  Các mục サービスの目的, 新規利用申請, アクセス先.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `HENNGE Secure Transfer`, `2GB`, `2週間`

### TC-INFO-017: Khi nào cần OTP

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE One ワンタイムパスワード.md`

- **Câu hỏi kiểm thử:**
  Khi nào HENNGE One yêu cầu OTP?

- **Câu trả lời chuẩn kỳ vọng:**
  OTP cần khi truy cập từ nơi có global IP thay đổi như chi nhánh/điểm bán ở nước ngoài hoặc công ty liên quan, hoặc khi đi công tác dùng tethering ngoài mạng nội bộ Meiko. Nếu PC đã từng truy cập HENNGE One từ mạng nội bộ thì sẽ có cookie “入場証” và trong 30 ngày không cần OTP; quá 30 ngày không truy cập nội bộ thì cần OTP lại.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần ワンタイムパスワードが必要になる場合; “固定グローバルIP”; “入場証”; “30日間”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ワンタイムパスワード`, `入場証`, `30日間`

### TC-INFO-018: Cài HENNGE Lock

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE One ワンタイムパスワード.md`

- **Câu hỏi kiểm thử:**
  Cách đăng ký HENNGE Lock để nhận OTP như thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Đăng nhập HENNGE One, bấm tên người dùng ở góc phải, chọn OTP設定, bấm OTPを設定する, chọn iOS, làm theo hướng dẫn cài HENNGE Lock, quét QR code bằng HENNGE Lock, khi thấy 完了しました thì đóng lại và xác nhận setting hiện là HENNGE Lock. Khi cần OTP, điện thoại sẽ nhận thông báo và người dùng bấm 許可する.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần ワンタイムパスワードのスマートフォン登録手順.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `HENNGE Lock`, `OTP設定`, `QRコード`

### TC-INFO-019: Bắt đầu LINE WORKS

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】LINE WORKS利用方法 rev1.2.md`

- **Câu hỏi kiểm thử:**
  Lần đầu dùng LINE WORKS thì cần làm gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu được quy trình cơ bản: bộ phận Hệ thống thông tin tạo account LINE WORKS và gửi email mời; người dùng mở email mời, chọn màn hình đăng ký mật khẩu, đặt password rồi hoàn tất. Khi dùng trên PC, truy cập LINE WORKS và đăng nhập bằng ID dạng name.surname@meiko chữ thường cùng password đã đặt.

- **Bằng chứng cần tìm trong tài liệu:**
  Heading はじめに và PCで利用する; “招待メール”; “パスワード登録画面”; “ID 名前．名字@meiko”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `LINE WORKS`, `招待メール`, `パスワード`

### TC-INFO-020: Tạo Teams meeting

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】TeamsのWeb会議作成手順.md`

- **Câu hỏi kiểm thử:**
  Muốn tạo cuộc họp Teams thì tài liệu hướng dẫn tìm ở đâu và cần trả lời theo dạng nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu trả lời nên chỉ ra tài liệu TeamsのWeb会議作成手順 hoặc Web会議システム【Teams】利用手順 và nêu các bước chính nếu retrieved được: dùng Teams/Outlook để tạo lịch họp, nhập người tham dự, thời gian, nội dung, sau đó gửi lời mời. Nếu đoạn chi tiết không xuất hiện trong context, phải nói cần mở đúng tài liệu Teams để xác nhận từng bước.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file TeamsのWeb会議作成手順; keyword Web会議, Teams, 作成手順.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Teams`, `Web会議`, `作成手順`

### TC-INFO-021: VPN

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】②VPN接続方法.md`

- **Câu hỏi kiểm thử:**
  Khi cần truy cập nội bộ từ ngoài công ty thì tài liệu nào hướng dẫn VPN?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất đúng tài liệu ②VPN接続方法 và trả lời theo hướng: đây là tài liệu hướng dẫn cách kết nối VPN. Nếu context chưa có bước chi tiết, không bịa cấu hình, server hay password; yêu cầu người dùng mở đúng hướng dẫn hoặc cung cấp trang được trích.

- **Bằng chứng cần tìm trong tài liệu:**
  File ②VPN接続方法; keyword VPN接続方法.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `VPN`, `接続方法`, `社外`

### TC-INFO-022: Wi-Fi tethering

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】Wi-Fiﾃｻﾞﾘﾝｸﾞ手順.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào hướng dẫn tethering Wi-Fi?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần xác định tài liệu 【情報】Wi-Fiﾃｻﾞﾘﾝｸﾞ手順.md. Câu trả lời không nên tự thêm bước kỹ thuật nếu không có context chi tiết; chỉ nên nói tài liệu này dùng cho thủ tục Wi-Fi tethering và cần làm theo hướng dẫn trong file.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file Wi-Fiﾃｻﾞﾘﾝｸﾞ手順.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Wi-Fi`, `テザリング`, `手順`

### TC-INFO-023: USB tethering cần gì

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】usbﾃｻﾞﾘﾝｸﾞ.md`

- **Câu hỏi kiểm thử:**
  USB tethering cần điều kiện gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Theo tài liệu, để dùng USB tethering cần cài iTunes trên PC. Câu trả lời không nên bịa thêm cấu hình khác nếu tài liệu không cung cấp.

- **Bằng chứng cần tìm trong tài liệu:**
  Dòng “USBテザリングを行うには、i-tuneのPCへのインストールが必要”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `USBテザリング`, `i-tune`, `PC`

### TC-INFO-024: Khôi phục file server

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】ファイルサーバー(192.1.1.33、192.1.1.34)ファイル復旧操作.md`

- **Câu hỏi kiểm thử:**
  Lỡ xóa file trên file server 192.1.1.33/192.1.1.34 thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất tài liệu ファイルサーバー(192.1.1.33、192.1.1.34)ファイル復旧操作. Nếu không có đoạn thao tác cụ thể trong context, trả lời rằng tài liệu này là hướng dẫn khôi phục file server và cần xem các bước trong tài liệu; không tự bịa thao tác restore.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file có 192.1.1.33, 192.1.1.34 và ファイル復旧操作.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ファイルサーバー`, `192.1.1.33`, `復旧`

### TC-INFO-025: Xin quyền file server

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】ファイルサーバ・ノーツDBアクセス権限の申請について.md`

- **Câu hỏi kiểm thử:**
  Muốn xin quyền vào file server hoặc Notes DB thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần chỉ đúng tài liệu ファイルサーバ・ノーツDBアクセス権限の申請について. Câu trả lời nên nêu đây là tài liệu về cách申請 quyền truy cập file server/Notes DB; nếu context không có mẫu form hay approver cụ thể thì không tự thêm.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file và keyword アクセス権限, 申請, ファイルサーバ, ノーツDB.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `アクセス権限`, `申請`, `ノーツDB`

### TC-INFO-026: Web danh bạ

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】PhoneAppli(Web電話帳)の使用方法について.md`

- **Câu hỏi kiểm thử:**
  PhoneAppli dùng để làm gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần xác định PhoneAppli là Web電話帳/danh bạ web. Nếu tài liệu cung cấp thao tác chi tiết trong context thì nêu; nếu không, trả lời ngắn rằng đây là hướng dẫn sử dụng danh bạ web PhoneAppli.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file PhoneAppli(Web電話帳)の使用方法について.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `PhoneAppli`, `Web電話帳`, `使用方法`

### TC-INFO-027: Zoom meeting reservation

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】予約システム「Zoom会議」について.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào nói về hệ thống đặt lịch Zoom meeting?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất file 予約システム「Zoom会議」について và trả lời rằng đây là tài liệu về hệ thống đặt lịch Zoom会議. Không nên lẫn sang Teams nếu câu hỏi nhắc Zoom.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 予約システム「Zoom会議」について.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Zoom会議`, `予約システム`, `会議`

### TC-INFO-028: DLP và Secure Transfer

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`, `【情報】取引先との間でサイズの大きいファイルを受け渡すサービス HENNGE Secure Transfer.md`

- **Câu hỏi kiểm thử:**
  Email DLP và HENNGE Secure Transfer khác nhau thế nào, khi nào dùng cái nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Email DLP hoạt động sau khi gửi email Office365: mail ngoài công ty bị giữ 5 phút, attachment được URL hóa, có thể kiểm tra lượt tải hoặc vô hiệu hóa link. Secure Transfer hoạt động trước khi gửi mail và dùng để gửi/nhận file lớn mà email không gửi được, tối đa 2GB/lần, file tự xóa sau 2 tuần. Nếu file vượt khả năng gửi của Office365 35MB thì dùng Secure Transfer thay vì Email DLP.

- **Bằng chứng cần tìm trong tài liệu:**
  Email DLP: “Office365からメールを送ってからの動作”; Secure Transfer: “メールでは送れないサイズ”, “最大2GB”, “2週間”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Email DLP`, `Secure Transfer`, `35MB`

### TC-INFO-029: Thay thế cloud cấm

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】使用禁止と判断したクラウドサービス.md`, `【情報】取引先との間でサイズの大きいファイルを受け渡すサービス HENNGE Secure Transfer.md`

- **Câu hỏi kiểm thử:**
  Nếu đối tác muốn tôi gửi file lớn qua Dropbox hoặc Google Drive thì nên làm gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Không dùng Dropbox/GoogleDrive vì tài liệu xếp vào cloud service bị cấm. Nếu phía đối tác có dịch vụ được họ quản lý như Box/SharePoint thì ưu tiên dùng dịch vụ có log/security do đối tác quản lý. Nếu Meiko cần cung cấp kênh gửi nhận, dùng HENNGE Secure Transfer, tối đa 2GB/lần và file tự xóa sau 2 tuần.

- **Bằng chứng cần tìm trong tài liệu:**
  Cloud cấm: Dropbox, GoogleDrive; Alternative: Box/SharePoint/HENNGE Secure Transfer; Secure Transfer purpose.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Dropbox`, `GoogleDrive`, `HENNGE Secure Transfer`

### TC-INFO-030: Nhập dữ liệu khách hàng vào AI

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Scenario
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`, `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Tôi muốn nhờ AI tóm tắt email khách hàng rồi gửi ra ngoài, cần lưu ý gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần cảnh báo không nhập thông tin khách hàng/cơ mật vào AI không được phép. Chỉ Copilot có Enterprise Data Protection/biểu tượng khiên mới được phép theo giới hạn; ChatGPT/DeepSeek và AI tương tự không được dùng. Khi gửi mail ra ngoài, Email DLP sẽ giữ 5 phút và URL hóa attachment nhưng không thay thế trách nhiệm kiểm tra nội dung trước khi gửi.

- **Bằng chứng cần tìm trong tài liệu:**
  AI: “顧客に関する情報” không gửi vào AI, Copilot only; DLP: 5分保留, 添付URL化.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `生成AI`, `顧客情報`, `Email DLP`

### TC-INFO-031: Dịch vụ này có dùng được không

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】使用禁止と判断したクラウドサービス.md`

- **Câu hỏi kiểm thử:**
  Dịch vụ này có dùng được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu tên dịch vụ và mục đích sử dụng. Câu trả lời chuẩn phải hỏi lại: “Bạn muốn kiểm tra dịch vụ nào, URL hoặc tên cụ thể là gì, dùng để gửi file, chat hay họp?” Sau đó mới đối chiếu với danh sách dịch vụ bị cấm và alternative trong tài liệu.

- **Bằng chứng cần tìm trong tài liệu:**
  Cần retrieve danh sách 使用禁止クラウドサービス nhưng không được đoán khi thiếu tên dịch vụ.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `使用可否`, `サービス名`, `URL`

### TC-INFO-032: Phần mềm không có trong danh sách

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】（４） 使用禁止と判断されたソフト.md`

- **Câu hỏi kiểm thử:**
  Notion có bị cấm trong danh sách phần mềm không?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu tài liệu không nhắc Notion trong danh sách phần mềm bị cấm thì không được kết luận Notion được phép hoặc bị cấm. Cần nói danh sách hiện truy xuất không thấy Notion, nên cần hỏi Information Systems để xác nhận chính thức trước khi dùng.

- **Bằng chứng cần tìm trong tài liệu:**
  Danh sách 使用禁止ソフト không có Notion.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Notion`, `使用禁止ソフト`, `情報システム`

### TC-INFO-033: Phần mềm hạn chế phân phối

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】（１） 配布制限ソフト.md`

- **Câu hỏi kiểm thử:**
  Phần mềm thuộc nhóm 配布制限ソフト thì cần làm thủ tục gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần trả lời rằng phần mềm nhóm 配布制限ソフト là phần mềm hạn chế phân phối, việc導入 cần申請 qua IT機器申請 theo tài liệu. Không nên gán thành “cấm tuyệt đối” nếu tài liệu chỉ nói hạn chế phân phối.

- **Bằng chứng cần tìm trong tài liệu:**
  Dòng “導入手続きは、『IT機器申請』にて申請を行う”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `配布制限ソフト`, `IT機器申請`, `導入手続き`

### TC-INFO-034: Phần mềm quản lý hạn chế

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】（２） 管理制限ソフト.md`

- **Câu hỏi kiểm thử:**
  管理制限ソフト khác 使用禁止ソフト ở điểm nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu trả lời cần phân biệt: 管理制限ソフト là nhóm phần mềm cần quản lý/hạn chế theo quy định riêng, còn 使用禁止ソフト là phần mềm bị cấm dùng. Nếu context không cung cấp danh sách cụ thể thì không được tự liệt kê; chỉ so sánh dựa theo tên nhóm và yêu cầu mở đúng tài liệu.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 管理制限ソフト và 使用禁止ソフト.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `管理制限ソフト`, `使用禁止ソフト`, `制限`

### TC-INFO-035: GO-GLOBAL account

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】GO-GLOBALのアカウント申請・管理方法について.md`

- **Câu hỏi kiểm thử:**
  Muốn xin tài khoản GO-GLOBAL thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất GO-GLOBALのアカウント申請・管理方法について và trả lời rằng tài liệu này hướng dẫn申請 và quản lý tài khoản GO-GLOBAL. Nếu context không có form/approver cụ thể, không bịa thêm.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file GO-GLOBALのアカウント申請・管理方法について.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `GO-GLOBAL`, `アカウント申請`, `管理方法`

### TC-INFO-036: Webex Calling

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】(Meiko-UC)WebexCalling_固定電話ユーザ操作マニュアル_Rev1.0.pptx_20240119.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào hướng dẫn thao tác điện thoại cố định Webex Calling?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần trỏ tới tài liệu (Meiko-UC)WebexCalling_固定電話ユーザ操作マニュアル. Nếu context có thao tác thì nêu; nếu không, trả lời rằng đây là manual user cho điện thoại cố định Webex Calling.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file WebexCalling_固定電話ユーザ操作マニュアル.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `WebexCalling`, `固定電話`, `操作マニュアル`

### TC-INFO-037: IP phone change

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】(連絡文書)IP固定電話の変更について_20250722.md`

- **Câu hỏi kiểm thử:**
  Có tài liệu nào nói về thay đổi IP fixed phone không?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất đúng file IP固定電話の変更について_20250722 và nêu đây là thông báo liên quan đến thay đổi IP fixed phone. Không lẫn sang Webex Calling manual nếu câu hỏi hỏi thông báo thay đổi.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file IP固定電話の変更について_20250722.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `IP固定電話`, `変更`, `連絡文書`

### TC-INFO-038: Kết nối nhà cung cấp ngoài

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】設備機器と外部業者接続の注意喚起について.md`

- **Câu hỏi kiểm thử:**
  Khi nhà cung cấp ngoài muốn kết nối vào thiết bị trong công ty thì cần lưu ý tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất file 設備機器と外部業者接続の注意喚起について. Câu trả lời nên nói đây là tài liệu cảnh báo/lưu ý về kết nối thiết bị với vendor bên ngoài; nếu context không có điều kiện chi tiết, không bịa quy trình phê duyệt.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 設備機器と外部業者接続の注意喚起について.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `設備機器`, `外部業者`, `接続`

### TC-INFO-039: Đọc email đáng ngờ cụ thể

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】迷惑メールが届いた場合の対応.md`

- **Câu hỏi kiểm thử:**
  Đây là email tôi nhận được, bạn xác nhận giúp có phải phishing 100% không?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu người dùng không đưa nội dung email cụ thể thì phải hỏi lại. Nếu có nội dung nhưng tài liệu chỉ cung cấp dấu hiệu và hướng xử lý, không nên khẳng định 100% nếu thiếu bằng chứng; cần khuyên không trả lời, không mở attachment/URL, không nhập thông tin, và liên hệ Information Systems với title/thời gian nhận.

- **Bằng chứng cần tìm trong tài liệu:**
  迷惑メール対応 nêu quy tắc xử lý, không phải công cụ xác minh tuyệt đối từng email.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `phishing`, `迷惑メール`, `確認`

### TC-INFO-040: Copilot policy tổng hợp

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`, `【情報】連絡書_Copilotの利用について_20260205.md`

- **Câu hỏi kiểm thử:**
  Tóm tắt chính sách Copilot/AI tạo sinh trong công ty.

- **Câu trả lời chuẩn kỳ vọng:**
  Cần tổng hợp: từ 2026-01-06 chỉ Microsoft Copilot doanh nghiệp được phép theo giới hạn; bản miễn phí Copilot Chat dùng bằng tài khoản công ty trên browser, bản Microsoft 365 Copilot trả phí cần license/IT機器申請; ChatGPT/DeepSeek và AI tương tự không dùng; không nhập thông tin mật/khách hàng/cá nhân; Copilot phải có dấu khiên Enterprise Data Protection; nội dung AI sinh ra vẫn phải được kiểm tra.

- **Bằng chứng cần tìm trong tài liệu:**
  Các đoạn Copilot only, ChatGPT/DeepSeek不可, 盾のマーク, 注意事項.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Copilot`, `生成AI`, `盾のマーク`

### TC-INFO-041: Printer network

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【情報】ネットワークに繋がっている複合機・プリンタの設定方法.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào hướng dẫn cài máy in/máy photocopy trong mạng?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất file ネットワークに繋がっている複合機・プリンタの設定方法 và nói đây là tài liệu hướng dẫn thiết lập multifunction printer/printer kết nối mạng.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file ネットワークに繋がっている複合機・プリンタの設定方法.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `複合機`, `プリンタ`, `ネットワーク`

### TC-INFO-042: Đối tác đòi ZIP mã hóa

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Đối tác yêu cầu không nhận link mà phải nhận ZIP mã hóa thì làm thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nói ZIP mã hóa không được khuyến nghị vì không virus-check được, nhưng nếu đối tác yêu cầu thì quản trị viên có thể cấu hình domain hoặc email cụ thể vào ZIP暗号化宛先. Người dùng phải nhờ bộ phận Hệ thống thông tin thêm vào danh sách; domain public như gmail.com/yahoo.co.jp không được đăng ký vì dùng chung nhiều công ty/cá nhân.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần 暗号化ZIPの指定; “管理者側の設定”; “＠gmail.comや＠yahoo.co.jpなどは登録不可”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `暗号化ZIP`, `ZIP暗号化宛先`, `gmail.com`

### TC-INFO-043: Gửi file không rõ size

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【情報】HENNGE Email DLP.md`

- **Câu hỏi kiểm thử:**
  Tôi muốn gửi file lớn cho đối tác, dùng email hay Secure Transfer?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần hỏi thêm dung lượng file và có gửi qua Office365 được không. Nếu tổng mail <=35MB thì Email DLP có thể xử lý sau khi gửi; nếu quá 35MB hoặc cần gửi/nhận file lớn thì dùng HENNGE Secure Transfer, tối đa 2GB/lần.

- **Bằng chứng cần tìm trong tài liệu:**
  Email DLP: 35MB; Secure Transfer: 最大2GB.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `file lớn`, `35MB`, `2GB`

### TC-INFO-044: Policy ngoài Nhật

- **Nhóm tài liệu đang test:** Công nghệ thông tin & Bảo mật
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【情報】20251201連絡書_生成AIの利用について.md`

- **Câu hỏi kiểm thử:**
  Chính sách AI này áp dụng ngay cho toàn bộ chi nhánh nước ngoài không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được trả lời quá phạm vi. Tài liệu nói vì Copilot dựa trên Microsoft365 nên giới hạn này áp dụng cho các cơ sở trong nước, còn cơ sở nước ngoài sẽ triển khai tuần tự từ kỳ sau. Cần nêu rõ tài liệu không cung cấp lịch chi tiết cho từng chi nhánh nước ngoài.

- **Bằng chứng cần tìm trong tài liệu:**
  Mục ４）国内拠点のみ対象; 海外拠点は来期以降、順次.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `国内拠点`, `海外拠点`, `順次`

### TC-LEGAL-001: URL 3rdWATCH/anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】1-1.サービスへのログイン方法.md`

- **Câu hỏi kiểm thử:**
  URL đăng nhập dịch vụ anpi/3rdWATCH là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  URL là https://bcp.myrescue.net/anpi/usr. Để đăng nhập cần 顧客コード, ログインID và パスワード; nếu quên login ID/password thì dùng link “ログインID、パスワードがわからない方はこちら”.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 2; “サービスのURL”; “顧客コード”, “ログインID”, “パスワード”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `bcp.myrescue.net`, `ログインID`, `パスワード`

### TC-LEGAL-002: Mã khách hàng mặc định

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】0_安否確認サービスで出来る事と応答操作について_250916.md`

- **Câu hỏi kiểm thử:**
  Mã khách hàng và login ID mặc định của hệ thống anpi là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu nêu mã chung toàn công ty là meiko6001. Login ID là mã nhân viên, bỏ số 0 đầu; ví dụ 001234 thì nhập 1234. Password mặc định là Meiko_社員番号, cũng bỏ số 0 đầu, ví dụ Meiko_1234. Người dùng nên đổi password sau đó.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 4; “meiko6001”; “ログインID 社員番号”; “接頭0は不要”; “Meiko_社員番号”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `meiko6001`, `社員番号`, `Meiko_`

### TC-LEGAL-003: Đăng nhập dịch vụ

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】1-1.サービスへのログイン方法.md`

- **Câu hỏi kiểm thử:**
  Cách đăng nhập dịch vụ anpi từng bước?

- **Câu trả lời chuẩn kỳ vọng:**
  Chuẩn bị 顧客コード, ログインID, パスワード; mở URL https://bcp.myrescue.net/anpi/usr; nhập 3 thông tin này và bấm ログイン. Khi đăng nhập thành công sẽ thấy マイページ; menu hiển thị tùy quyền.

- **Bằng chứng cần tìm trong tài liệu:**
  4 Steps; page 1-3.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ログイン`, `マイページ`, `顧客コード`

### TC-LEGAL-004: Đăng ký email anpi bằng browser

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】2-1.メールアドレスの新規登録 - 方法A：ブラウザ画面からの登録.md`

- **Câu hỏi kiểm thử:**
  Tôi muốn đăng ký email nhận thông báo an toàn bằng trình duyệt thì làm thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần chuẩn bị 顧客コード, ID/password専用 cho đăng ký email, số nhân viên và tên. Trước tiên cho phép domain myrescue.net trong spam filter. Mở login URL, chọn “メールアドレスを登録する方はこちら”, đăng nhập bằng ID/password đăng ký chung, nhập số/tên, nhập email và email xác nhận, bấm確認/OK. Chờ email xác nhận, bấm URL trong mail để hoàn tất; nếu không nhận được mail thì kiểm tra spam và allow myrescue.net rồi làm lại.

- **Bằng chứng cần tìm trong tài liệu:**
  11 Steps; page 1-6; “myrescue.net”; “登録確認メール”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `メールアドレス`, `新規登録`, `myrescue.net`

### TC-LEGAL-005: ID cá nhân hay ID chung

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】2-1.メールアドレスの新規登録 - 方法A：ブラウザ画面からの登録.md`

- **Câu hỏi kiểm thử:**
  Khi đăng ký email ban đầu, tôi dùng login ID cá nhân hay ID đăng ký chung?

- **Câu trả lời chuẩn kỳ vọng:**
  Phải dùng “メールアドレス登録専用” login ID và password dùng chung toàn công ty, không dùng ID/password cá nhân. Tài liệu nhấn mạnh đây là ID/password đăng ký riêng và là thông tin chung cho toàn user.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 3; “個人のログインIDとパスワードではなく”; “登録専用のIDとパスワード”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `メールアドレス登録専用`, `個人のログインID`, `共通`

### TC-LEGAL-006: Đổi email anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】2-2.メールアドレスの変更.md`

- **Câu hỏi kiểm thử:**
  Đã đăng ký email rồi, muốn đổi email trong anpi thì làm sao?

- **Câu trả lời chuẩn kỳ vọng:**
  Đăng nhập bằng 顧客コード, ログインID, パスワード; mở 設定変更; chọn メールアドレス変更; nhập email cần thêm hoặc đổi; bấm 変更確認; kiểm tra nội dung và bấm 変更. Sau đó nhận email thông báo hoàn tất. Trước khi làm nên allow domain myrescue.net; nếu mail không tới thì kiểm tra spam/allowlist rồi đăng ký lại.

- **Bằng chứng cần tìm trong tài liệu:**
  9 Steps; page 1-5.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `メールアドレス変更`, `設定変更`, `変更確認`

### TC-LEGAL-007: Phản hồi anpi bằng email

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`

- **Câu hỏi kiểm thử:**
  Khi nhận email xác nhận an toàn thì phản hồi thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Mở email 安否確認/非常呼集, bấm URL trong mail trong vòng 7 ngày/168 giờ tính từ thời điểm khởi động, chọn hoặc nhập tình trạng hiện tại, bấm 登録. Nếu có 家族安否 thì chọn tình trạng gia đình, nhưng với 非常呼集 thì lựa chọn 家族安否 không hiển thị. Khi hoàn tất sẽ có màn hình trạng thái đã đăng ký.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 2-3; “URLの有効期限は起動時刻から7日間（168時間）”; “登録”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `安否確認メール`, `168時間`, `登録`

### TC-LEGAL-008: Domain cần allow

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`

- **Câu hỏi kiểm thử:**
  Để nhận mail anpi thì phải allow domain nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Phải cấu hình bộ lọc spam của điện thoại/mail để cho phép domain myrescue.net. Nếu bị spam filter chặn hoặc vào spam folder thì có thể không nhận được mail xác nhận/anpi.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 1; “myrescue.netを許可ドメインとして設定”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `myrescue.net`, `迷惑メールフィルタ`, `許可ドメイン`

### TC-LEGAL-009: Không nhận được email anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-2.安否確認・非常呼集に画面からログインして応答する方法.md`

- **Câu hỏi kiểm thử:**
  Nếu tôi không nhận được email anpi thì có thể phản hồi trên màn hình như thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần đăng nhập dịch vụ bằng 顧客コード, ログインID, パスワード; trên マイページ chọn ユーザ案件履歴一覧; chọn 案件No. của case cần phản hồi; bấm 状態修正; chọn/nhập trạng thái hiện tại và bấm 登録. Nếu là 家族安否 thì chọn thêm trạng thái gia đình, nhưng 非常呼集 không hiển thị lựa chọn này.

- **Bằng chứng cần tìm trong tài liệu:**
  7 Steps; “メールが受信できない場合”; “ユーザ案件履歴一覧”; “状態修正”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ユーザ案件履歴一覧`, `状態修正`, `メールが受信できない`

### TC-LEGAL-010: 案件履歴一覧 khác gì

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-2.安否確認・非常呼集に画面からログインして応答する方法.md`

- **Câu hỏi kiểm thử:**
  Khi tự đăng ký trạng thái, nên chọn 案件履歴一覧 hay ユーザ案件履歴一覧?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu tự đăng ký tình trạng của chính mình thì chọn ユーザ案件履歴一覧. Người có quyền 確認権限 trở lên có thể thấy 案件履歴一覧, nhưng để đăng ký trạng thái cá nhân thì dùng ユーザ案件履歴一覧.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 2; “自分の状況を登録する場合はユーザ案件履歴一覧を選択”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `案件履歴一覧`, `ユーザ案件履歴一覧`, `確認権限`

### TC-LEGAL-011: Thời hạn URL anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`

- **Câu hỏi kiểm thử:**
  URL trong email anpi có hiệu lực bao lâu?

- **Câu trả lời chuẩn kỳ vọng:**
  URL trong email có hiệu lực 7 ngày, tức 168 giờ kể từ thời điểm khởi động. Quá hạn thì không truy cập được.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 2; “URLの有効期限は起動時刻から7日間（168時間）”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `URL`, `7日間`, `168時間`

### TC-LEGAL-012: Quên login/password

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】2-4.ログインIDの取り寄せとパスワードの再発行.md`

- **Câu hỏi kiểm thử:**
  Quên login ID hoặc password anpi thì xử lý thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất tài liệu 2-4 về lấy lại login ID và cấp lại password. Câu trả lời nên hướng người dùng dùng link “ログインID、パスワードがわからない方はこちら” trên màn hình login hoặc làm theo tài liệu 2-4. Không tự cấp password nếu tài liệu không cho phép.

- **Bằng chứng cần tìm trong tài liệu:**
  Link từ tài liệu 1-1 và các file 2-4.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ログインID`, `パスワード再発行`, `取り寄せ`

### TC-LEGAL-013: Thông tin user anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】2-8.ユーザ情報の確認.md`

- **Câu hỏi kiểm thử:**
  Màn hình user information của anpi cho xem những gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu nêu có thể xác nhận thông tin như login ID, password, số, tên, phòng ban, chức vụ, base, quyền, điều kiện gửi email anpi, 居住地, 全社権限, 部署権限 và 安否起動設定. Cần đăng nhập trước khi xem.

- **Bằng chứng cần tìm trong tài liệu:**
  File 2-8; các dòng “ログインID”, “部署”, “役職”, “安否起動設定”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `ユーザ情報`, `部署`, `安否起動設定`

### TC-LEGAL-014: Xem thông tin khủng hoảng

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-1.危機管理情報を画面で確認する.md`

- **Câu hỏi kiểm thử:**
  Muốn xem thông tin quản lý khủng hoảng trên màn hình thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất tài liệu 4-1.危機管理情報を画面で確認する. Nếu context không có bước chi tiết, chỉ trả lời đây là manual kiểm tra thông tin crisis management trên màn hình và yêu cầu mở đúng tài liệu để xem thao tác.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-1.危機管理情報を画面で確認する.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `危機管理情報`, `画面`, `確認`

### TC-LEGAL-015: Mail dự báo thời tiết

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-2.天気予報メールの受信設定.md`

- **Câu hỏi kiểm thử:**
  Có thể cài nhận email dự báo thời tiết không?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất tài liệu 4-2.天気予報メールの受信設定 và trả lời rằng có manual cài đặt nhận mail dự báo thời tiết. Nếu không retrieve được bước chi tiết thì không bịa menu, chỉ nêu cần theo tài liệu này.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-2.天気予報メールの受信設定.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `天気予報メール`, `受信設定`, `4-2`

### TC-LEGAL-016: Mail động đất

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-3.地震情報メールの受信設定.md`

- **Câu hỏi kiểm thử:**
  Muốn nhận email thông tin động đất thì tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần trỏ đúng tài liệu 4-3.地震情報メールの受信設定. Không lẫn với 避難情報 hoặc 鉄道運行 nếu câu hỏi chỉ nói động đất.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-3.地震情報メールの受信設定.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `地震情報メール`, `受信設定`, `地震`

### TC-LEGAL-017: Mail vận hành đường sắt

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-4.鉄道運行情報メールの受信設定.md`

- **Câu hỏi kiểm thử:**
  Cài email thông tin vận hành tàu thì tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 4-4.鉄道運行情報メールの受信設定 và trả lời đây là tài liệu cài nhận mail thông tin vận hành đường sắt.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-4.鉄道運行情報メールの受信設定.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `鉄道運行情報`, `メール`, `受信設定`

### TC-LEGAL-018: Mail sơ tán

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-5.避難情報メールの受信設定.md`

- **Câu hỏi kiểm thử:**
  Cài email thông tin sơ tán thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 4-5.避難情報メールの受信設定 và nói đây là tài liệu cài nhận thông tin sơ tán. Không tự thêm vùng/điều kiện nếu context không có.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-5.避難情報メールの受信設定.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `避難情報メール`, `受信設定`, `避難`

### TC-LEGAL-019: Dừng mail thông tin

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】4-6.情報メールの停止・配信設定の削除.md`

- **Câu hỏi kiểm thử:**
  Muốn dừng hoặc xóa thiết lập nhận mail thông tin thì làm thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất tài liệu 4-6.情報メールの停止・配信設定の削除 và trả lời rằng tài liệu này hướng dẫn dừng mail thông tin hoặc xóa setting phân phối. Nếu context không đủ bước thì không bịa menu cụ thể.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 4-6.情報メールの停止・配信設定の削除.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `情報メール`, `停止`, `配信設定削除`

### TC-LEGAL-020: Đăng ký liên hệ gia đình

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】5-1.家族安否（オプション）：家族の連絡先登録と削除.md`

- **Câu hỏi kiểm thử:**
  Có thể đăng ký liên hệ gia đình trong anpi không?

- **Câu trả lời chuẩn kỳ vọng:**
  Có tài liệu option 家族安否 về đăng ký/xóa liên hệ gia đình. Cần trả lời theo hướng đây là chức năng optional; nếu doanh nghiệp đang dùng 家族安否 thì làm theo manual 5-1 để đăng ký hoặc xóa liên hệ gia đình.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 5-1 家族安否 家族の連絡先登録と削除.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `家族安否`, `連絡先登録`, `削除`

### TC-LEGAL-021: Ghi family bulletin

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】5-2.家族安否（オプション）：家族掲示板への書き込み.md`

- **Câu hỏi kiểm thử:**
  Sau thảm họa, tôi muốn nhắn tin cho gia đình qua hệ thống thì tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 5-2.家族掲示板への書き込み và nói đây là manual ghi vào family bulletin nếu chức năng 家族安否 được dùng. Không cam kết chức năng có bật nếu tài liệu chỉ nói option.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 5-2 家族掲示板への書き込み.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `家族掲示板`, `書き込み`, `家族安否`

### TC-LEGAL-022: Gửi request mail cho gia đình

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】5-3.家族安否（オプション）：要請メールの送信.md`

- **Câu hỏi kiểm thử:**
  Muốn gửi mail yêu cầu gia đình phản hồi an toàn thì hỏi tài liệu nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 5-3.家族安否：要請メールの送信. Câu trả lời phải nêu đây là chức năng optional 家族安否 và hướng dẫn nằm trong tài liệu 5-3.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 5-3 要請メールの送信.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `要請メール`, `家族安否`, `送信`

### TC-LEGAL-023: Báo tình trạng gia đình

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】5-4.家族安否（オプション）：家族の安否状況を所属組織に報告する.md`

- **Câu hỏi kiểm thử:**
  Báo tình trạng an toàn của gia đình cho tổ chức như thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 5-4.家族の安否状況を所属組織に報告する. Nếu context không có bước chi tiết, nói rõ tài liệu này là manual báo cáo tình trạng gia đình cho tổ chức trong option 家族安否.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 5-4 家族の安否状況を所属組織に報告する.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `家族の安否状況`, `所属組織`, `報告`

### TC-LEGAL-024: Domain nhận mail

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】6.補足：メール受信許可設定で指定するドメインと設定方法.md`

- **Câu hỏi kiểm thử:**
  Domain nào phải cho phép để nhận mail anpi?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu domain myrescue.net, liên kết tới tài liệu bổ sung về cấu hình allow domain nhận mail.

- **Bằng chứng cần tìm trong tài liệu:**
  File 6.補足 và các tài liệu 2-1/3-1 đều nhắc myrescue.net.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `myrescue.net`, `メール受信許可`, `ドメイン`

### TC-LEGAL-025: Đóng gáy hợp đồng bằng giấy

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】契約書の製本方法.md`

- **Câu hỏi kiểm thử:**
  Hợp đồng nhiều trang thì đóng gáy bằng giấy như thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu các bước chính: cắt giấy tape theo chiều dài hợp đồng, gấp ba tạo nếp, đặt mép trái tape trùng mép hợp đồng, gập và bôi keo, dán mặt trước, lật mặt sau, bôi keo phần còn lại và dán hoàn tất. Hợp đồng nhiều trang thường cần đóng gáy và đóng dấu giáp lai ở bìa trước/sau.

- **Bằng chứng cần tìm trong tài liệu:**
  契約書を製本する方法～紙使用～; “紙テープ”; “割印は表紙、裏表紙”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `契約書`, `製本`, `紙テープ`

### TC-LEGAL-026: Đóng gáy bằng tape chuyên dụng

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】契約書の製本方法.md`

- **Câu hỏi kiểm thử:**
  Nếu dùng製本テープ để đóng hợp đồng thì quy trình chính là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cắt製本テープ dài hơn chiều dài hợp đồng khoảng 1cm trên/dưới, gập theo ranh giới protective tape, cố định với mép hợp đồng, lật mặt sau, bóc một bên protective tape và dán, cắt phần thừa, gập phần keo vào mặt trước, bóc phần còn lại và dán hoàn tất. Đóng dấu giáp lai ở bìa trước và sau.

- **Bằng chứng cần tìm trong tài liệu:**
  契約書を製本する方法～製本テープ使用～; “上下1cm程度長く”; “割印”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `製本テープ`, `1cm`, `割印`

### TC-LEGAL-027: In hợp đồng nộp pháp chế

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`

- **Câu hỏi kiểm thử:**
  Khi nộp hợp đồng cho pháp chế thì in thế nào theo số trang?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu hợp đồng A4 có 1-2 trang: in A3 một tờ 2-up hoặc A4 một tờ hai mặt. Nếu 3-4 trang: in A3 một tờ 2-up hai mặt. Nếu từ 5 trang trở lên: đóng ghim/đóng gáy theo file 製本方法.pdf.

- **Bằng chứng cần tìm trong tài liệu:**
  Phần 【契約書の印刷について】 trong 捺印管理システム.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `契約書の印刷`, `A3`, `5枚以上`

### TC-LEGAL-028: Các menu hệ thống con dấu

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`

- **Câu hỏi kiểm thử:**
  捺印管理システム có các menu chính nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nêu các menu nhóm general user/approver: 申請登録, 承認・決裁, 捺印処理, 申請検索, 一時保存呼出, 申請取下, パスワード変更. Ngoài ra có khu vực quản trị: 受付, 捺印受付, メンテナンス, 申請検索, ユーザー登録, 共通操作設定 và khu vực 業務部用 gồm 承認, 申請検索.

- **Bằng chứng cần tìm trong tài liệu:**
  Top page menu của 捺印管理システム.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `捺印管理システム`, `申請登録`, `承認・決裁`

### TC-LEGAL-029: Hợp đồng 5 trang cần nộp thế nào

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`, `【法務】契約書の製本方法.md`

- **Câu hỏi kiểm thử:**
  Hợp đồng A4 5 trang cần nộp pháp chế thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Theo 捺印管理システム, A4 từ 5 trang trở lên phảiホチキス止め製本 theo 製本方法.pdf. Từ tài liệu 製本方法, hợp đồng nhiều trang thường đóng gáy và đóng dấu giáp lai ở bìa trước/sau; có thể dùng giấy tape hoặc製本テープ.

- **Bằng chứng cần tìm trong tài liệu:**
  Top page “A4用紙5枚以上”; 製本方法 “割印は表紙、裏表紙”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `5枚以上`, `ホチキス止め製本`, `割印`

### TC-LEGAL-030: URL anpi hết hạn

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`

- **Câu hỏi kiểm thử:**
  Tôi mở URL anpi sau 8 ngày thì có phản hồi được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. URL trong email chỉ hiệu lực 7 ngày/168 giờ từ thời điểm khởi động. Sau 8 ngày quá hạn nên không truy cập được bằng URL đó; người dùng nên đăng nhập hệ thống hoặc liên hệ người quản trị theo hướng dẫn.

- **Bằng chứng cần tìm trong tài liệu:**
  “URLの有効期限は起動時刻から7日間（168時間）”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `8日`, `168時間`, `期限`

### TC-LEGAL-031: Không nhận mail xác nhận

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】2-1.メールアドレスの新規登録 - 方法A：ブラウザ画面からの登録.md`

- **Câu hỏi kiểm thử:**
  Đăng ký email anpi nhưng không thấy mail xác nhận thì làm gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Kiểm tra spam folder. Nếu bị spam filter hoặc deny, thêm myrescue.net vào allow domain rồi thực hiện đăng ký lại từ đầu. Không kết luận hệ thống hỏng nếu chưa kiểm tra filter/domain.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 5; “メールが届かない場合”; “myrescue.netを許可ドメイン”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `登録確認メール`, `迷惑メール`, `myrescue.net`

### TC-LEGAL-032: Đóng hợp đồng mơ hồ

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】契約書の製本方法.md`

- **Câu hỏi kiểm thử:**
  Hợp đồng này đóng thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu số trang, khổ giấy và yêu cầu nộp. Câu trả lời nên hỏi lại số trang A4 và hình thức nộp; nếu nhiều trang thì hướng dẫn đóng gáy và đóng dấu giáp lai theo 製本方法, còn quy tắc in trước khi nộp pháp chế tùy số trang theo 捺印管理システム.

- **Bằng chứng cần tìm trong tài liệu:**
  製本方法 + 捺印管理システム, nhưng cần số trang để áp dụng.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `契約書`, `何枚`, `製本`

### TC-LEGAL-033: Tình trạng nào chọn

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-2.安否確認・非常呼集に画面からログインして応答する方法.md`

- **Câu hỏi kiểm thử:**
  Tôi nên chọn trạng thái anpi nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu tình trạng thực tế và options của case. Tài liệu nói lựa chọn phụ thuộc 案件 và có thể khác nhau, nên hệ thống cần hỏi lại: hiện tại bạn an toàn hay bị thương/cần hỗ trợ, và trên màn hình có những option nào?

- **Bằng chứng cần tìm trong tài liệu:**
  “選択肢は案件によって異なります”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `状態`, `選択肢`, `案件`

### TC-LEGAL-034: Reset password trực tiếp

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】1-1.サービスへのログイン方法.md`

- **Câu hỏi kiểm thử:**
  Bạn cấp lại mật khẩu anpi cho tôi được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được giả vờ cấp mật khẩu. Tài liệu chỉ hướng dẫn dùng chức năng/link lấy lại login ID/password hoặc hỏi admin. Cần trả lời rằng hệ thống QA không có quyền cấp/reset mật khẩu và hướng người dùng làm theo tài liệu 2-4 hoặc liên hệ quản trị viên.

- **Bằng chứng cần tìm trong tài liệu:**
  “ログインID、パスワードがわからない方はこちら”; link 2-4.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `パスワード`, `再発行`, `管理者`

### TC-LEGAL-035: Quyền xem quy định

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`

- **Câu hỏi kiểm thử:**
  Tôi không xem được quyết裁権限基準表, bạn mở quyền giúp tôi được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được nói có thể mở quyền. Tài liệu chỉ ghi nếu cần thêm quyền閲覧 quy định thì yêu cầu pháp chế. Cần hướng người dùng gửi yêu cầu tới 法務部, không tự thao tác thay họ.

- **Bằng chứng cần tìm trong tài liệu:**
  “規程の閲覧権限追加は法務部へご依頼ください”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `閲覧権限`, `法務部`, `決裁権限`

### TC-LEGAL-036: Email vs màn hình

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`, `【法務】3-2.安否確認・非常呼集に画面からログインして応答する方法.md`

- **Câu hỏi kiểm thử:**
  Khi nào phản hồi anpi bằng email, khi nào đăng nhập màn hình?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu nhận được email thì mở URL trong mail và phản hồi trong 7 ngày/168 giờ. Nếu không nhận được email hoặc cần phản hồi khi mail không dùng được thì đăng nhập màn hình, vào ユーザ案件履歴一覧, chọn 案件No., bấm 状態修正 và đăng ký trạng thái.

- **Bằng chứng cần tìm trong tài liệu:**
  3-1 email response; 3-2 login response.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `メールから応答`, `画面からログイン`, `ユーザ案件履歴一覧`

### TC-LEGAL-037: Đăng ký mới vs thay đổi email

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【法務】2-1.メールアドレスの新規登録 - 方法A：ブラウザ画面からの登録.md`, `【法務】2-2.メールアドレスの変更.md`

- **Câu hỏi kiểm thử:**
  Đăng ký email mới và đổi email đã đăng ký khác nhau thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Đăng ký mới dùng màn hình メールアドレス初期登録, cần ID/password đăng ký chuyên dụng chung, số và tên; sau khi nhập email phải bấm URL trong email xác nhận để hoàn tất. Đổi email là thao tác sau khi đăng nhập bằng account cá nhân, vào 設定変更 -> メールアドレス変更, nhập email và xác nhận thay đổi. Cả hai đều cần allow myrescue.net.

- **Bằng chứng cần tìm trong tài liệu:**
  2-1 method A vs 2-2 email change.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `新規登録`, `メールアドレス変更`, `myrescue.net`

### TC-LEGAL-038: FAQ app

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】0_FAQ_app_20231020.md`

- **Câu hỏi kiểm thử:**
  Có tài liệu FAQ cho ứng dụng anpi không?

- **Câu trả lời chuẩn kỳ vọng:**
  Có file 0_FAQ_app_20231020. Câu trả lời cần trỏ đúng tài liệu FAQ, không tự tạo danh sách FAQ nếu context chưa retrieve nội dung.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 0_FAQ_app_20231020.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `FAQ`, `app`, `安否`

### TC-LEGAL-039: Family safety không hiện

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】3-1.安否確認・非常呼集にメールから応答する.md`

- **Câu hỏi kiểm thử:**
  Vì sao khi 非常呼集 lại không thấy lựa chọn tình trạng gia đình?

- **Câu trả lời chuẩn kỳ vọng:**
  Theo tài liệu, nếu dùng 家族安否 thì thường có thể chọn trạng thái gia đình, nhưng với 案件種別 là 非常呼集 thì lựa chọn 家族安否 không hiển thị.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 2; “案件種別が非常呼集の場合は、家族安否の選択は表示されません”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `非常呼集`, `家族安否`, `表示されません`

### TC-LEGAL-040: App guide anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】0_anpi_app_guide_20250325.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào là guide app anpi mới nhất?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất file 0_anpi_app_guide_20250325. Nếu người dùng cần thao tác cụ thể thì đề nghị hỏi rõ phần login, email, response, family safety hoặc mail setting.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 0_anpi_app_guide_20250325.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `anpi_app_guide`, `20250325`, `安否`

### TC-LEGAL-041: Kiểm tra trigger anpi

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Scenario
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【法務】1-1.サービスへのログイン方法.md`, `【法務】2-8.ユーザ情報の確認.md`

- **Câu hỏi kiểm thử:**
  Tôi muốn biết khu vực nào/khi nào sẽ gửi mail anpi cho tôi thì xem ở đâu?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần đăng nhập hệ thống rồi kiểm tra user information. Tài liệu 2-8 nói 安否起動設定 cho biết vùng/khu vực hoặc quy mô động đất/cảnh báo nào sẽ làm gửi mail anpi. Nếu không biết login thì dùng tài liệu 1-1/2-4.

- **Bằng chứng cần tìm trong tài liệu:**
  2-8 “安否起動設定”; 1-1 login.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `安否起動設定`, `ユーザ情報`, `ログイン`

### TC-LEGAL-042: Thời tiết ngày mai

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【法務】4-2.天気予報メールの受信設定.md`

- **Câu hỏi kiểm thử:**
  Ngày mai thời tiết ở Atsugi thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được trả lời thời tiết thực tế từ tài liệu vì tài liệu chỉ hướng dẫn cài nhận mail dự báo thời tiết, không chứa dữ liệu thời tiết hiện thời. Cần nói tài liệu không cung cấp dự báo cụ thể và hướng dẫn người dùng cài/kiểm tra mail hoặc dùng nguồn thời tiết chính thức.

- **Bằng chứng cần tìm trong tài liệu:**
  File 4-2 là setting manual, không phải dữ liệu thời tiết realtime.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `天気予報メール`, `現在の天気`, `リアルタイム`

### TC-LEGAL-043: Decision item number

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`

- **Câu hỏi kiểm thử:**
  捺印管理システム có thêm trường gì ở màn hình申請登録?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu top page ghi đã thêm 決裁項目No. vào màn hình 申請登録 và yêu cầu kiểm tra 決裁権限基準表.

- **Bằng chứng cần tìm trong tài liệu:**
  “申請登録画面に決裁項目No.を追加しました”; “決裁権限基準表をご確認ください”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `決裁項目No`, `申請登録`, `決裁権限基準表`

### TC-LEGAL-044: Cần đóng dấu hợp đồng

- **Nhóm tài liệu đang test:** Pháp chế & Quản lý rủi ro
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【法務】捺印管理システム＿トップページ.md`

- **Câu hỏi kiểm thử:**
  Tôi cần nộp hợp đồng để đóng dấu, ngoài file hợp đồng cần xem gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần xem 捺印管理システム, đặc biệt hướng dẫn in hợp đồng theo số trang, quyết裁項目No. và 決裁権限基準表 nếu hợp đồng cần稟議決裁. Nếu cần quyền xem quy định thì yêu cầu 法務部.

- **Bằng chứng cần tìm trong tài liệu:**
  Top page: 契約書の印刷, 稟議決裁が必要な契約書, 閲覧権限追加は法務部へ.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `捺印`, `契約書`, `決裁権限`

### TC-ACC-001: URL login Rakuraku

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【経理】000_001_20250324_(改)楽楽精算　操作マニュアル.md`

- **Câu hỏi kiểm thử:**
  URL đăng nhập Rakuraku Seisan cho 本社 là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  PC版URL của 本社 là https://rschime.rakurakuseisan.jp/iDcp_TdjpCa/ và smartphone browser URL là https://rschime.rakurakuseisan.jp/iDcp_TdjpCm/. Người dùng nhập login ID và password đã được thông báo hoặc password tự đặt.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 5; 本社 PC版URL và スマートフォン版ブラウザURL.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `楽楽精算`, `本社`, `rschime`

### TC-ACC-002: Dùng smartphone

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260122_楽楽精算　スマートフォン版利用マニュアル.md`

- **Câu hỏi kiểm thử:**
  Dùng Rakuraku Seisan trên điện thoại bằng cách nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Có 3 cách: smartphone browser Safari/Chrome, iPhone app 楽楽精算, Android app 楽楽精算. Với browser, mở smartphone URL riêng, thường PC URL đuôi a/ đổi thành m/. Với app, cài app rồi nhập 楽楽精算URL, login ID và password.

- **Bằng chứng cần tìm trong tài liệu:**
  はじめに; ログイン page 5-6.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `スマートフォン`, `Safari`, `Androidアプリ`

### TC-ACC-003: Giới hạn mobile

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260122_楽楽精算　スマートフォン版利用マニュアル.md`

- **Câu hỏi kiểm thử:**
  Smartphone版 có dùng được mọi chức năng như PC không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Tài liệu nói chức năng smartphone版 về cơ bản bị giới hạn ở申請 và承認 cơ bản. Smartphone版の承認者追加 chưa hỗ trợ, 複写機能 cũng không dùng được.

- **Bằng chứng cần tìm trong tài liệu:**
  ご利用時の注意事項; page 9 “承認者追加は未対応”; page 10 “複写機能は利用できません”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `スマートフォン版`, `機能差異`, `承認者追加`

### TC-ACC-004: Tạo account Rakuraku

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Muốn tạo tài khoản Rakuraku Seisan thì cần nộp gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần nộp form 〔(楽楽精算)経費精算･仮払金申請 振込依頼書〕 cho 宮谷 thuộc経理部. Vị trí form: 全社ポータル -> 経理部からのお知らせ -> 7.各種フォーマット.

- **Bằng chứng cần tìm trong tài liệu:**
  Q&A section 2 Q1/A1.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `アカウント`, `振込依頼書`, `宮谷`

### TC-ACC-005: Quên password Rakuraku

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Quên password Rakuraku Seisan thì bạn có thể cho tôi biết không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không thể cung cấp password. Tài liệu nói không thể trả lời password; cần reset password và liên hệ経理担当者 của từng công ty.

- **Bằng chứng cần tìm trong tài liệu:**
  Q&A section 2 Q3/A3.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `パスワード`, `再設定`, `経理担当者`

### TC-ACC-006: Đổi password

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Đổi password Rakuraku Seisan ở đâu?

- **Câu trả lời chuẩn kỳ vọng:**
  Đổi tại TOP画面 -> 個人設定 -> パスワード.

- **Bằng chứng cần tìm trong tài liệu:**
  Q&A section 2 Q4/A4.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `TOP画面`, `個人設定`, `パスワード`

### TC-ACC-007: Định kỳ tuyến đường

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Có cần đăng ký định kỳ区間 không?

- **Câu trả lời chuẩn kỳ vọng:**
  Có nếu người dùng được trả通勤手当. Đăng ký định期区間 giúp tự động控除 phần旅費 của tuyến対象区間, nhưng chỉ áp dụng khi tìm bằng乗換案内. Cách đăng ký xem 操作マニュアル P.30.

- **Bằng chứng cần tìm trong tài liệu:**
  Q&A section 2 Q7/A7.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `定期区間`, `通勤手当`, `乗換案内`

### TC-ACC-008: Chứng từ giấy vs điện tử

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Hóa đơn/biên lai giấy và file điện tử có cần nộp về kế toán không?

- **Câu trả lời chuẩn kỳ vọng:**
  Hóa đơn giấy nhận được phải nộp cùng伝票 cho経理部. Biên lai giấy phải gửi社内便 cho経理部 và phải rõ申請部署. Hóa đơn/biên lai nhận bằng dữ liệu điện tử thì không cần nộp, kể cả伝票.

- **Bằng chứng cần tìm trong tài liệu:**
  Page 1 “経費精算における提出書類について”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `提出書類`, `紙`, `電子データ`

### TC-ACC-009: 出張申請 bắt buộc

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Đi công tác có bắt buộc tạo 出張申請 không?

- **Câu trả lời chuẩn kỳ vọng:**
  Có. 出張申請 là bắt buộc; nếu chưa có 出張申請 thì không thể làm 出張精算. Nếu quên申請 trước, có thể申請 sau chuyến đi nhưng phải làm nhanh.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 3 Q1/A1 và Q5/A5.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `出張申請`, `必須`, `出張精算`

### TC-ACC-010: Mất receipt tàu xe

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Tôi làm mất hóa đơn giao thông công cộng, có thanh toán được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Với phương tiện công cộng, cần viết 支払証明書 và nộp cho経理部. Nếu có chứng từ chứng minh số tiền như credit statement thì đính kèm vào明細 để精算. Chỉ tinh toán được nếu税込5万円以下 và sau khi nhận支払証明書. Với phương tiện không phải công cộng, nếu không chứng minh được số tiền thì không tinh toán được.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 5 Q1/A1.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `領収書紛失`, `支払証明書`, `税込5万円以下`

### TC-ACC-011: Tàu/bus nội địa có cần receipt

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Đi tàu thường hoặc bus nội địa có cần nhận hóa đơn không?

- **Câu trả lời chuẩn kỳ vọng:**
  Nguyên tắc là không cần. Điện/bus nội địa được tinh toán dựa trên kết quả検索 của 乗換案内, nên không cần nhận/đính kèm領収書. Khi tinh toán, chọn〈出張旅費特例〉 trong【特例選択用】. Nhưng nếu dùng phương tiện ngoài tàu/bus thì cần receipt.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 5 Q2/A2.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `在来線`, `路線バス`, `乗換案内`

### TC-ACC-012: Vé tàu + đặc急

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Có thể gộp vé tàu và đặc급 trong một明細 không?

- **Câu trả lời chuẩn kỳ vọng:**
  Nguyên tắc chia明細 theo từng phương tiện. Tuy nhiên nếu vé乗車券 và đặc急券/new shinkansen nằm chung một領収書 thì có thể gộp vào một明細.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 5 Q3/A3.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `明細`, `乗車券`, `特急券`

### TC-ACC-013: Khách sạn vượt quy định

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Chi phí khách sạn vượt mức quy định thì chọn mục nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu宿泊費 vượt quy định, phải được cấp trên phê duyệt, chọn〈国内宿泊費(規定超過)〉 hoặc〈海外宿泊費(規定超過)〉 và nhập lý do vượt trong備考.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 5 Q9/A9.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `宿泊費`, `規定超過`, `備考`

### TC-ACC-014: Xe cá nhân đi công tác

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Scenario
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Lỡ dùng xe cá nhân đi công tác, tiền xăng tính thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Nguyên tắc không được dùng xe cá nhân cho công tác. Nếu bất khả kháng, tiền xăng tính theo công thức 走行距離 ÷ 19.8 × 185, chọn〈インボイス不要〉 ở【特例選択用】 và ghi trong備考 rằng dùng xe cá nhân cùng quãng đường.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 5 Q17/A17.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `私有車`, `走行距離÷19.8×185`, `インボイス不要`

### TC-ACC-015: Tạm ứng và quyết toán

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Muốn xin tạm ứng chi phí thì điều kiện và lưu ý gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu muốn仮払金 thì trong各種申請 mục【仮払金】chọn【希望する】. 支払希望日 phải đặt ít nhất 1 tuần sau ngày申請; yêu cầu chuyển khoản 2-3 ngày trước không xử lý được. Nguyên tắc trả bằng振込, tiền mặt chỉ khi có lý do đặc biệt như慶弔金/切手印紙 và phải ghi “現金” trong備考. Đã申請 tạm ứng thì bắt buộc làm精算; nếu số tinh toán thấp hơn tạm ứng thì cần返金 theo chỉ thị kế toán.

- **Bằng chứng cần tìm trong tài liệu:**
  Section 4 仮払金申請・精算方法.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `仮払金`, `1週間以上`, `返金`

### TC-ACC-016: Hỏi thiếu loại chi phí

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Chi phí này tinh toán được không?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu loại chi phí, chứng từ, mục đích và申請種別. Cần hỏi lại: đó là交通費/出張/立替/交際費/海外出張 hay支払依頼, có receipt hay dữ liệu điện tử không, ngày phát sinh và số tiền bao nhiêu.

- **Bằng chứng cần tìm trong tài liệu:**
  Q&A có nhiều quy tắc phụ thuộc loại expense; không được đoán.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `精算`, `申請種別`, `領収書`

### TC-ACC-017: Tỷ giá hôm nay

- **Nhóm tài liệu đang test:** Kế toán
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【経理】000_001_20260629_楽楽精算QA.md`

- **Câu hỏi kiểm thử:**
  Tỷ giá USD hôm nay để tinh toán công tác nước ngoài là bao nhiêu?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được bịa tỷ giá hiện tại. Tài liệu chỉ nói社内レート dựa trên foreign exchange cuối tháng trước; rate tháng sau chưa設定 trên Rakuraku. Nếu cần số cụ thể phải kiểm tra hệ thống hoặc kế toán, không suy đoán.

- **Bằng chứng cần tìm trong tài liệu:**
  海外出張精算 Q1/A1; “前月末のレート”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `社内レート`, `前月末`, `海外出張`

### TC-GA-001: Báo cáo tai nạn trong 24h

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Bản速報 của báo cáo tai nạn lao động phải nộp khi nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu ghi “速報版” phải nộp trong vòng 24 giờ sau khi phát sinh. Câu trả lời cần nêu rõ mốc 24 giờ.

- **Bằng chứng cần tìm trong tài liệu:**
  記入上の注意事項; “速報版は発生24時間以内に提出すること”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `速報版`, `24時間以内`, `労働災害`

### TC-GA-002: Thông tin người bị nạn

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Báo cáo tai nạn lao động cần ghi thông tin gì về người bị nạn?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần ghi các trường như họ tên, tuổi, giới tính, ngày vào công ty, phòng ban, kinh nghiệm tại nơi làm việc. Ngoài ra báo cáo còn cần thời điểm, địa điểm, tình trạng thương tích, bệnh viện, mức độ thương tật và tình huống phát sinh.

- **Bằng chứng cần tìm trong tài liệu:**
  Bảng 別紙１; các trường 被災者, 氏名, 年齢, 性別, 入社日, 所属, 職場での経験.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `被災者`, `氏名`, `職場での経験`

### TC-GA-003: Thời gian 24h

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Trong báo cáo tai nạn, thời gian phát sinh ghi theo định dạng nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu yêu cầu thời gian phát sinh ghi theo hệ 24 giờ.

- **Bằng chứng cần tìm trong tài liệu:**
  記入上の注意事項; “発生時間は24時間制にて記入する”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `発生時間`, `24時間制`, `発生日時`

### TC-GA-004: Không nghỉ làm

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  不休労災 trong form có bao gồm nghỉ/ngày đi viện ngay hôm phát sinh không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Ghi chú nói 不休 không bao gồm nghỉ làm hoặc đi viện trong ngày phát sinh. Cần phân biệt với休業労災 nếu có ngày nghỉ sau đó.

- **Bằng chứng cần tìm trong tài liệu:**
  Ghi chú 傷病の程度; “不休に発生当日の休業/通院は含まない”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `不休労災`, `発生当日`, `通院`

### TC-GA-005: 休業見込み báo cáo

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Nếu lúc nộp確定版 vẫn chỉ biết số ngày nghỉ dự kiến thì làm thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Nếu báo cáo確定版 nhưng休業日数 vẫn là見込み, ghi予定休業日数 và chọn見込み để nộp確定(見込み)版. Sau khi復職, ghi休業日数 và復職日, chọn確定 rồi nộp確定(最終)版.

- **Bằng chứng cần tìm trong tài liệu:**
  記入上の注意事項; “確定版 제출時に休業日数が見込みの場合”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `確定(見込み)`, `復職日`, `確定(最終)`

### TC-GA-006: Phân tích nguyên nhân tai nạn

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書Rev3_分析シート付.md`

- **Câu hỏi kiểm thử:**
  Mẫu phân tích tai nạn yêu cầu đào sâu nguyên nhân theo những phần nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu Rev3 có sheet phân tích: kiểm chứng sự kiện theo thời gian, trích xuất trạng thái không an toàn và hành động không an toàn, phân tích yếu tố theo nạn nhân, phương pháp làm việc, thiết bị, người quản lý, môi trường hiện trường; dùng FTA/năm lần vì sao để chọn đối sách quan trọng và đưa vào báo cáo.

- **Bằng chứng cần tìm trong tài liệu:**
  Sections 時系列事実検証, 災害要因, FTA.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `時系列事実検証`, `不安全状態`, `FTA`

### TC-GA-007: Rev3 vs Rev4

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書Rev3_分析シート付.md`, `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Rev3 phân tích tai nạn và Rev4 báo cáo tai nạn khác nhau thế nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Rev4 là mẫu báo cáo労働災害 phát sinh, có chú ý về速報 trong 24 giờ, 24h time format,復職/確定見込み/最終. Rev3 có thêm analysis sheet để kiểm chứng thời gian, phân tích trạng thái/hành vi không an toàn, yếu tố tai nạn và FTA. Câu trả lời nên nói Rev4 dùng cho báo cáo chính, Rev3 hỗ trợ phân tích nguyên nhân.

- **Bằng chứng cần tìm trong tài liệu:**
  Rev4 別紙１; Rev3 分析シート付, 時系列事実検証, 災害要因, FTA.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `Rev3`, `Rev4`, `分析シート`

### TC-GA-008: Xin cấp đồng phục

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`

- **Câu hỏi kiểm thử:**
  Muốn xin cấp hoặc đổi đồng phục thì cần điền gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Dùng 作業服貸与依頼票 gửi 総務本部総務課/メールルーム. Điền ngày, 部署CD, 部署名, 社員No, 内線No, 氏名, chọn 新規 hoặc 交換, có返却品 hay không, cách nhận, lý do như破損/汚れ/サイズ違い/未貸与, sau đó chọn item, size và số lượng.

- **Bằng chứng cần tìm trong tài liệu:**
  作業服貸与依頼票 table fields.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `作業服貸与依頼票`, `新規`, `交換`

### TC-GA-009: Cách nhận đồng phục

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`

- **Câu hỏi kiểm thử:**
  Có những cách nhận đồng phục nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Có 4 cách: người dùng tự nhận là nguyên tắc, người đại diện nhận thay với tên người đại diện và số nội tuyến, nhận qua社内便 cho bản社・神奈川 ngoài phạm vi, hoặc cách khác ghi rõ.

- **Bằng chứng cần tìm trong tài liệu:**
  受取り方法; ①本人受取り(原則), ②代理受取り, ③社内便, ④他.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `受取り方法`, `本人受取り`, `代理受取り`

### TC-GA-010: Lý do đổi đồng phục

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`

- **Câu hỏi kiểm thử:**
  Mẫu cấp đồng phục có những lý do đổi/cấp nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Các lý do mẫu gồm破損, 汚れ, サイズ違い, 未貸与 và mục “他” để ghi lý do khác.

- **Bằng chứng cần tìm trong tài liệu:**
  理　　　　由 row in 作業服貸与依頼票.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `破損`, `汚れ`, `サイズ違い`

### TC-GA-011: Mã giày/sandal

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】作業靴出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  作業靴出庫依頼票 có những mã nào cho sandal PS-01S màu xám?

- **Câu trả lời chuẩn kỳ vọng:**
  Màu xám 静電サンダル PS-01S グレー có mã P0321 23.5cm, P0322 24.0cm, P0323 24.5cm, P0324 25.0cm, P0325 25.5cm, P0326 26.0cm, P0327 26.5cm, P0328 27.0cm, P0329 27.5cm, P0330 28.0cm, P0331 29.0cm.

- **Bằng chứng cần tìm trong tài liệu:**
  作業靴出庫依頼票 table rows P0321-P0331.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `P0321`, `静電サンダル`, `グレー`

### TC-GA-012: Mã sandal trắng

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】作業靴出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  静電サンダル PS-01S màu trắng size 24.0cm có mã gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Mã là P0338 cho 静電サンダル PS-01S 白 size 24.0cm.

- **Bằng chứng cần tìm trong tài liệu:**
  作業靴出庫依頼票 row P0338.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `P0338`, `白`, `24.0cm`

### TC-GA-013: Xin xuất kho vật tư chung

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】共通在庫品出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  Muốn xin xuất kho vật tư chung thì mẫu cần ghi gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Dùng 共通在庫品出庫依頼票 gửi 人事総務部総務グループ/メールルーム. Điền ngày, 部署, CD, 氏名, tên hàng, số lượng và備考. Mẫu có ghi CD欄 dùng để付け替え chi phí cho bộ phận được ghi.

- **Bằng chứng cần tìm trong tài liệu:**
  共通在庫品出庫依頼票 fields; “CD欄へ記載の部門へ経費の付け替え”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `共通在庫品`, `出庫依頼票`, `CD欄`

### TC-GA-014: Đơn vị cấp phong bì

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】共通在庫品出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  Phong bì công ty trong 共通在庫品 được cấp theo đơn vị nào?

- **Câu trả lời chuẩn kỳ vọng:**
  社名入り封筒 gồm 長3, 長4, 角2, エアメール, 窓付（長3） đều theo đơn vị 100枚. Túi 手提げ袋 theo 10枚, 購買依頼票 theo 1冊, 白シーツ theo 2kg.

- **Bằng chứng cần tìm trong tài liệu:**
  Rows 社名入り封筒, 手提げ袋, 購買依頼票, 白シーツ.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `社名入り封筒`, `100枚単位`, `手提げ袋`

### TC-GA-015: Mẫu số 5 lao động

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】元データ‗様式第5号‗労災用（療養補償給付請求書）.md`

- **Câu hỏi kiểm thử:**
  様式第5号 dùng cho trường hợp nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu là mẫu 労災用「療養補償給付請求書」. Câu trả lời nên nói đây là mẫu claim/cấp療養補償 cho tai nạn lao động nghiệp vụ, có hướng dẫn mặt sau về các trường cần ghi và lưu ý không làm bẩn/đục lỗ/gấp quá mạnh vì máy đọc.

- **Bằng chứng cần tìm trong tài liệu:**
  Title 元データ‗様式第5号‗労災用; page 2 注意事項.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `様式第5号`, `労災用`, `療養補償給付`

### TC-GA-016: Mẫu 16-3 commute accident

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】元データ‗様式16号の3‗通勤災害用（療養給付たる療養の給付請求書）.md`

- **Câu hỏi kiểm thử:**
  様式16号の3 dùng cho trường hợp nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Đây là mẫu 通勤災害用「療養給付たる療養の給付請求書」 của bảo hiểm tai nạn lao động, dùng cho tai nạn trong quá trình đi lại. Mặt sau có phần 通勤災害に関する事項 như loại đi lại từ nhà đến nơi làm việc, từ nơi làm việc về nhà, giữa nơi làm việc, thời điểm và địa điểm phát sinh.

- **Bằng chứng cần tìm trong tài liệu:**
  Title 様式第16号の3 通勤災害用; page 2 通勤災害に関する事項.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `様式16号の3`, `通勤災害`, `療養給付`

### TC-GA-017: 労災 vs 通勤災害

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】元データ‗様式第5号‗労災用（療養補償給付請求書）.md`, `【総務】元データ‗様式16号の3‗通勤災害用（療養給付たる療養の給付請求書）.md`

- **Câu hỏi kiểm thử:**
  Tai nạn trong nhà máy và tai nạn trên đường đi làm dùng cùng mẫu không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không nên dùng lẫn. Tai nạn nghiệp vụ/lao động dùng 様式第5号 労災用 療養補償給付請求書. Tai nạn đi lại dùng 様式第16号の3 通勤災害用 療養給付たる療養の給付請求書.

- **Bằng chứng cần tìm trong tài liệu:**
  Titles of both form files.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `労災用`, `通勤災害用`, `様式第5号`

### TC-GA-018: Ngày áp dụng bảng quyết裁

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】●決裁権限基準表_結合_2025最新.md`

- **Câu hỏi kiểm thử:**
  Bảng quyết裁権限 bản mới áp dụng từ khi nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Heading của bảng ghi 本社29版 và ngày áp dụng 2025年11月1日 cho (株)メイコー.

- **Bằng chứng cần tìm trong tài liệu:**
  Heading “29版：2025年11月1日”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `決裁権限基準表`, `29版`, `2025年11月1日`

### TC-GA-019: Mã form quyết裁

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】●決裁権限基準表_結合_2025最新.md`

- **Câu hỏi kiểm thử:**
  Trong bảng quyết裁, ký hiệu A/B/C nghĩa là gì?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu ghi【書式】A là 取締役会審議依頼書, B là 一般稟議, 人事B là 人事稟議, C là その他.

- **Bằng chứng cần tìm trong tài liệu:**
  Header row “【書式】A:取締役会審議依頼書 B:一般稟議 人事B:人事稟議 C:その他”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `書式`, `A`, `一般稟議`

### TC-GA-020: Giá trị chưa thuế

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Condition / Exception
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【総務】●決裁権限基準表_結合_2025最新.md`

- **Câu hỏi kiểm thử:**
  Khi bảng quyết裁 có ngưỡng tiền thì tính theo giá gồm thuế hay chưa thuế?

- **Câu trả lời chuẩn kỳ vọng:**
  Tài liệu ghi các mục có hiển thị số tiền về nguyên tắc xử lý theo giá chưa thuế cho cả Nhật và overseas.

- **Bằng chứng cần tìm trong tài liệu:**
  Header supplement “金額表示のあるものは原則、日本・海外ともに税抜価格にて扱う”.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `金額表示`, `税抜価格`, `日本・海外`

### TC-GA-021: Đổi sai size

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`

- **Câu hỏi kiểm thử:**
  Đồng phục mới bị sai size, tôi chọn mục nào trên phiếu?

- **Câu trả lời chuẩn kỳ vọng:**
  Chọn 交換, mục返却品 有/無 tùy có trả đồ cũ, lý do chọn サイズ違い, sau đó điền item, size đúng, số lượng và cách nhận.

- **Bằng chứng cần tìm trong tài liệu:**
  作業服貸与依頼票: 新規・交換, 返却品, 理由 サイズ違い.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `交換`, `サイズ違い`, `返却品`

### TC-GA-022: Xin mask và phong bì

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Scenario
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】共通在庫品出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  Tôi muốn xin mask than hoạt tính và phong bì công ty thì dùng phiếu nào và đơn vị nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Dùng 共通在庫品出庫依頼票. 活性炭入マスク theo đơn vị 5枚, còn 社名入り封筒 như 長3/長4/角2/エアメール/窓付（長3） theo đơn vị 100枚.

- **Bằng chứng cần tìm trong tài liệu:**
  Rows 活性炭入マスク, 社名入り封筒.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `活性炭入マスク`, `社名入り封筒`, `100枚`

### TC-GA-023: Người bị nạn nghỉ ước tính

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Scenario
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【総務】V5101 別紙1 労働災害発生報告書〈速報 ・ 確定(見込み／最終)〉Rev4.md`

- **Câu hỏi kiểm thử:**
  Có tai nạn lao động, người bị nạn chưa復職 và số ngày nghỉ chỉ là dự kiến, phải nộp bản nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Nộp 確定(見込み)版: ghi予定休業日数 và chọn見込み. Sau khi người bị nạn復職, ghi休業日数 và復職日, chọn確定 rồi nộp 確定(最終)版.

- **Bằng chứng cần tìm trong tài liệu:**
  記入上の注意事項 for 確定版 with 見込み.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `確定(見込み)`, `予定休業日数`, `復職日`

### TC-GA-024: Hỏi phê duyệt thiếu thông tin

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】●決裁権限基準表_結合_2025最新.md`

- **Câu hỏi kiểm thử:**
  Việc này cần ai phê duyệt?

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu loại quyết裁事項, công ty áp dụng, số tiền và bối cảnh. Câu trả lời chuẩn phải hỏi lại: đây là hợp đồng/mua sắm/nhân sự/thiết bị hay hạng mục nào, số tiền chưa thuế bao nhiêu, thuộc pháp nhân nào; sau đó mới đối chiếu決裁権限基準表.

- **Bằng chứng cần tìm trong tài liệu:**
  Decision table depends on 決裁事項, 基準, 金額, pháp nhân.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `決裁事項`, `金額`, `基準`

### TC-GA-025: Xin giày thiếu size

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Ambiguous / Underspecified
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】作業靴出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  Tôi cần xin giày bảo hộ, làm giúp tôi.

- **Câu trả lời chuẩn kỳ vọng:**
  Câu hỏi thiếu size, màu/loại và số lượng. Cần hỏi lại size cm, loại 静電サンダル PS-01S màu xám hay trắng, số lượng, bộ phận/CD trước khi xác định mã P03xx và điền phiếu.

- **Bằng chứng cần tìm trong tài liệu:**
  作業靴出庫依頼票 cần size/mã P0321-P0340 và 部署/CD/数量.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `作業靴`, `サイズ`, `数量`

### TC-GA-026: Lịch thu gom rác

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】廃棄物・有価物保管場所配置図.md`

- **Câu hỏi kiểm thử:**
  Lịch thu gom rác tuần này là khi nào?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được bịa lịch. Tài liệu 廃棄物・有価物保管場所配置図 là sơ đồ vị trí lưu giữ rác/vật có giá trị, không cung cấp lịch thu gom theo tuần. Cần nói tài liệu hiện tại chưa đủ căn cứ và đề nghị hỏi tổng vụ/môi trường hoặc cung cấp tài liệu lịch thu gom.

- **Bằng chứng cần tìm trong tài liệu:**
  File title 配置図; không phải schedule.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `廃棄物`, `保管場所`, `収集日`

### TC-GA-027: Giá đồng phục

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Hallucination Guard
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`

- **Câu hỏi kiểm thử:**
  Một bộ đồng phục giá bao nhiêu?

- **Câu trả lời chuẩn kỳ vọng:**
  Không được bịa giá. Phiếu 作業服貸与依頼票 chỉ có item/size/số lượng và cách nhận/lý do, không có giá. Cần nói tài liệu không cung cấp đơn giá và hướng hỏi総務/購買 nếu cần chi phí.

- **Bằng chứng cần tìm trong tài liệu:**
  作業服貸与依頼票 không có cột giá.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `作業服`, `価格`, `単価`

### TC-GA-028: So sánh form tổng vụ

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Multi-hop
- **Độ khó:** Hard
- **Nguồn tài liệu:** `【総務】作業服貸与依頼票_202601版.md`, `【総務】作業靴出庫依頼票.md`, `【総務】共通在庫品出庫依頼票.md`

- **Câu hỏi kiểm thử:**
  Xin đồng phục, xin giày và xin vật tư chung dùng cùng một mẫu không?

- **Câu trả lời chuẩn kỳ vọng:**
  Không. Đồng phục dùng 作業服貸与依頼票, có lý do新規/交換,返却品 và cách nhận. Giày/sandal dùng 共通在庫品（作業靴）出庫依頼票 với mã P0321-P0340 theo size/màu. Vật tư chung như phong bì, mask, găng dùng 共通在庫品出庫依頼票 với đơn vị cấp phát riêng.

- **Bằng chứng cần tìm trong tài liệu:**
  Three form titles and fields.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `作業服貸与依頼票`, `作業靴出庫依頼票`, `共通在庫品`

### TC-GA-029: Điều tra nguyên nhân sự cố

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Procedure
- **Độ khó:** Medium
- **Nguồn tài liệu:** `【総務】事象の原因系究明_元データ.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào dùng để truy nguyên nguyên nhân sự cố?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần truy xuất 事象の原因系究明_元データ và trả lời đây là tài liệu/form dùng để làm rõ nguyên nhân sự cố. Nếu context không có chi tiết, không tự tạo quy trình ngoài tài liệu; có thể liên hệ với mẫu 労働災害分析 nếu là tai nạn lao động.

- **Bằng chứng cần tìm trong tài liệu:**
  Tên file 事象の原因系究明_元データ.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `事象`, `原因究明`, `元データ`

### TC-GA-030: Bản đồ nơi lưu rác/vật có giá trị

- **Nhóm tài liệu đang test:** Hành chính tổng hợp
- **Loại câu hỏi:** Direct Lookup
- **Độ khó:** Easy
- **Nguồn tài liệu:** `【総務】廃棄物・有価物保管場所配置図.md`

- **Câu hỏi kiểm thử:**
  Tài liệu nào có sơ đồ khu vực lưu giữ phế thải và vật có giá trị?

- **Câu trả lời chuẩn kỳ vọng:**
  Cần trỏ đúng 廃棄物・有価物保管場所配置図. Nếu user hỏi vị trí cụ thể, trả lời dựa trên sơ đồ; nếu context không có vị trí, nói cần mở đúng sơ đồ.

- **Bằng chứng cần tìm trong tài liệu:**
  File title 廃棄物・有価物保管場所配置図.

- **Tiêu chí chấm điểm:**
  - Đúng nội dung chính trong tài liệu: phải khớp nguồn và ý chính.
  - Có đủ điều kiện / ngoại lệ nếu có: không bỏ qua giới hạn quan trọng.
  - Không bịa thêm ngoài tài liệu: nếu thiếu dữ liệu phải nói rõ thiếu căn cứ.
  - Trả lời rõ ràng, có cấu trúc: người dùng nội bộ đọc là làm theo được.

- **Từ khóa truy xuất gợi ý:**
  `廃棄物`, `有価物`, `保管場所`
