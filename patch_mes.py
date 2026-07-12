import re

with open('src/integrations/mes_sql_agent.py', 'r') as f:
    content = f.read()

new_system_content = """                    "Bạn là bộ lập kế hoạch SQL cho MES snapshot SQLite. "
                    "Chỉ dùng view và cột trong semantic model. Chỉ sinh một "
                    "SELECT hoặc WITH...SELECT. Không dùng markdown, comment, "
                    "PRAGMA, ATTACH, DDL hay câu lệnh ghi. Khi câu hỏi không thể "
                    "trả lời từ schema, đặt can_answer=false. Cụm 'loại lỗi' "
                    "nghĩa là nhóm error_id + error_name. Luôn đặt alias dễ hiểu "
                    "và BẮT BUỘC có giới hạn LIMIT (tối đa 50) để tránh tràn bộ nhớ. "
                    "Các view MES đã loại dữ liệu Lot/sản phẩm test; không cố truy "
                    "xuất lại dữ liệu test. Chỉ trả đúng JSON: "
                    '{"can_answer":true,"sql":"...","reason":"..."}.'"""

new_user_content = """                    f"Semantic model:\\n{semantic_model}\\n\\n"
                    f"Câu hỏi: {question}{retry}\\n\\n"
                    "Ví dụ suy luận: nếu hỏi top lỗi trong Lot, dùng CTE tìm lot_id "
                    "đứng đầu từ v_lot_error_summary, sau đó lọc v_lot_error_breakdown, "
                    "SUM(total_error_qty), GROUP BY lot_id,product_id,error_id,error_name "
                    "và lấy top N.\\n"
                    "LƯU Ý VỀ THỜI GIAN: Khi phân tích dữ liệu theo thời gian hoặc tính "
                    "toán khoảng ngày, luôn ưu tiên sử dụng các trường UNIX timestamp "
                    "như `error_time_unix` (đơn vị milliseconds) để so sánh (>, <, BETWEEN). "
                    "Chỉ dùng hàm `strftime` hoặc `date()` trên các cột `_time` (chuỗi TEXT) "
                    "nếu cần GROUP BY theo chu kỳ ngày/tháng để hiển thị kết quả cuối cùng." """

content = re.sub(
    r'"Bạn là bộ lập kế hoạch SQL cho MES snapshot SQLite\. ".*?\'\{"can_answer":true,"sql":"\.\.\.","reason":"\.\.\."\}\.\'',
    new_system_content,
    content,
    flags=re.DOTALL
)

content = re.sub(
    r'f"Semantic model:\\n\{semantic_model\}\\n\\n".*?"câu hỏi nói rõ \'ngày sản xuất\'\."',
    new_user_content,
    content,
    flags=re.DOTALL
)

with open('src/integrations/mes_sql_agent.py', 'w') as f:
    f.write(content)

print("Patched successfully")
