import re

with open('src/integrations/mes_sql_agent.py', 'r') as f:
    content = f.read()

# Fix the broken string formatting
content = content.replace(
    '                                        f"Semantic model:\n{semantic_model}\n\n"\n                    f"Câu hỏi: {question}{retry}\n\n"\n                    "Ví dụ suy luận: nếu hỏi top lỗi trong Lot, dùng CTE tìm lot_id "\n                    "đứng đầu từ v_lot_error_summary, sau đó lọc v_lot_error_breakdown, "\n                    "SUM(total_error_qty), GROUP BY lot_id,product_id,error_id,error_name "\n                    "và lấy top N.\n"\n                    "LƯU Ý VỀ THỜI GIAN: Khi phân tích dữ liệu theo thời gian hoặc tính "\n                    "toán khoảng ngày, luôn ưu tiên sử dụng các trường UNIX timestamp "\n                    "như `error_time_unix` (đơn vị milliseconds) để so sánh (>, <, BETWEEN). "\n                    "Chỉ dùng hàm `strftime` hoặc `date()` trên các cột `_time` (chuỗi TEXT) "\n                    "nếu cần GROUP BY theo chu kỳ ngày/tháng để hiển thị kết quả cuối cùng." ',
    '                    f"Semantic model:\\n{semantic_model}\\n\\n"\n                    f"Câu hỏi: {question}{retry}\\n\\n"\n                    "Ví dụ suy luận: nếu hỏi top lỗi trong Lot, dùng CTE tìm lot_id "\n                    "đứng đầu từ v_lot_error_summary, sau đó lọc v_lot_error_breakdown, "\n                    "SUM(total_error_qty), GROUP BY lot_id,product_id,error_id,error_name "\n                    "và lấy top N.\\n"\n                    "LƯU Ý VỀ THỜI GIAN: Khi phân tích dữ liệu theo thời gian hoặc tính "\n                    "toán khoảng ngày, luôn ưu tiên sử dụng các trường UNIX timestamp "\n                    "như `error_time_unix` (đơn vị milliseconds) để so sánh (>, <, BETWEEN). "\n                    "Chỉ dùng hàm `strftime` hoặc `date()` trên các cột `_time` (chuỗi TEXT) "\n                    "nếu cần GROUP BY theo chu kỳ ngày/tháng để hiển thị kết quả cuối cùng."'
)

with open('src/integrations/mes_sql_agent.py', 'w') as f:
    f.write(content)
