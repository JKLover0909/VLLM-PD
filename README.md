# Chạy nhanh bằng 1 file (khuyến nghị)
Vào môi trường conda `docmind`, sau đó chạy:

cd /home/jkl/Code/VLLM-PD/docmind/scripts
./run_all.sh start

Các lệnh quản lý:

./run_all.sh status
./run_all.sh stop

Nếu chỉ muốn chạy vLLM + Backend (không mở frontend):

./run_all.sh start --no-frontend

# Chạy thủ công từng phần (khi cần debug)

Bắt buộc 1: vLLM server
Vào môi trường conda docmind
Cd vào /home/jkl/Code/VLLM-PD/docmind/scripts
Chạy start_vllm.sh

Bắt buộc 2: Backend FastAPI
Vào môi trường conda docmind
Cd vào /home/jkl/Code/VLLM-PD/docmind/backend
Chạy main.py qua lệnh python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload (đứng trong thư mục backend)

Tùy chọn 3: Frontend hoặc Notebook để test
Vào môi trường conda docmind
Cd vào /home/jkl/Code/VLLM-PD/docmind/frontend
Chạy app.py qua lệnh streamlit run app.py --server.port 8501

Đã thêm

run_all.sh
start: chạy vLLM + backend + frontend
start --no-frontend: chỉ vLLM + backend
status: xem trạng thái PID
stop: dừng toàn bộ
ghi log vào docmind/logs/
Cập nhật hướng dẫn trong README.md.
Cách dùng

cd /home/jkl/Code/VLLM-PD/docmind/scripts
./run_all.sh start
./run_all.sh status
./run_all.sh stop