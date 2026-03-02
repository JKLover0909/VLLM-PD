Bắt buộc 1: vLLM server
Chạy start_vllm.sh

Bắt buộc 2: Backend FastAPI
Chạy main.py qua lệnh python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload (đứng trong thư mục backend)

Tùy chọn 3: Frontend hoặc Notebook để test

Webapp: app.py bằng streamlit run [app.py](http://_vscodecontentref_/5) --server.port 8501
Hoặc notebook: Code.ipynb