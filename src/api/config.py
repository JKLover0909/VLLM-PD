"""Hằng số cấu hình đọc từ biến môi trường cho API Gateway.

Tách khỏi ``main.py`` để mọi thiết lập vận hành (cổng, giới hạn upload, cache,
rate limit, đường dẫn) nằm gọn một chỗ, dễ tra cứu và điều chỉnh. ``main.py``
gọi ``load_dotenv()`` trước khi import module này nên các ``os.getenv`` ở đây đã
thấy giá trị trong ``.env``.
"""

import os
from pathlib import Path

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
RESEARCH_DEMO_DIR = Path(os.getenv("RESEARCH_DEMO_DIR", "./documents/Research"))
RESEARCH_DEMO_SESSION_ID = os.getenv(
    "RESEARCH_DEMO_SESSION_ID",
    "00000000-0000-4000-8000-000000000001",
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-local")
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
QUERY_RATE_LIMIT = int(os.getenv("QUERY_RATE_LIMIT_PER_MINUTE", "15"))
QUERY_RESPONSE_CACHE_TTL_SECONDS = max(
    0,
    int(os.getenv("QUERY_RESPONSE_CACHE_TTL_SECONDS", "600")),
)
QUERY_RESPONSE_CACHE_SIZE = max(
    0,
    int(os.getenv("QUERY_RESPONSE_CACHE_SIZE", "256")),
)
# MES snapshot là dữ liệu tĩnh tới lần re-import nên câu trả lời cache được lâu
# hơn nhiều (khóa cache đã gắn phiên bản snapshot nên re-import tự vô hiệu hóa).
MES_QUERY_CACHE_TTL_SECONDS = max(
    0,
    int(os.getenv("MES_QUERY_CACHE_TTL_SECONDS", "86400")),
)
MIN_QUERY_RESPONSE_SECONDS = max(
    0.0,
    float(os.getenv("MIN_QUERY_RESPONSE_SECONDS", "2.0")),
)
UPLOAD_RATE_LIMIT = int(os.getenv("UPLOAD_RATE_LIMIT_PER_HOUR", "10"))
UPLOAD_PROCESSING_CONCURRENCY = max(
    1, int(os.getenv("UPLOAD_PROCESSING_CONCURRENCY", "1"))
)
UPLOAD_QUEUE_SIZE = max(0, int(os.getenv("UPLOAD_QUEUE_SIZE", "4")))
ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}
PREVIEW_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
EMPLOYEE_DIRECTORY_DB_PATH = Path(
    os.getenv("EMPLOYEE_DIRECTORY_DB_PATH", "data/employee_directory.sqlite")
)
MKAC_PAGE_IMAGE_DIR = Path(os.getenv("MKAC_PAGE_IMAGE_DIR", "mkac_processed/pages"))
ENABLE_AGENT = os.getenv("ENABLE_AGENT", "true").lower() in {"1", "true", "yes", "on"}
