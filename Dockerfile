FROM node:20-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
RUN pip install \
        "google-api-python-client>=2.192.0,<3.0.0" \
        "google-auth-httplib2>=0.3.0,<1.0.0" \
        "google-auth-oauthlib>=1.3.0,<2.0.0"

COPY src ./src
COPY config ./config
COPY documents ./documents
COPY scripts ./scripts
COPY database/schema ./database/schema
COPY litellm_config.yaml ./litellm_config.yaml
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/uploads /app/logs /app/data /app/mkac_processed/pages

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
