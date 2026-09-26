# ClarityFI backend image. The frontend is NOT built here — it's served as static
# files directly by Caddy on the host (see docs/deployment_aws.md), so this image
# only needs to run the FastAPI + LangGraph backend.
FROM python:3.12-slim

# build-essential: defensive — most deps (spaCy, presidio, qdrant-client, faiss-cpu)
# ship manylinux wheels, but this avoids a failed build if any transitive dep doesn't.
# curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.yaml uvicorn_log_config.json ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "src.web_app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "uvicorn_log_config.json"]
