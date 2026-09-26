#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -x .venv/bin/uvicorn ] || [ ! -d frontend/node_modules ]; then
  echo "First-run setup is required."
  bash setup_mac.command
fi

echo "Starting the FastAPI backend..."
.venv/bin/uvicorn src.web_app.api.main:app --host 127.0.0.1 --port 8000 --log-config uvicorn_log_config.json > /tmp/clarity-api.log 2>&1 &
API_PID=$!

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for attempt in {1..30}; do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "FastAPI did not start. See /tmp/clarity-api.log for details."
  exit 1
fi

echo "Opening Clarity at http://127.0.0.1:5173"
(sleep 2; open "http://127.0.0.1:5173") &
npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173
