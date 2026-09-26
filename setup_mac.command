#!/bin/bash
set -e

cd "$(dirname "$0")"

fail_with_link() {
  echo
  echo "$1"
  echo "$2"
  open "$2" >/dev/null 2>&1 || true
  read -r -p "Press Return to close..."
  exit 1
}

command -v python3 >/dev/null 2>&1 || fail_with_link "Python 3.12 or newer is required." "https://www.python.org/downloads/macos/"
command -v node >/dev/null 2>&1 || fail_with_link "Node.js 20 or newer is required." "https://nodejs.org/en/download"

PYTHON_OK=$(python3 -c "import sys; print(int(sys.version_info >= (3, 12)))")
if [ "$PYTHON_OK" != "1" ]; then
  fail_with_link "Python 3.12 or newer is required." "https://www.python.org/downloads/macos/"
fi

NODE_MAJOR=$(node -p "Number(process.versions.node.split('.')[0])")
if [ "$NODE_MAJOR" -lt 20 ]; then
  fail_with_link "Node.js 20 or newer is required." "https://nodejs.org/en/download"
fi

if [ ! -f .env ]; then
  echo "Missing .env. Copy .env.example to .env and add your OPENAI_API_KEY."
  exit 1
fi

echo "Creating the Python environment..."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Installing the React packages..."
npm --prefix frontend install

CHUNK_COUNT=$(.venv/bin/python -c "from src.web_app.api.main import store; print(store.count())")
if [ "$CHUNK_COUNT" = "0" ]; then
  echo "Loading the finance knowledge base (uses OpenAI embeddings)..."
  .venv/bin/python -m src.rag.ingest --file src/data/kb/clarity_financial_education_common.json --reset
fi

echo
echo "Setup complete. Double-click start.command to open Clarity."
read -r -p "Press Return to close..."
