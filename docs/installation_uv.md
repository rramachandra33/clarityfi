# Installing with uv

An alternative to `setup_mac.command` for people who prefer [uv](https://docs.astral.sh/uv/) (a fast,
single-binary Python package/venv manager) over the manual `python3 -m venv` + `pip install -r requirements.txt`
flow. Every command below was run against this exact repo while writing this doc: `uv sync` produces a working
`.venv` from `pyproject.toml`, and the full test suite (1,276 tests) passes against it unchanged.

Nothing else in the project changes — `start.command`, `setup_mac.command` and CI can keep using
`requirements.txt` if you prefer; `pyproject.toml` and `requirements.txt` list the same direct dependencies.

## 1. Prerequisites

- **uv** itself:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  (or `brew install uv`). Verify with `uv --version`.
- **Node.js 20+** for the React frontend (uv only manages the Python side): `node --version`.
- You do **not** need Python pre-installed — `uv sync` will download and manage Python 3.12 itself if it's
  missing (the project pins `requires-python = ">=3.12"` in `pyproject.toml` and `.python-version`).
- An OpenAI API key (required). Alpha Vantage and SerpAPI keys are optional — the app falls back to yFinance
  and sample data without them, and says so.

## 2. Get the project and configure secrets

```bash
cd clarityFI                      # this repo
cp .env.example .env
```

Open `.env` and set at least `OPENAI_API_KEY`. `.env` is git-ignored — never commit it.

## 3. Install the Python dependencies

```bash
uv sync
```

This reads `pyproject.toml` (dependencies below) and `[dependency-groups] dev` (pytest, pytest-cov), creates
`.venv/` in the project root, and installs everything into it — including the spaCy `en_core_web_lg` model,
which is declared as a direct wheel URL and downloads automatically (~600 MB the first time; PII redaction
needs it). `uv sync` also writes `uv.lock`, a reproducible lockfile — commit it if you want everyone on the
team to resolve to the exact same versions (not committed by default here, so as not to fight with the
existing `requirements.txt`-based workflow; delete this note once you've decided which one is the source of
truth for your team).

No manual `source .venv/bin/activate` is required for the rest of this doc — either run commands with
`uv run <command>` (uses the project's `.venv` automatically) or activate it yourself
(`source .venv/bin/activate`) and drop the `uv run` prefix.

### What's declared in `pyproject.toml`

```toml
dependencies = [
    # Web framework
    "fastapi==0.141.1", "uvicorn[standard]==0.53.0", "httpx==0.28.1",
    "python-multipart==0.0.32", "python-dotenv==1.2.3", "pydantic>=2", "pyyaml>=6",

    # Data / retrieval
    "numpy>=1.26", "faiss-cpu>=1.8", "qdrant-client>=1.9",

    # LLM / agents / workflow
    "langchain>=1.0", "langchain-core>=0.3", "langchain-openai>=0.3", "langgraph>=1.0",
    "openai>=1.0", "langfuse>=3",

    # Market / news data
    "yfinance>=0.2",

    # Privacy (PII redaction)
    "presidio-analyzer>=2.2", "presidio-anonymizer>=2.2", "spacy>=3.8",
    "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl",

    # MCP server
    "mcp>=2",
]

[dependency-groups]
dev = ["pytest==9.1.1", "pytest-cov>=5"]
```

(`tenacity`, `tiktoken`, `pandas` and similar show up in `uv.lock`/`pip list` too — they're transitive, pulled
in by `langchain`/`yfinance`/`openai`, not something this project imports directly, so they're intentionally
not listed as direct dependencies.)

## 4. Install the frontend dependencies

```bash
npm --prefix frontend install
```

(uv only manages Python; the React app still uses npm.)

## 5. Load the knowledge base (one-time)

```bash
uv run python -m src.rag.ingest --file src/data/kb/clarity_financial_education_common.json --reset
```

Embeds and indexes the 230-article knowledge base into both Qdrant and FAISS (uses OpenAI embeddings — a few
cents). Skip this if `data/qdrant`/`data/faiss` already have content; `start.command` checks the chunk count
and runs this automatically if it's empty.

Synthetic demo users (`learner-101`…`learner-110`) and the glossary are seeded automatically the first time
the API starts — no separate command needed for those.

## 6. Run the app

Either use the existing script (works as-is, no changes needed — it just needs `.venv/bin/uvicorn` to exist,
which `uv sync` provides):

```bash
./start.command
```

or run the two processes yourself:

```bash
uv run uvicorn src.web_app.api.main:app --host 127.0.0.1 --port 8000 --log-config uvicorn_log_config.json &
npm --prefix frontend run dev
```

Open http://127.0.0.1:5173 (API docs at http://127.0.0.1:8000/docs).

## 7. Run the tests

```bash
uv run pytest -q                    # backend (1,276 tests, ~99% coverage)
uv run pytest --cov=src             # with a coverage report
npm --prefix frontend test          # frontend (42 tests)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv sync` is slow the first time | It's downloading the ~600 MB spaCy model wheel and resolving ~450 packages. Subsequent syncs reuse uv's global cache and are fast. |
| `ModuleNotFoundError` when running a bare `python` command | You forgot `uv run`, or didn't activate `.venv`. `uv run python -c "import fastapi"` should always work from the project root. |
| Embedded Qdrant error: "already accessed by another instance" | Only one process (the API server *or* an ingest run) can hold `data/qdrant` at a time. Stop the running server before re-running `src.rag.ingest`. |
| `uv sync` wants to install a different Python than the one you expected | Pin one explicitly: `uv python install 3.12` then `uv sync --python 3.12`. |
| You already have a `.venv` built the old way (`python3 -m venv` + `pip install -r requirements.txt`) | It keeps working as-is; nothing forces you to switch. If you do want to move it to `uv`, stop any process using it first (e.g. the running API server) before running `uv sync`, since this wasn't tested against a `.venv` in active use. |
