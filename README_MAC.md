# ClarityFI – local Mac setup

Prerequisites: Python 3.12+, Node.js 20+, an OpenAI API key.

1. `cp .env.example .env` and set `OPENAI_API_KEY` (optionally `ALPHAVANTAGE_API_KEY`, `SERPAPI_API_KEY`, Langfuse/LangSmith keys). `.env` is git-ignored; never share it.
2. Double-click `start.command` (right-click > Open if macOS blocks it). First run creates `.venv`, installs Python and npm packages plus the spaCy model, and embeds the knowledge base with OpenAI (small cost).
3. The browser opens at http://127.0.0.1:5173. Sign in with a demo account (`learner-101` … `learner-110`; the password is shown on the sign-in screen, default `clarity-demo`) or create your own account. Stop with Control+C.

Services: React http://127.0.0.1:5173 · FastAPI http://127.0.0.1:8000 (`/docs`) · data in `data/` (Qdrant + FAISS indexes, app database, glossary).

If something is already listening on port 8000 (an older run), stop it first, or the new API cannot start and the UI will show "Not Found" for newer features.

Voice: the browser converts speech to text (Chrome/Safari) and reads answers aloud. Tick **HD voice (OpenAI)** in Finance Q&A to use Whisper and OpenAI text-to-speech instead; audio is processed in memory and never stored.

See README.md for the full guide, API and troubleshooting. Educational use only; not personalized financial, legal, investment or tax advice.
