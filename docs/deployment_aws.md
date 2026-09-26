# Deploying ClarityFI to AWS (Tier 1: EC2 + Docker Compose + Caddy)

**Status: LIVE.** Deployed 2026-09-26 — `https://clarityfi.rajur-dev.com` is running on EC2 (`54.91.224.15`, key pair `capstone-delpoy`), serving the real app end to end (backend healthy, RAG ingested, glossary loaded, valid Let's Encrypt cert). See "What actually happened" near the end of this doc for the two real bugs hit along the way — worth reading before repeating this for project #2.

**Why this pattern:** ClarityFI's backend uses an embedded (local-file) Qdrant index and a local SQLite database — both need a persistent disk and a long-running process, which rules out pure serverless (Lambda/Vercel functions). A single EC2 instance with an attached volume, running Docker Compose, with Caddy as a reverse proxy, is the simplest way to host that reliably — and the same box + pattern can host your other capstone projects later, each as its own subdomain.

```
Browser
  │  HTTPS
  ▼
Caddy (auto HTTPS, subdomain routing)
  ├─ clarityfi.yourdomain.com  → localhost:8000 (FastAPI) + static frontend build
  ├─ project2.yourdomain.com   → localhost:8001
  └─ project3.yourdomain.com   → localhost:8002
        │
        ▼
  Docker containers, each with its own persistent volume
```

---

## Phase 1 — AWS console setup

### 1. Launch the EC2 instance
- **Region**: pick one close to you (e.g. `us-east-1`).
- **AMI**: Ubuntu 22.04 LTS (well-documented, works with every guide below verbatim).
- **Instance type**: start with **t3.small** (2 GB RAM). ClarityFI loads a spaCy `en_core_web_lg` model plus an embedded Qdrant index — a `t3.micro` (1 GB) will likely swap/struggle. You can resize later (stop → change instance type → start) with no data loss, since state lives on the attached volume, not the instance itself.
- **Storage**: bump the root volume to **30 GB gp3** (default 8 GB is tight once you add Docker images, the Qdrant index, and spaCy's model files across several projects).
- **Key pair**: create a new one, download the `.pem` file, then `chmod 400 your-key.pem` locally.
- **Security group**: create new, allow:
  - SSH (22) — restrict to "My IP" (not `0.0.0.0/0`) so it's not open to the internet.
  - HTTP (80) — `0.0.0.0/0` (needed for Let's Encrypt's HTTP challenge).
  - HTTPS (443) — `0.0.0.0/0`.
- Launch.

### 2. Allocate an Elastic IP
EC2 → Elastic IPs → Allocate → Associate with your new instance. Without this, the public IP changes every time the instance stops/starts, breaking DNS.

### 3. Point DNS at it
- If you don't have a domain yet, buy one (Route 53, Namecheap, Cloudflare — any registrar works).
- Add an **A record**: `clarityfi.yourdomain.com` → the Elastic IP.
- For each future project, you'll just add another A record (`project2.yourdomain.com` → same Elastic IP) — no new domain purchase needed.

---

## Phase 2 — Server setup (one-time, covers all future projects too)

SSH in: `ssh -i your-key.pem ubuntu@<elastic-ip>`

```bash
sudo apt update && sudo apt upgrade -y

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# log out and back in for the group change to apply

# Caddy (native install, not in Docker — it needs to reverse-proxy to
# every project's containers by localhost port, which is simplest as a
# host-level service)
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

Verify: `docker --version`, `docker compose version`, `caddy version`.

---

## Phase 3 — Get ClarityFI onto the server

### 4. Copy the code
Two options, since pushing this repo to GitHub is a separate decision from deploying it:
- **Without GitHub**: `rsync -avz --exclude node_modules --exclude .venv --exclude data -e "ssh -i your-key.pem" /Users/rramachandra/raju/CAPSTONE2/clarityFI/ ubuntu@<elastic-ip>:/srv/clarityfi/`
- **With GitHub** (once you're ready to push): `git clone <repo-url> /srv/clarityfi` on the server, then `git pull` to update later.

### 5. Secrets
Never commit `.env` or put it in the rsync/git payload. Instead, create it directly on the server:
```bash
ssh -i your-key.pem ubuntu@<elastic-ip>
nano /srv/clarityfi/.env   # paste real keys directly on the server
```

### 6. Files this needed (now in the repo)
- **`Dockerfile`** — single-stage `python:3.12-slim`, installs `requirements.txt` (which pulls the pinned `en_core_web_lg` spaCy wheel directly by URL — no separate download step), copies only `src/`, `config.yaml`, `uvicorn_log_config.json`. The frontend is **not** built in this image — it's served as static files straight off the host by Caddy, so the image stays backend-only and small.
- **`.dockerignore`** — keeps the build context lean. **Gotcha hit and fixed**: every directory pattern must be anchored with a leading `/` (e.g. `/data/`, not `data/`). An unanchored `data/` also matches `src/data/` — a real Python package — and silently drops it from the build, which is exactly what happened the first time (see "What actually happened" below).
- **`docker-compose.yml`** — one `backend` service, port bound to `127.0.0.1:8000` only (never exposed directly to the internet — only Caddy on the same host can reach it), `.env` loaded via `env_file`, `./data:/app/data` bind-mounted so `clarityfi.db`, `glossary.db`, the Qdrant index, and the FAISS fallback all persist across rebuilds, `restart: unless-stopped`, a `HEALTHCHECK` hitting `/api/health`.
- **The Caddyfile** (`/etc/caddy/Caddyfile` on the server, replacing the default placeholder):
  ```
  clarityfi.rajur-dev.com {
      handle /api/* { reverse_proxy localhost:8000 }
      handle { root * /srv/clarityfi/frontend/dist; file_server; try_files {path} /index.html }
      encode gzip
  }
  ```
  Validate before applying: `sudo caddy validate --config /etc/caddy/Caddyfile`, then `sudo systemctl reload caddy`. Caddy issues the Let's Encrypt cert automatically the moment it sees real traffic for a domain whose DNS already resolves to this box — no manual cert step.

### 7. First deploy
```bash
cd /srv/clarityfi
docker compose build
docker compose up -d
# one-time: ingest the knowledge base into the fresh, empty volume.
# IMPORTANT: the running container already holds the embedded-Qdrant lock, so
# `docker compose exec backend python -m src.rag.ingest` while it's up will fail
# with "Storage folder ... already accessed by another instance". Stop first,
# run ingest as a one-off container, then bring the service back up:
docker compose stop
docker compose run --rm backend python -m src.rag.ingest --reset
docker compose up -d
```

### 8. Verify
- `docker compose ps` — should show `Up ... (healthy)`.
- `curl http://127.0.0.1:8000/api/health` on the server — check `status: "ready"`, `knowledge_chunks` > 0, `glossary_terms` > 0.
- Visit `https://clarityfi.yourdomain.com` from anywhere — Caddy should show a valid HTTPS cert automatically.
- Sign in with a demo account to confirm the full stack (DB, RAG, auth) works end to end.
- Restart the container (`docker compose restart`) and confirm your data (saved goals, chat history) survived — proves the volume is actually persistent, not just working by luck.
- `systemctl is-enabled docker` and `systemctl is-enabled caddy` should both say `enabled`, so a server reboot brings everything back on its own (`restart: unless-stopped` handles the container itself).

---

## Adding project #2 onward (once this pattern is proven)
1. `rsync`/`git clone` the new project to `/srv/project2/`.
2. Give it its own `docker-compose.yml` with a **different host port** (e.g. `8001:8000` inside the container).
3. Add its own `.env` on the server.
4. Add one Caddyfile block + one DNS A record for `project2.yourdomain.com`.
5. `docker compose up -d --build`, reload Caddy. Done — no new server, no new domain, no new AWS account setup.

---

## Cost estimate
- `t3.small` on-demand: roughly **$15/mo** (less if you stop it outside of review windows — storage-only cost while stopped is a few cents/GB/month).
- Elastic IP: free while attached to a running instance.
- Domain: ~$10-15/year, one-time shared cost across all projects.
- No per-project recurring cost beyond the one box, until it's genuinely out of capacity for a 6th–10th project.

---

## What actually happened (real deploy log, 2026-09-26)

Two real bugs came up during the first deploy — both worth knowing before repeating this for project #2:

1. **`.dockerignore`'s unanchored `data/` pattern excluded `src/data/` too.** The container built and ran, but crashed on startup with `ModuleNotFoundError: No module named 'src.data'`. `.dockerignore` (like `.gitignore`) matches an unanchored pattern at *any* depth — `data/` meant to exclude the top-level runtime data folder also matched the real `src/data/` Python package (`repositories.py`, `db.py`, `seed.py`, `glossary_store.py`, the KB JSON). Fixed by anchoring every directory pattern with a leading `/` (`/data/`, `/tests/`, `/docs/`, etc.). **If you hit an unexplained `ModuleNotFoundError` for a package you know exists in the repo, suspect an overly-broad `.dockerignore`/`.gitignore`-style exclude first.**

2. **The initial code copy to the server was missing `src/data/` entirely — independent of Docker.** The server's `.git` had zero commits (`git log` → "does not have any commits yet", every file showed as `??` untracked) — the code got there via a plain file copy, not a real `git clone`. A file-by-file diff (`find src -type f` locally vs. on the server) showed only `src/data/` (11 files) was missing everywhere else matched. Almost certainly the same class of bug as #1, in whatever `rsync`/copy exclude pattern was used outside this session. **Lesson for future projects: after any non-git copy to a server, diff the file lists — don't assume the copy was complete just because the app half-starts.**

3. **The embedded-Qdrant single-process lock applies inside Docker too.** Running `docker compose exec backend python -m src.rag.ingest` while the service is up fails with `Storage folder ... already accessed by another instance` — the running uvicorn process already holds it. Fix: `docker compose stop` → `docker compose run --rm backend python -m src.rag.ingest --reset` (a fresh one-off container, old one not running) → `docker compose up -d` again. This is the same constraint that's applied all along in local dev, just showing up in a new context.

4. **`data/glossary.db` is git-ignored** (`data/*.db` in `.gitignore`) and isn't rebuilt automatically at startup (the auto-seed path expects a `data/financial_glossary.json` that doesn't exist locally — the real 203-term glossary only exists as the already-built `.db` file). It has to be copied to the server separately, once: `scp data/glossary.db ubuntu@<ip>:/srv/clarityfi/data/glossary.db`. Worth fixing properly later (commit a real seed JSON, or stop gitignoring this one `.db` file) so project #2 onward doesn't need this manual step.

Everything else — EC2 launch, Elastic IP, DNS, Docker/Caddy install, the Dockerfile/Compose/Caddyfile themselves, the RAG ingest, HTTPS issuance — worked exactly as planned on the first try.
