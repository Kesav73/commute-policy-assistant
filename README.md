# Commute Policy Assistant

An assistant that answers employees' commute and transport questions **only from their own company's policy**, cites the document and section, and says plainly when the policy doesn't cover something.

Built for the MoveInSync AI Developer case study B. Work in progress: a full README with design decisions, trade-offs and evaluation results will follow.

## Stack
Python · FastAPI · Chroma (embedded) · all-MiniLM-L6-v2 embeddings (local ONNX) · Groq (`qwen/qwen3.8-27b`, fallback `openai/gpt-oss-120b`) · JWT (HS256)

## Run it: one command
```bash
docker compose up
```
Then open **http://localhost:8000/docs**. The first build takes ~2–3 minutes (the embedding model is baked into the image); after that, startup takes ~5 seconds.

- **No API key needed.** Without one, the service runs in **offline mode** with a deterministic fake LLM, and `/health` shows `"llm_provider": "fake"`.
- **Real answers:** `cp .env.example .env`, set `GROQ_API_KEY` (free at https://console.groq.com/keys), then `docker compose up` again.

## Check it (inside the container)
```bash
docker compose run --rm app pytest -q                                # 138 tests, never call a real LLM
docker compose run --rm app python -m eval.run_retrieval             # retrieval scorecard: recall@1, recall@3, avg latency
```

## Try it
1. `POST /auth/token` with `{"employee_id": "E-ACME-001"}` (Bolt: `E-BOLT-001`, Crest: `E-CRST-001`) and copy `access_token`.
2. Click **Authorize** in `/docs` and paste the token.
3. `POST /ask` with `{"question": "Can I bring a guest in the company cab?"}`. Repeat with a Bolt token to see a different, correctly cited answer.

<details><summary>Without Docker</summary>

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app
.venv/bin/python -m pytest -q
```
</details>

## API
| Endpoint | Purpose |
|---|---|
| `POST /auth/token` | Mock identity provider: `{"employee_id": "E-ACME-001"}` → JWT |
| `POST /ask` | `{"question": "..."}` with `Authorization: Bearer <token>` → answer + citations |
| `GET /health` | Status, LLM mode, chunks loaded per company |

## Docs
- [Problem statement and requirement checklist](docs/PS.md)
- [High-level design](docs/HLD.md)
- [Policy ground truth](docs/POLICY_MATRIX.md)
- [Engineering journey: every experiment and its results](docs/journey/README.md)
