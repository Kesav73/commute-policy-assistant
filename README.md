# Commute Policy Assistant

An assistant that answers employees' commute and transport questions **only from their own company's policy**, cites the document and section, and says plainly when the policy doesn't cover something.

Built for the MoveInSync AI Developer case study B. Work in progress: a full README with design decisions, trade-offs and evaluation results will follow.

## Stack
Python · FastAPI · Chroma (embedded) · nomic-embed-text-v1.5 embeddings (local ONNX via fastembed) · Groq (`qwen/qwen3.8-27b`, fallback `openai/gpt-oss-120b`) · JWT (HS256)

## Run it: one command
```bash
docker compose up
```
Then open **http://localhost:8000/docs**. The first build takes a few minutes (the ~0.5 GB embedding model is baked into the image; image ~1.9 GB, ~1.3 GB RAM); after that, startup takes ~5 seconds.

- **No API key needed.** Without one, the service runs in **offline mode** with a deterministic fake LLM, and `/health` shows `"llm_provider": "fake"`.
- **Real answers:** `cp .env.example .env`, set `GROQ_API_KEY` (free at https://console.groq.com/keys), then `docker compose up` again.

## Check it (inside the container)
```bash
docker compose run --rm app pytest -q                                # full test suite (~10 s), never calls a real LLM
docker compose run --rm app python -m eval.run_retrieval             # retrieval scorecard: recall@1, recall@3, avg latency
docker compose run --rm app python -m eval.run_e2e --offline         # end-to-end flow with the fake LLM (plumbing check)
docker compose run --rm app python -m eval.run_e2e                   # end-to-end with the real LLM (needs key, ~12 min)
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

## Update a policy (no restart)
1. `POST /auth/token` with `{"employee_id": "A-ACME-ADM"}` (an Acme admin).
2. `POST /admin/policies` with `{"markdown": "<contents of sample_data/policy_updates/acme_v2.md>"}`. It moves the women's escort from 8 PM to 9 PM → `{"version": 2, "previous_version": 1}`.
3. Ask an Acme question again: the answer and `policy_version` come from v2 immediately. A broken file (`sample_data/policy_updates/acme_broken.md`) is rejected with 400 and v2 stays live.

The response lists what changed (`"changed": ["4.1 …", "4.2 …"]`). Repeated questions are answered from a cache keyed by company and policy version (`"cached": true`, 0 LLM calls); after an update the old answers can't be served.

Policy versions live in the `app-data` volume and survive restarts. `docker compose down -v` resets to the seed policies in `policies/`.

## API
| Endpoint | Purpose |
|---|---|
| `POST /auth/token` | Mock identity provider: `{"employee_id": "E-ACME-001"}` → JWT |
| `POST /ask` | `{"question": "..."}` with `Authorization: Bearer <token>` → answer + citations |
| `GET /health` | Status, LLM mode, chunks loaded and live policy version per company |
| `POST /admin/policies` | Admin token (e.g. `A-ACME-ADM`) + `{"markdown": "..."}` → publishes a new version of the admin's company policy on the running service; reports which sections changed |

## Docs
- [Problem statement and requirement checklist](docs/PS.md)
- [High-level design](docs/HLD.md)
- [Policy ground truth](docs/POLICY_MATRIX.md)
