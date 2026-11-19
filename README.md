# Commute Policy Assistant

An assistant that answers employees' commute and transport questions **only from their own company's policy**, cites the document and section, and says plainly when the policy doesn't cover something.

Built for the MoveInSync AI Developer case study B. Work in progress: a full README with design decisions, trade-offs and evaluation results will follow.

## Stack
Python · FastAPI · Chroma (embedded) · all-MiniLM-L6-v2 embeddings (local ONNX) · Groq (`qwen/qwen3.8-27b`, fallback `openai/gpt-oss-120b`) · JWT (HS256)

## Run locally (current state)
```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env              # optional: add GROQ_API_KEY; without it the app runs offline with a fake LLM
.venv/bin/uvicorn app.main:app    # ingests policies/ at startup, then open http://localhost:8000/docs
.venv/bin/python -m pytest -q     # tests never call a real LLM
.venv/bin/python -m eval.run_retrieval   # 90-question retrieval evaluation
```

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
