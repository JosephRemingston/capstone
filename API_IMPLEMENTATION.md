# CogniMem API implementation

This change set implements **only** the six requested features:

1. REST API
2. Lightweight demo UI
3. Authentication/authorization
4. User export/deletion
5. Monitoring/metrics/logging/health checks
6. Approximate-nearest-neighbor vector search

The following remaining TODOs are intentionally untouched: production database
migration, deployment setup, background reminder delivery, broader entity/coreference/relationship extraction, and human feedback.

## Start

```bash
python -m pip install -r requirements.txt
export COGNIMEM_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
python -m api
```

Open `/demo/` for the browser demo or `/docs` for the OpenAPI UI.

## API surface

- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `GET /api/v1/me`
- `POST /api/v1/memories`
- `GET /api/v1/memories`
- `GET /api/v1/memories/{id}`
- `POST /api/v1/memories/search`
- `GET /api/v1/me/export?format=json|csv`
- `DELETE /api/v1/me`
- `GET /health/live`
- `GET /health/ready`
- `GET /metrics`

Protected endpoints use Bearer JWT authentication. Memory ownership is always
derived from the token subject. Passwords are stored using Argon2id. API errors
use a consistent JSON envelope with a request ID.

## ANN behavior

Small corpora use the existing exact SQLite cosine search. At or above
`COGNIMEM_ANN_EXACT_THRESHOLD` (default `256`), semantic retrieval uses the
`ApproximateVectorIndex`, implemented as deterministic random-hyperplane LSH
with multi-table and multi-probe candidate generation. The SQLite vector table
remains authoritative and the LSH tables are rebuildable derived state.
