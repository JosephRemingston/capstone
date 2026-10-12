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

Semantic retrieval uses the exact SQLite cosine scan (one numpy matrix product)
by default. ANN is opt-in: with `COGNIMEM_ANN_ENABLED=true` and a user corpus at
or above `COGNIMEM_ANN_EXACT_THRESHOLD` (default `5000`; values below 1 are
rejected at startup), semantic retrieval uses the `ApproximateVectorIndex`,
implemented as deterministic random-hyperplane LSH with multi-table,
query-directed multi-probe candidate generation. It widens the probe radius
until it has `max(limit, 400)` candidates and otherwise falls back to the exact
scan; each hit reports the path taken under `vector_index`. The SQLite vector table
remains authoritative and the LSH tables are rebuildable derived state.
ANN is experimental: on real conversational BGE vectors at about 5.9k chunks its
top-20 agreed with exact search only about 41% of the time, so keep it off unless
exact scans are too slow.
