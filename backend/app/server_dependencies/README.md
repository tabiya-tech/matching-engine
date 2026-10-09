# Database access

The matching service reads one MongoDB collection: the ranked/enriched job ads written by the
llm-reranker. This document covers how the connection is owned, which collections and indexes exist,
and how to add a query. The layout follows Compass's backend (`app/server_dependencies/` +
per-feature repositories).

```
app/
├── server_dependencies/
│   ├── db_dependencies.py       MatchingDBProvider: the one client; settings; JOBS_INDEX_MODELS
│   └── database_collections.py  Collections: collection names
├── jobs/
│   ├── repository.py            IJobsRepository: the interface routes and services depend on (and tests mock)
│   ├── mongo_repository.py      MongoJobsRepository: the Mongo implementation, every query on the jobs collection
│   ├── get_jobs_repository.py   get_jobs_repository (FastAPI dependency), standalone_jobs_repository
│   ├── job_document.py          projection + build_job_dict_from_ranked (stored doc → flat job dict)
│   ├── browse_filter.py         GET /jobs filter builder
│   └── pagination.py            GET /jobs keyset cursor codec
├── services/
│   ├── location_query.py        /match location prefilter (pure query building)
│   └── job_retrieval.py         retrieve_jobs_with_timing: prefilter policy + repository read
├── occupations/                 occupation corpus (local JSON/NPZ files, not Mongo)
├── warmup.py                    startup work: ping, indexes, caches, models
└── main.py                      lifespan: creates and closes the client
```

## Connection lifecycle

There is exactly one `AsyncIOMotorClient` per process, owned by `MatchingDBProvider`. No module
creates a client at import time, so importing the app never needs `MONGO_URL`.

1. **Startup** (`app.main.lifespan`): `await MatchingDBProvider.get_jobs_db()` reads the settings
   and creates the client. Motor connects lazily, so this does no I/O.
2. **Warmup** (`app.warmup.warmup_on_startup(jobs_db)`): pings Mongo, then calls
   `MatchingDBProvider.initialize_jobs_mongo_db` to ensure indexes, then preloads the occupation
   corpus and (optionally) the WA lookup and models. Every step is toggled by an env flag and is
   **non-fatal**: a failure is logged and the app still starts. With `WARMUP_NON_BLOCKING=1` it runs
   as a background task.
3. **Requests**: routes declare `jobs_repository: IJobsRepository = Depends(get_jobs_repository)`.
   FastAPI resolves `Depends(MatchingDBProvider.get_jobs_db)` (which returns the cached database) and
   builds a `JobsRepository` around it.
4. **Shutdown**: the lifespan cancels a still-running background warmup, then calls
   `MatchingDBProvider.close()`, which closes the client and forgets it.

`get_jobs_db()` creates the client under an `asyncio.Lock` with a double check, so concurrent first
callers share one client.

### Settings

Read from the environment on first use (`MongoDbSettings.from_env`); see `backend/.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `MONGO_URL` | — (required) | Connection string |
| `MONGO_DB_NAME` | — | Database (one per country/stack) |
| `MONGO_JOBS_COLLECTION` | `RankedJobsEnriched` | Jobs collection (read via `app.config`) |
| `MONGO_SERVER_SELECTION_TIMEOUT_MS` | `30000` | Server selection timeout |
| `MONGO_MAX_POOL_SIZE` / `MONGO_MIN_POOL_SIZE` | `50` / `0` | Connection pool bounds |
| `MONGO_TLS_CA_FILE` | — | CA bundle path, or `certifi` for `certifi.where()` |
| `MONGO_TLS_ALLOW_INVALID_CERTIFICATES` / `MONGO_TLS_INSECURE` | off | Skip TLS verification (dev only) |

A TLS URI (`mongodb+srv://`, `tls=true`) without `MONGO_TLS_CA_FILE` uses the `certifi` bundle when
it is installed.

### Warmup flags

| Variable | Default | Step |
|---|---|---|
| `MONGO_WARMUP_ON_STARTUP` | on | Ping Mongo |
| `ENSURE_INDEXES_ON_STARTUP` | on | Create `JOBS_INDEX_MODELS` |
| `WARMUP_OCCUPATIONS_CACHE` | on | Load + flatten the occupation corpus |
| `WARMUP_WA_LOOKUP` | off | Build the O*NET work-activity lookup |
| `WARMUP_MATCH_V3_MODELS` | off | Preload the /match_v3 matcher + cross-encoder |
| `WARMUP_NON_BLOCKING` | off | Run the warmup in the background |

### Outside the app (scripts)

CLI scripts have no lifespan. They open a short-lived client bound to their own event loop:

```python
from app.jobs.get_jobs_repository import standalone_jobs_repository
from app.services.job_retrieval import retrieve_jobs_with_timing

async with standalone_jobs_repository() as jobs_repository:
    jobs, timing = await retrieve_jobs_with_timing(jobs_repository, users)
```

The client is closed when the block exits. Never use this inside the app.

## Collections

| `Collections.` | Name | Written by | Read by |
|---|---|---|---|
| `JOBS` | `MONGO_JOBS_COLLECTION` (default `RankedJobsEnriched`) | llm-reranker | `JobsRepository` |

The document shape is the contract with the reranker. Only fields in `RANKED_JOB_FIND_PROJECTION`
are read (when `JOBS_FIND_USE_PROJECTION` is on), and `build_job_dict_from_ranked` maps them to the
flat dict every engine consumes. Both live in `app/jobs/job_document.py` and must stay aligned: a new
reranker field reaches matching only if it is projected and mapped there (tests in
`tests/unit/test_build_job_dict_from_ranked.py`).

Occupations are **not** in Mongo. They come from `OCCUPATION_JSON_PATH` and the concat-embeddings
NPZ, are flattened by `app.occupations.flatten`, and are cached in-process by
`app.occupations.loader`.

### Indexes

`JOBS_INDEX_MODELS` in `db_dependencies.py` is the full set, created idempotently at startup. Each
leads with `is_active`, because every query filters on it:

| Name | Keys | Serves |
|---|---|---|
| `is_active_-_id` | `is_active, _id desc` | /match active load, /jobs browse sort + cursor seek, counts |
| `is_active_employment_type_-_id` | `is_active, employment_type, _id desc` | /jobs filtered by employment type |
| `is_active_category` | `is_active, category` | /jobs/stats `distinct` |
| `is_active_isco_group` | `is_active, isco_occupation_group` | /jobs/stats `distinct` |
| `is_active_source_platform` | `is_active, source_platform` | /jobs/stats `distinct` |

Regex filters (title, location, skills) cannot use these and run as residual filters.

## Adding a query

1. **Declare it on the interface.** Add an abstract method to `IJobsRepository`
   (`app/jobs/repository.py`) with a docstring that states its filter, sort and return shape. Routes
   and services depend on the interface, so this is also what tests mock.
2. **Implement it in `MongoJobsRepository`** (`app/jobs/mongo_repository.py`). Use `self._collection`. Do not open a client or reach for
   the provider. Lead the filter with `RANKED_JOBS_ACTIVE_FILTER` unless the query really needs
   inactive jobs, use `self._find(...)` to apply the standard projection, and shape documents with
   `build_job_dict_from_ranked` so callers get the same job object as /match.
3. **Keep domain rules out of it.** If the filter depends on user profiles or business policy (like
   the location prefilter), build it in `app/services/` and pass it in, as
   `retrieve_jobs_with_timing` does for `find_jobs_with_timing`. Pure filter builders for a single
   endpoint can live next to the repository (like `browse_filter.py`).
4. **Back it with an index** if it filters or sorts on a new field: add an `IndexModel` to
   `JOBS_INDEX_MODELS` with a name and a comment on the access pattern it serves.
5. **Use it from a route** by taking `jobs_repository: IJobsRepository = Depends(get_jobs_repository)`
   as a parameter.
6. **Test it offline.**
   - Query construction: `tests/unit/test_jobs_repository.py` runs the repository against an
     in-process fake collection that records `find`/`sort`/`limit`.
   - Route behaviour: the `test_client` fixture overrides `get_jobs_repository` with
     `mocked_jobs_repository` (an `AsyncMock` of `IJobsRepository`). Set
     `mocked_jobs_repository.<method>.side_effect` in the test.

### Adding a collection

Add the name to `Collections`, create a repository package like `app/jobs/` (interface,
implementation, `get_…_repository` dependency taking `Depends(MatchingDBProvider.get_jobs_db)`), and
add its indexes to a `MatchingDBProvider.initialize_…` method called from `app.warmup`. If it lives
in a different database or cluster, add a `get_…_db` getter to the provider alongside `get_jobs_db`.
Then close that client in `MatchingDBProvider.close()`.
