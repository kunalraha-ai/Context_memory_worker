# 🐝 Hive — Context Memory Worker

A **Worker Bee** node for the Hive autonomous agent framework. Provides persistent long-term vector memory to the **Queen Bee** orchestrator using a fully local stack — no cloud, no API keys, no C++ compiler required.

---

## Stack

| Layer | Technology |
|---|---|
| Vector Database | [LanceDB](https://lancedb.github.io/lancedb/) — local persistent |
| Embeddings | [Ollama](https://ollama.com) · `nomic-embed-text` (768-dim) |
| Similarity | Cosine distance via LanceDB HNSW |
| Runtime | Python 3.12 |
| Container | Docker + Docker Compose |

---

## Project Structure

```
Context_memory_worker/
├── engine.py      # Worker Bee node (main source)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Compose orchestration
├── .dockerignore         # Build context exclusions
└── hive_memory/          # LanceDB data (auto-created, git-ignored)
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.12 recommended |
| [Ollama](https://ollama.com/download) | Must be running locally |
| Docker Desktop | Only needed for containerised usage |

Pull the required embedding model once:

```bash
ollama pull nomic-embed-text
```

---

## Quick Start

### Option A — Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Run the smoke-test
python engine.py
```

### Option B — Run with Docker

```bash
# 1. Start Ollama on your host
ollama serve

# 2. Build and run the container
docker compose up --build

# 3. Run detached (background)
docker compose up --build -d

# 4. View live logs
docker compose logs -f memory_worker

# 5. Stop
docker compose down
```

---

## API Reference

All interaction goes through the `process(payload: dict) → dict` method.

### `store` — Save a context chunk

**Input**
```json
{
    "action": "store",
    "text": "The user prefers concise bullet-point responses.",
    "metadata": {
        "source": "conversation_turn_12",
        "tags": ["preference", "communication"]
    }
}
```

**Output**
```json
{
    "status": "success",
    "action": "store",
    "doc_id": "df0955de-82f7-aabc-24ea-9c2ad5528f45",
    "message": "Stored 1 document (id=df0955de-...)."
}
```

---

### `query` — Retrieve relevant context

**Input**
```json
{
    "action": "query",
    "text": "What are the user's communication preferences?",
    "top_k": 3
}
```

**Output**
```json
{
    "status": "success",
    "action": "query",
    "results": [
        {
            "rank": 1,
            "text": "The user prefers concise bullet-point responses.",
            "distance": 0.354988,
            "metadata": {
                "source": "conversation_turn_12",
                "tags": "preference, communication",
                "timestamp": "2026-03-03T17:44:09.269821+00:00"
            }
        }
    ],
    "count": 1
}
```

> `distance` is cosine distance — **lower = more relevant**. Range: `0.0` (identical) → `2.0` (opposite).

---

### Error Response (all actions)

```json
{
    "status": "error",
    "action": "delete",
    "error_type": "InvalidActionError",
    "message": "Unknown action 'delete'. Valid actions are: 'store', 'query'."
}
```

| `error_type` | Cause |
|---|---|
| `InvalidActionError` | Action is not `store` or `query` |
| `ValidationError` | `text` field is missing or empty |
| `OllamaUnreachableError` | Ollama server is not running or unreachable |
| `EmptyCollectionError` | Query attempted before any documents are stored |

---

## Registering in a Hive Graph

```python
from memory_worker import ContextMemoryWorker

# Initialise the node
worker = ContextMemoryWorker(
    persist_dir="./hive_memory",       # LanceDB storage path
    ollama_base_url="http://localhost:11434",
    embed_model="nomic-embed-text",
)

# Register with your Hive orchestrator
hive.register_node("memory_worker", worker)

# The Queen Bee can now call:
result = worker.process({
    "action": "query",
    "text": "What did the user say about the project budget?",
})
```

---

## Configuration

All defaults can be overridden in the constructor or via environment variables in Docker.

| Parameter | Default | Description |
|---|---|---|
| `persist_dir` | `./hive_memory` | LanceDB storage directory |
| `table_name` | `hive_context` | LanceDB table name |
| `ollama_base_url` | `http://localhost:11434` | Ollama server URL |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |

**Docker environment override:**
```yaml
environment:
  OLLAMA_BASE_URL: "http://host.docker.internal:11434"
```

---

## How It Works

```
Queen Bee Orchestrator
        │
        │  JSON payload  {"action": "store" / "query", "text": "..."}
        ▼
┌─────────────────────────┐
│  ContextMemoryWorker    │
│  .process(payload)      │
│                         │
│  1. Validate input      │
│  2. Call Ollama         │──► POST /api/embeddings → 768-dim vector
│  3. LanceDB upsert/     │
│     cosine search       │──► ./hive_memory/ (persisted to disk)
│  4. Return JSON         │
└─────────────────────────┘
        │
        │  JSON response {"status": "success", "results": [...]}
        ▼
Queen Bee Orchestrator
```

**Deduplication:** Document IDs are SHA-256 hashes of the text content. Re-storing identical text is a no-op (idempotent upsert).

---

## Data Persistence

LanceDB writes to `./hive_memory/` as Lance columnar files. In Docker this directory is mounted as a named volume (`hive_memory`) so data survives:

- Container restarts
- Image rebuilds (`docker compose up --build`)
- `docker compose down` (data is **not** deleted unless you run `docker compose down -v`)

---

## Troubleshooting

**`OllamaUnreachableError` on startup**
```bash
# Make sure Ollama is running
ollama serve

# Verify the model is pulled
ollama list
```

**`EmptyCollectionError` on query**
> You must `store` at least one document before querying.

**`DeprecationWarning: table_names() is deprecated`**
> Harmless warning from LanceDB internals. Does not affect functionality.

**Docker: Ollama unreachable from container**
> On Linux, ensure `extra_hosts: ["host.docker.internal:host-gateway"]` is present in `docker-compose.yml` (already included).

---

## License

MIT
