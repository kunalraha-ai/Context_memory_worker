"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             ADEN HIVE — Context Memory Worker Node                          ║
║                                                                              ║
║  Role    : Worker Bee  (long-term memory provider for the Queen Bee)        ║
║  Storage : LanceDB    (local persistent client → ./hive_memory)             ║
║  Embed   : Ollama     nomic-embed-text  @ http://localhost:11434             ║
║  Actions : store | query                                                     ║
║                                                                              ║
║  ✅ Windows / macOS / Linux — no C++ compiler required                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
import lancedb
import pyarrow as pa

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("hive.memory_worker")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str        = "http://localhost:11434"
EMBED_MODEL: str            = "nomic-embed-text"
LANCE_PERSIST_DIR: str      = "./hive_memory"
TABLE_NAME: str             = "hive_context"
TOP_K_RESULTS: int          = 3
OLLAMA_TIMEOUT_SECONDS: int = 30
OLLAMA_HEALTH_RETRIES: int  = 3
OLLAMA_HEALTH_RETRY_DELAY: float = 2.0   # seconds

# nomic-embed-text outputs 768-dimensional vectors
EMBEDDING_DIM: int = 768


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────
class OllamaUnreachableError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""

class EmptyCollectionError(RuntimeError):
    """Raised when a query is attempted on an empty table."""

class MemoryWorkerError(RuntimeError):
    """Generic wrapper for unexpected MemoryWorker failures."""


# ─────────────────────────────────────────────────────────────────────────────
# Context Memory Worker Node
# ─────────────────────────────────────────────────────────────────────────────
class ContextMemoryWorker:
    """
    Aden Hive Worker Bee — long-term context memory node.

    Registration example
    --------------------
    .. code-block:: python

        from memory_worker import ContextMemoryWorker

        worker = ContextMemoryWorker()
        hive.register_node("memory_worker", worker)

    Input schema (``process`` method)
    ----------------------------------
    Store:
    .. code-block:: json

        {
            "action": "store",
            "text": "<chunk of text to memorise>",
            "metadata": {
                "source": "conversation_turn_12",
                "tags":   ["user_preference", "project_alpha"]
            }
        }

    Query:
    .. code-block:: json

        {
            "action": "query",
            "text":   "<natural language question>",
            "top_k":  3
        }

    Output schema
    -------------
    Success (store):
        {"status": "success", "action": "store", "doc_id": "...", "message": "..."}

    Success (query):
        {"status": "success", "action": "query", "results": [...], "count": 3}

    Error:
        {"status": "error", "action": "...", "error_type": "...", "message": "..."}
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        persist_dir: str     = LANCE_PERSIST_DIR,
        table_name: str      = TABLE_NAME,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embed_model: str     = EMBED_MODEL,
    ) -> None:
        self._persist_dir = persist_dir
        self._table_name  = table_name
        self._ollama_base = ollama_base_url.rstrip("/")
        self._embed_model = embed_model
        self._embed_url   = f"{self._ollama_base}/api/embeddings"

        logger.info(
            "Initialising ContextMemoryWorker — persist_dir=%s  table=%s",
            self._persist_dir, self._table_name,
        )
        self._verify_ollama_health()
        self._db, self._table = self._init_lance()
        logger.info(
            "ContextMemoryWorker ready — %d document(s) in table.",
            self._table.count_rows(),
        )

    # --------------------------------------------------------- public process
    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Main entry-point for the Hive graph executor."""
        action: str = str(payload.get("action", "")).strip().lower()
        logger.info("MemoryWorker received action='%s'", action)

        try:
            if action == "store":
                return self._handle_store(payload)
            elif action == "query":
                return self._handle_query(payload)
            else:
                return self._error_response(
                    error_type="InvalidActionError",
                    action=action or "unknown",
                    message=(
                        f"Unknown action '{action}'. "
                        "Valid actions are: 'store', 'query'."
                    ),
                )
        except OllamaUnreachableError as exc:
            logger.error("Ollama unreachable: %s", exc)
            return self._error_response("OllamaUnreachableError", action, str(exc))
        except EmptyCollectionError as exc:
            logger.warning("Table is empty: %s", exc)
            return self._error_response("EmptyCollectionError", action, str(exc))
        except Exception as exc:                            # noqa: BLE001
            logger.exception("Unexpected error: %s", exc)
            return self._error_response(type(exc).__name__, action, str(exc))

    # ------------------------------------------------------- action handlers
    def _handle_store(self, payload: dict[str, Any]) -> dict[str, Any]:
        text: str = payload.get("text", "").strip()
        if not text:
            return self._error_response(
                "ValidationError", "store",
                "'text' field is required and must be non-empty for action 'store'.",
            )

        meta: dict = dict(payload.get("metadata") or {})
        source: str    = str(meta.get("source", "unknown"))
        tags: str      = self._coerce_tags(meta.get("tags", ""))
        timestamp: str = str(meta.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        ))

        doc_id: str = self._text_to_id(text)
        vector      = self._embed(text)

        record = {
            "doc_id":    doc_id,
            "text":      text,
            "vector":    vector,
            "source":    source,
            "tags":      tags,
            "timestamp": timestamp,
        }

        # Idempotent upsert: remove old row with same hash, then insert
        try:
            self._table.delete(f"doc_id = '{doc_id}'")
        except Exception:
            pass  # table may have no matching row — that's fine
        self._table.add([record])

        logger.info("Stored document id=%s  text_len=%d", doc_id, len(text))
        return {
            "status":  "success",
            "action":  "store",
            "doc_id":  doc_id,
            "message": f"Stored 1 document (id={doc_id}).",
        }

    def _handle_query(self, payload: dict[str, Any]) -> dict[str, Any]:
        question: str = payload.get("text", "").strip()
        if not question:
            return self._error_response(
                "ValidationError", "query",
                "'text' field is required and must be non-empty for action 'query'.",
            )

        top_k: int     = max(1, min(int(payload.get("top_k", TOP_K_RESULTS)), 20))
        row_count: int = self._table.count_rows()

        if row_count == 0:
            raise EmptyCollectionError(
                "The memory table is empty. "
                "Store some context first using action='store'."
            )

        effective_k = min(top_k, row_count)
        query_vec   = self._embed(question)

        logger.info(
            "Querying table (rows=%d)  top_k=%d  question_len=%d",
            row_count, effective_k, len(question),
        )

        rows = (
            self._table
            .search(query_vec)
            .metric("cosine")
            .limit(effective_k)
            .to_list()
        )

        formatted = [
            {
                "rank":     rank,
                "text":     row["text"],
                "distance": round(float(row.get("_distance", 0.0)), 6),
                "metadata": {
                    "source":    row["source"],
                    "tags":      row["tags"],
                    "timestamp": row["timestamp"],
                },
            }
            for rank, row in enumerate(rows, start=1)
        ]

        return {
            "status":  "success",
            "action":  "query",
            "results": formatted,
            "count":   len(formatted),
        }

    # --------------------------------------------------- initialisation
    def _verify_ollama_health(self) -> None:
        """Ping Ollama before the first embedding call — retries with back-off."""
        health_url = f"{self._ollama_base}/api/tags"
        last_exc: Exception | None = None

        for attempt in range(1, OLLAMA_HEALTH_RETRIES + 1):
            try:
                resp = requests.get(health_url, timeout=10)
                resp.raise_for_status()
                logger.info(
                    "Ollama health-check passed (attempt %d/%d).",
                    attempt, OLLAMA_HEALTH_RETRIES,
                )
                return
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Ollama health-check failed (attempt %d/%d): %s",
                    attempt, OLLAMA_HEALTH_RETRIES, exc,
                )
                if attempt < OLLAMA_HEALTH_RETRIES:
                    time.sleep(OLLAMA_HEALTH_RETRY_DELAY * attempt)

        raise OllamaUnreachableError(
            f"Ollama at '{self._ollama_base}' unreachable after "
            f"{OLLAMA_HEALTH_RETRIES} attempts. Run:  ollama serve\n"
            f"Last error: {last_exc}"
        )

    def _init_lance(self) -> tuple:
        """Open (or create) the LanceDB database and table."""
        logger.info("Opening LanceDB at '%s' …", self._persist_dir)
        db = lancedb.connect(self._persist_dir)

        schema = pa.schema([
            pa.field("doc_id",    pa.utf8()),
            pa.field("text",      pa.utf8()),
            pa.field("vector",    pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("source",    pa.utf8()),
            pa.field("tags",      pa.utf8()),
            pa.field("timestamp", pa.utf8()),
        ])

        if self._table_name in db.table_names():
            table = db.open_table(self._table_name)
            logger.info("Opened existing table '%s'.", self._table_name)
        else:
            table = db.create_table(self._table_name, schema=schema)
            logger.info("Created new table '%s'.", self._table_name)

        return db, table

    # --------------------------------------------------- embedding
    def _embed(self, text: str) -> list[float]:
        """Call Ollama /api/embeddings and return the float vector."""
        try:
            resp = requests.post(
                self._embed_url,
                json={"model": self._embed_model, "prompt": text},
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            vector: list[float] = resp.json().get("embedding", [])
            if not vector:
                raise ValueError(
                    f"Ollama returned empty embedding for model '{self._embed_model}'."
                )
            return vector
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnreachableError(
                f"Cannot connect to Ollama at {self._embed_url}. "
                "Is the server running?  →  ollama serve"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaUnreachableError(
                f"Ollama request timed out after {OLLAMA_TIMEOUT_SECONDS}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise OllamaUnreachableError(
                f"Ollama HTTP error: {exc.response.status_code} — {exc.response.text}"
            ) from exc

    # --------------------------------------------------- utilities
    @staticmethod
    def _text_to_id(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return str(uuid.UUID(digest[:32]))

    @staticmethod
    def _coerce_tags(tags: Any) -> str:
        if isinstance(tags, list):
            return ", ".join(str(t) for t in tags)
        return str(tags)

    @staticmethod
    def _error_response(
        error_type: str, action: str, message: str
    ) -> dict[str, str]:
        return {
            "status":     "error",
            "action":     action,
            "error_type": error_type,
            "message":    message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone smoke-test   →   python memory_worker.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print("\n" + "═" * 70)
    print("  ADEN HIVE — ContextMemoryWorker  smoke-test  (LanceDB backend)")
    print("═" * 70 + "\n")

    worker = ContextMemoryWorker()

    # ── 1. Store several context chunks ─────────────────────────────────────
    samples = [
        {
            "action": "store",
            "text": (
                "Project Alpha is a confidential initiative to build a "
                "next-generation recommendation engine using transformer models."
            ),
            "metadata": {"source": "kickoff_meeting", "tags": ["strategy", "ml"]},
        },
        {
            "action": "store",
            "text": (
                "The user prefers concise, bullet-point responses and dislikes "
                "overly verbose prose or filler phrases."
            ),
            "metadata": {"source": "user_prefs_v1", "tags": ["preference", "communication"]},
        },
        {
            "action": "store",
            "text": (
                "Budget for Q3 is capped at $250,000. Any expenditure above "
                "$10,000 requires VP approval via the finance portal."
            ),
            "metadata": {"source": "finance_policy_2025", "tags": ["budget", "policy"]},
        },
    ]

    print("── Storing 3 context chunks ────────────────────────────────────\n")
    for s in samples:
        print(json.dumps(worker.process(s), indent=2))
        print()

    print("── Querying memory ─────────────────────────────────────────────\n")
    print(json.dumps(worker.process({
        "action": "query",
        "text":   "What are the user's communication preferences?",
        "top_k":  3,
    }), indent=2))

    print("\n── Invalid action demo ──────────────────────────────────────────\n")
    print(json.dumps(worker.process({"action": "delete", "text": "x"}), indent=2))