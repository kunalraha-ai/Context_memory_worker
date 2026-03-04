# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║             ADEN HIVE — Context Memory Worker                               ║
# ║             Docker Image                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Base: slim Python 3.12 on Debian Bookworm (no C++ toolchain needed) ───────
FROM python:3.12-slim-bookworm

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Aden Hive"
LABEL description="Context Memory Worker — LanceDB + Ollama nomic-embed-text"
LABEL version="1.0.0"

# ── System deps (curl for healthcheck only, no build tools needed) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user for security ─────────────────────────────────────────
RUN groupadd --gid 1001 hive && \
    useradd  --uid 1001 --gid hive --shell /bin/bash --create-home hive

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache:
# requirements change rarely → this layer won't rebuild unless they do.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY engine.py .

# ── Persistent memory volume ──────────────────────────────────────────────────
# The ./hive_memory directory is where LanceDB stores its data files.
# Mount a named volume here so memory survives container restarts.
RUN mkdir -p /app/hive_memory && chown -R hive:hive /app
VOLUME ["/app/hive_memory"]

# ── Drop to non-root ───────────────────────────────────────────────────────────
USER hive

# ── Environment defaults (can be overridden at runtime with -e) ───────────────
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Healthcheck — verifies Ollama is reachable from inside the container ──────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f ${OLLAMA_BASE_URL}/api/tags || exit 1

# ── Default command: run the smoke-test ───────────────────────────────────────
# In production, replace this with your Hive graph entry-point, e.g.:
#   CMD ["python", "your_hive_graph.py"]
CMD ["python", "engine.py"]