# CLAUDE.md

## Project

SentrySearch — semantic search over dashcam/video footage. Splits videos into chunks, embeds them (Gemini API or local Qwen3-VL model), stores vectors in ChromaDB, and retrieves clips via natural language queries.

## Role inside MARKETING-TOOLS

This package is **also a library dependency of CLIPS-FINDER's web-demo** (`../web-demo/server.py`, `../web-demo/indexing.py`). The web-demo imports `sentrysearch.embedder.embed_query`, `sentrysearch.store.SentryStore`, and `sentrysearch.chunker.{chunk_video, scan_directory, is_still_frame_chunk, preprocess_chunk}` and runs them inside the Flask process — it does not shell out to the `sentrysearch` CLI. Changes to those public surfaces ripple into the web app on next deploy.

The web-demo's PEP 723 inline header pulls this package from a git remote (`sentrysearch @ git+ssh://git@github.com/ldvclt-code/sentrysearch.git`), so commits to the `ldvclt-code/sentrysearch` remote are what the production Mac mini consumes, not this local checkout.

The standalone CLI surface (`init`, `index`, `search`, `trim`, `stats`, `reset`, `remove`) still works for dev workflows and is documented in `README.md`.

## Commands

```bash
# Dev install
uv sync                          # core deps
uv sync --group test             # + test deps

# User install (provides `sentrysearch` CLI)
uv tool install .                       # core (Gemini backend)
uv tool install ".[local]"              # + local model deps
uv tool install ".[local-quantized]"    # + local model deps (4-bit)
uv tool install ".[tesla]"              # + Tesla overlay deps

# Run tests
uv run pytest
uv run pytest --cov --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_store.py -v

# CLI
sentrysearch init                          # set up Gemini API key
sentrysearch index /path/to/footage        # index videos
sentrysearch search "red car"              # search indexed footage
sentrysearch stats                         # show index info
```

## Architecture

- **Embedder factory pattern**: `base_embedder.py` (ABC) -> `gemini_embedder.py` + `local_embedder.py`. The factory in `embedder.py` caches a global singleton via `get_embedder(backend)` / `reset_embedder()`.
- **Gemini backend** (`gemini_embedder.py`): model `gemini-embedding-2-preview` (overridable via `SENTRYSEARCH_GEMINI_MODEL` env var), 768 dims. Sliding-window rate limiter at **55 req/min** with exponential backoff on 429/503. `embed_video_chunk` uses `task_type="RETRIEVAL_DOCUMENT"`; `embed_query` uses `RETRIEVAL_QUERY` so query and chunk vectors live in the same task-typed space.
- **Local backend** (`local_embedder.py`): Qwen3-VL-Embedding (`qwen8b` = `Qwen/Qwen3-VL-Embedding-8B`, `qwen2b` = `Qwen/Qwen3-VL-Embedding-2B`). Auto-detects hardware: NVIDIA GPU → 8B; Apple Silicon ≥24 GB RAM → 8B; smaller Macs / CPU → 2B. Optional 4-bit quantization on NVIDIA via bitsandbytes. MRL-truncates to 768 dims and L2-normalizes so the local and Gemini vector spaces are dimensionally interchangeable, but they are stored in **separate ChromaDB collections** because the actual vectors are not interoperable.
- **Store**: `store.py` wraps ChromaDB. Separate collections per backend (`dashcam_chunks` for gemini, `dashcam_chunks_local` for local) with `metadata = {"hnsw:space": "cosine", "embedding_backend": ...}` so a query against the wrong backend can be detected and surfaced. Cosine **distance** is converted to similarity via `score = 1.0 - distance` at search time.
- **Video ingestion**: `chunker.py` defines `SUPPORTED_VIDEO_EXTENSIONS` (`.mp4`, `.mov`) and `is_supported_video_file()` for directory scanning. All formats ffmpeg can decode are processable.
- **Pipeline**: `chunker.py` (split video) -> `embedder.py` (embed chunks) -> `store.py` (persist) -> `search.py` (query) -> `trimmer.py` (extract clip).
- **Tesla overlay**: `metadata.py` parses SEI NAL units from Tesla firmware, `overlay.py` renders HUD via ASS subtitles.

## CLI surface (Click, in `cli.py`)

| Command | Purpose |
|---|---|
| `sentrysearch init` | Prompt for Gemini API key, write to `~/.sentrysearch/.env`, validate with a test embedding. |
| `sentrysearch index <dir>` | Walk a directory, chunk videos (default 30s + 5s overlap), preprocess (480p/5fps), embed, persist. `--backend local --model {qwen8b,qwen2b}` switches to local inference. |
| `sentrysearch search <query>` | Embed query, top-N retrieve, optionally trim the best match. `--threshold` gates auto-trim on confidence. Backend/model auto-detected from the indexed collection. |
| `sentrysearch trim <video>` | Three-stage ffmpeg trim with fallbacks (`trimmer.py`). |
| `sentrysearch stats` | Print collection counters. |
| `sentrysearch reset` | Wipe the index for a backend. |
| `sentrysearch remove <files...>` | Drop specific files by basename match. |

## Testing patterns

- All Gemini API calls are mocked — tests never hit external services.
- ChromaDB uses real in-memory instances via `tmp_store` fixture (no mocking).
- Synthetic test videos generated via ffmpeg (`tiny_video`, `longer_video` fixtures).
- `conftest.py` has autouse fixtures that reset the embedder singleton and ffmpeg cache between tests.
- Patch targets for gemini: `google.genai.Client` (lazy import), `sentrysearch.gemini_embedder.time.*`.
- `dashcam_pb2.py` is excluded from coverage (protobuf generated).

## Build

- Package manager: **uv**
- Build backend: **hatchling**
- Python: **>=3.11**
- CI: GitHub Actions, matrix of (ubuntu, macos, windows) x (3.11, 3.12)

## Key files

- `sentrysearch/cli.py` — Click CLI entry point, all user-facing commands
- `sentrysearch/embedder.py` — Factory + convenience wrappers (embed_query, embed_video_chunk)
- `sentrysearch/gemini_embedder.py` — Gemini API backend with rate limiter and retry logic
- `sentrysearch/local_embedder.py` — Qwen3-VL local inference backend
- `sentrysearch/store.py` — ChromaDB wrapper, backend detection, chunk ID generation
- `sentrysearch/chunker.py` — Video splitting, preprocessing, still-frame detection, directory scanning (.mp4/.mov)
- `sentrysearch/trimmer.py` — Three-stage ffmpeg clip extraction with fallbacks
- `tests/conftest.py` — Shared fixtures (mock embedders, tmp store, synthetic videos)
