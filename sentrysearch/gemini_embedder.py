"""Gemini embedding backend using the google-genai SDK.

Embeds video chunks inline via Part.from_bytes — no Files API needed.
"""

import os
import sys
import time
from collections import deque

from dotenv import load_dotenv

from .base_embedder import BaseEmbedder

load_dotenv()

EMBED_MODEL = os.environ.get("SENTRYSEARCH_GEMINI_MODEL", "gemini-embedding-2-preview")
DIMENSIONS = 768
DEFAULT_RPM = 55

# Vertex AI defaults. gemini-embedding-2-preview is served at us-central1
# (verified 2026-06-02; the `global` location 404s for this model). The SA
# key path mirrors the Node tools' convention (singkeo/shared/node/vertex-auth.js).
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_VERTEX_SA_KEY_PATH = "~/.config/anika/vertex-sa-key.json"

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple sliding-window rate limiter based on request timestamps."""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GeminiAPIKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is missing."""


def _embed_provider() -> str:
    """Resolve the embedding provider.

    Honors SENTRYSEARCH_EMBED_PROVIDER first, then GEMINI_PROVIDER (the same
    flag the Node tools use), defaulting to AI Studio. Returns 'vertex' or
    'ai_studio'.
    """
    raw = (
        os.environ.get("SENTRYSEARCH_EMBED_PROVIDER")
        or os.environ.get("GEMINI_PROVIDER")
        or "ai_studio"
    )
    return raw.strip().lower()


def _make_vertex_client(genai):
    """Build a google-genai client in Vertex AI mode.

    Auth uses a GCP service-account JSON (the same one the Node tools use):
    VERTEX_SA_KEY_PATH (default ~/.config/anika/vertex-sa-key.json). The SDK
    picks it up via GOOGLE_APPLICATION_CREDENTIALS, which we set from that
    path if it isn't already set. Project comes from VERTEX_PROJECT_ID or the
    SA JSON's project_id; region from VERTEX_LOCATION (default us-central1).
    """
    import json

    sa_path = os.path.expanduser(
        os.environ.get("VERTEX_SA_KEY_PATH", DEFAULT_VERTEX_SA_KEY_PATH)
    )
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and os.path.exists(sa_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    project = os.environ.get("VERTEX_PROJECT_ID")
    if not project:
        try:
            with open(sa_path) as f:
                project = json.load(f).get("project_id")
        except OSError as exc:
            raise GeminiAPIKeyError(
                "Vertex embedding provider selected but the service-account "
                f"key is not readable at {sa_path}: {exc}\n\n"
                "Set VERTEX_SA_KEY_PATH (or VERTEX_PROJECT_ID), or switch back "
                "to AI Studio with GEMINI_PROVIDER=ai_studio + GEMINI_API_KEY."
            ) from exc
    if not project:
        raise GeminiAPIKeyError(
            "Vertex embedding provider selected but no project id found "
            "(set VERTEX_PROJECT_ID or use a service-account JSON that "
            "includes project_id)."
        )

    location = os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
    return genai.Client(vertexai=True, project=project, location=location)


class GeminiQuotaError(RuntimeError):
    """Raised when API quota is exceeded."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry(fn, *, max_retries: int = 5, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Call *fn* with exponential back-off on transient errors (429, 503)."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            retryable = status in (429, 503)
            if not retryable:
                retryable = "resource exhausted" in msg or "503" in msg or "429" in msg
            if not retryable or attempt == max_retries:
                if "resource exhausted" in msg or status == 429:
                    raise GeminiQuotaError(
                        "Gemini API rate limit exceeded.\n\n"
                        "The free tier allows 60 requests/minute. Options:\n"
                        "  - Wait a minute and retry\n"
                        "  - Use a smaller --chunk-duration to create fewer chunks\n"
                        "  - Upgrade your API plan at https://aistudio.google.com"
                    ) from exc
                raise
            wait = min(delay, max_delay)
            print(
                f"  Retryable error (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay *= 2


# ---------------------------------------------------------------------------
# GeminiEmbedder
# ---------------------------------------------------------------------------

class GeminiEmbedder(BaseEmbedder):
    """Gemini Embedding 2 backend (API-based)."""

    def __init__(self):
        from google import genai
        from google.genai import types  # noqa: F811

        if _embed_provider() == "vertex":
            # Vertex AI: same gemini-embedding-2-preview model (so vectors stay
            # compatible with an AI-Studio-built index), auth via service
            # account. Bills the GCP project credit instead of an API key.
            self._client = _make_vertex_client(genai)
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise GeminiAPIKeyError(
                    "GEMINI_API_KEY is not set.\n\n"
                    "Run: sentrysearch init\n\n"
                    "Or set it manually:\n"
                    "  export GEMINI_API_KEY=your-key\n\n"
                    "Or use Vertex AI instead (service-account auth):\n"
                    "  export GEMINI_PROVIDER=vertex\n\n"
                    "Or use a local model instead (no API key needed):\n"
                    "  sentrysearch index <directory> --backend local"
                )
            self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter()

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        from google.genai import types

        video_part = self._make_video_part(chunk_path, types)

        if verbose:
            size_kb = os.path.getsize(chunk_path) / 1024
            print(
                f"    [verbose] sending {size_kb:.0f}KB to {EMBED_MODEL}",
                file=sys.stderr,
            )

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=EMBED_MODEL,
                contents=types.Content(parts=[video_part]),
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIMENSIONS,
                ),
            )
        )
        elapsed = time.monotonic() - t0
        embedding = response.embeddings[0].values

        if verbose:
            size_kb = os.path.getsize(chunk_path) / 1024
            print(
                f"    [verbose] dims={len(embedding)}, "
                f"chunk_size={size_kb:.0f}KB, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        from google.genai import types

        self._limiter.wait()
        t0 = time.monotonic()
        response = _retry(
            lambda: self._client.models.embed_content(
                model=EMBED_MODEL,
                contents=query_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=DIMENSIONS,
                ),
            )
        )
        elapsed = time.monotonic() - t0
        embedding = response.embeddings[0].values

        if verbose:
            print(
                f"  [verbose] query embedding: dims={len(embedding)}, "
                f"api_time={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def dimensions(self) -> int:
        return DIMENSIONS

    @staticmethod
    def _make_video_part(chunk_path: str, types):
        with open(chunk_path, "rb") as f:
            video_bytes = f.read()
        if hasattr(types.Part, "from_bytes"):
            return types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        return types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))
