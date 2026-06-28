import os
import json
import asyncio
import hashlib
from typing import Any, Optional, Sequence
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core import Settings
from src.ast_extractor import KG_NODES_KEY

_CACHE_FILE = "data/semantic_cache.json"
_global_cache = None


def _load_cache() -> dict:
    global _global_cache
    if _global_cache is None:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                try:
                    _global_cache = json.load(f)
                except Exception:
                    _global_cache = {}
        else:
            _global_cache = {}
            os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    return _global_cache


def _save_cache():
    """Atomically flush the in-memory cache to disk via a tmp → rename."""
    if _global_cache is not None:
        tmp_path = _CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_global_cache, f)
        os.replace(tmp_path, _CACHE_FILE)  # atomic on POSIX; near-atomic on Windows


def _make_cache_key(name: str, code: str) -> str:
    """
    Deterministic, process-restart-safe cache key using SHA-256.
    Line endings are NORMALISED (CRLF → LF) so keys are stable
    regardless of whether git cloned with core.autocrlf=true (Windows).
    Python's built-in hash() is randomised per-process (PYTHONHASHSEED).
    """
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{name}_{digest}"


class SemanticEnrichmentComponent(TransformComponent):
    """
    Takes the deterministically extracted EntityNodes (Functions), reads their raw code,
    and asks the LLM to generate a 1-sentence summary enriching the Graph node properties.

    Features:
    - Pre-flight Ollama check: skips ALL LLM calls if Ollama is unreachable (cache-only mode).
    - Async semaphore throttling (Semaphore(4)) to avoid saturating the local GPU.
    - Persistent SHA-256 keyed cache (CRLF-normalised) so restarts skip already-summarised functions.
    - Exponential backoff (3 attempts) for transient LLM/network failures.
    - Atomic cache writes after every successful LLM call (no progress lost on crash).
    - Ollama VRAM eviction + CUDA embedding upgrade after LLM phase completes.
    """

    llm: Any
    # If set, the Ollama model will be evicted from VRAM after enrichment so the
    # embedding model can claim the full GPU budget for the embedding phase.
    ollama_model: Optional[str] = None

    def __call__(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        # Bridge LlamaIndex's synchronous caller into our async pipeline.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.acall(nodes, **kwargs))
        else:
            return asyncio.run(self.acall(nodes, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        _load_cache()

        # ── Pre-flight: determine how many nodes need a live LLM call ──────
        cache_misses = self._count_cache_misses(nodes)
        skip_llm = False

        if cache_misses > 0:
            ollama_ok = await self._check_ollama_available()
            if not ollama_ok:
                print(
                    f"\n⚠️  Ollama is not reachable. "
                    f"{cache_misses} function(s) are not in the semantic cache and will be skipped.\n"
                    f"   Run Ollama and re-ingest to enrich them. "
                    f"The ingestion will still complete successfully."
                )
                skip_llm = True
            else:
                print(f"\n🔎 {cache_misses} function(s) need LLM enrichment (Ollama is available).")
        else:
            print("\n✅ All functions found in semantic cache — no Ollama calls needed.")

        # ── Enrichment phase ────────────────────────────────────────────────
        semaphore = asyncio.Semaphore(4)
        jobs = [self._aenrich_node(node, semaphore, skip_llm) for node in nodes]
        await asyncio.gather(*jobs)

        # ── GPU HANDOFF ─────────────────────────────────────────────────────
        # The LLM phase is 100% complete. Evict Ollama from VRAM (if loaded),
        # then load the embedding model on CUDA for the embedding phase.
        if self.ollama_model:
            await self._evict_ollama_and_switch_to_cuda()

        return nodes

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _count_cache_misses(self, nodes: Sequence[BaseNode]) -> int:
        """Count how many FUNCTION entities are missing from the cache."""
        count = 0
        for node in nodes:
            for entity in node.metadata.get(KG_NODES_KEY, []):
                if entity.label != "FUNCTION":
                    continue
                code = entity.properties.get("code")
                if not code:
                    continue
                if "summary" in entity.properties:
                    continue
                if _make_cache_key(entity.name, code) not in _global_cache:
                    count += 1
        return count

    async def _check_ollama_available(self) -> bool:
        """Returns True if Ollama is reachable at its configured host."""
        import httpx
        ollama_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{ollama_base}/", timeout=3.0)
                return resp.status_code < 500
        except Exception:
            return False

    async def _evict_ollama_and_switch_to_cuda(self):
        """
        1. Check /api/ps — if model is already not in VRAM (auto-evicted or all-cache run), skip eviction.
        2. If loaded, evict via keep_alive=0 and verify via /api/ps.
        3. Hard stop if eviction cannot be confirmed.
        4. Load embedding model on CUDA — no CPU fallback.
        """
        import httpx

        ollama_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model_base = self.ollama_model.split(":")[0]

        async with httpx.AsyncClient() as client:

            # ── Step 1: Check if model is actually in VRAM ─────────────────
            def _vram_used(models: list) -> bool:
                return any(
                    m.get("name", "").startswith(model_base) and m.get("size_vram", 0) > 0
                    for m in models
                )

            try:
                ps = await client.get(f"{ollama_base}/api/ps", timeout=10.0)
                ps.raise_for_status()
                running = ps.json().get("models", [])
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Ollama is not running at all → it cannot be using any VRAM.
                # This is the expected case when all LLM calls were cache hits.
                print(
                    f"\nℹ️  Ollama is not reachable at {ollama_base} "
                    f"(not started, or all LLM calls were cache hits). "
                    f"No VRAM to evict."
                )
                running = []
            except Exception as e:
                raise RuntimeError(
                    f"❌ Unexpected error querying Ollama /api/ps: {e}\n"
                    f"   Check that Ollama is healthy at {ollama_base}."
                )

            if not _vram_used(running):
                print(
                    f"\nℹ️  '{self.ollama_model}' is not in VRAM "
                    f"(all-cache run or auto-evicted). Skipping eviction."
                )
            else:
                # ── Step 2: Evict ───────────────────────────────────────────
                print(f"\n🔓 Evicting '{self.ollama_model}' from VRAM...")
                evict_ok = False
                for endpoint, payload in [
                    # No prompt field — Ollama rejects empty prompt strings.
                    # keep_alive=0 is the documented unload trigger.
                    ("generate", {"model": self.ollama_model, "keep_alive": 0}),
                    ("chat",     {"model": self.ollama_model, "keep_alive": 0, "messages": []}),
                ]:
                    try:
                        resp = await client.post(
                            f"{ollama_base}/api/{endpoint}",
                            json=payload,
                            timeout=20.0,
                        )
                        if resp.status_code == 200:
                            evict_ok = True
                            break
                    except Exception:
                        continue

                if not evict_ok:
                    raise RuntimeError(
                        f"❌ Eviction API call failed for '{self.ollama_model}'.\n"
                        f"   Aborting — cannot run embedding without free VRAM.\n"
                        f"   Manually run:  ollama stop {self.ollama_model}\n"
                        f"   Then restart:  uv run python -m src.ingestion <url>"
                    )

                # ── Step 3: Verify eviction via /api/ps ─────────────────────
                await asyncio.sleep(2)
                try:
                    ps2 = await client.get(f"{ollama_base}/api/ps", timeout=10.0)
                    ps2.raise_for_status()
                    if _vram_used(ps2.json().get("models", [])):
                        raise RuntimeError(
                            f"❌ Eviction verified FAILED — '{self.ollama_model}' is still in VRAM.\n"
                            f"   Aborting — CPU embedding will crash this system.\n"
                            f"   Manually run:  ollama stop {self.ollama_model}\n"
                            f"   Then restart:  uv run python -m src.ingestion <url>"
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"❌ Could not verify eviction via /api/ps: {e}")

                print("✅ Ollama model evicted and confirmed gone from VRAM.")

        # ── Step 4: Load embedding model on CUDA ───────────────────────────
        import gc
        import torch
        old_model = Settings.embed_model
        Settings.embed_model = None
        del old_model
        gc.collect()
        torch.cuda.empty_cache()

        print("\n📦 Loading embedding model on CUDA...")
        if not torch.cuda.is_available():
            raise RuntimeError("❌ No CUDA device available. Cannot load embedding model.")

        free_vram_mb = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1024 / 1024

        print(f"📊 Free VRAM: {free_vram_mb:.0f} MB ({torch.cuda.get_device_name(0)})")

        REQUIRED_MB = 1400  # bge-m3 fp16 ~1.1 GB weights + activation headroom
        if free_vram_mb < REQUIRED_MB:
            raise RuntimeError(
                f"❌ Insufficient VRAM: {free_vram_mb:.0f} MB free, need {REQUIRED_MB} MB.\n"
                f"   Aborting to prevent system crash."
            )

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from src.config import EMBEDDING_MODEL_NAME

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True,        # Required by bge-m3 for custom pooling
            device="cuda",
            embed_batch_size=4,            # Conservative: prevents VRAM OOM on 4GB GPU
            model_kwargs={"dtype": torch.float16},  # fp16: ~1.1 GB instead of ~2.2 GB
        )
        print("✅ Embedding model on CUDA — proceeding to embedding phase.")

    async def _aenrich_node(
        self, node: BaseNode, semaphore: asyncio.Semaphore, skip_llm: bool = False
    ):
        kg_nodes = node.metadata.get(KG_NODES_KEY, [])
        for entity in kg_nodes:
            if entity.label != "FUNCTION":
                continue
            if "summary" in entity.properties:
                continue  # already enriched (idempotency)
            code = entity.properties.get("code")
            if not code:
                continue

            cache_key = _make_cache_key(entity.name, code)
            if cache_key in _global_cache:
                entity.properties["summary"] = _global_cache[cache_key]
                continue  # cache hit — no LLM call needed

            if skip_llm:
                continue  # Ollama not available — silently skip this node

            prompt = (
                f"Summarize the purpose of this function in exactly one sentence:\n\n{code}"
            )

            # Exponential backoff: 3 attempts (1s → 2s between retries)
            for attempt in range(3):
                try:
                    async with semaphore:
                        response = await self.llm.acomplete(prompt)
                    safe_summary = str(response.text).strip().split(".")[0] + "."
                    entity.properties["summary"] = safe_summary
                    _global_cache[cache_key] = safe_summary
                    _save_cache()  # persist immediately — crash-safe
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt
                        print(
                            f"⚠️  Retry {attempt + 1}/3 for '{entity.name}' "
                            f"(waiting {wait}s): {e}"
                        )
                        await asyncio.sleep(wait)
                    else:
                        print(
                            f"❌ Semantic enrichment permanently failed for "
                            f"'{entity.name}': {e}"
                        )