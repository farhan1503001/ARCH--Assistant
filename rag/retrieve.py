"""
ARCH Assistant — retrieval for RAG.

Hybrid search over two on-disk indexes built by ingest.py:
  * ChromaDB   — dense vectors (embeddinggemma), cosine similarity
  * SQLite FTS5 — keyword search with BM25 ranking
Results are fused with reciprocal rank fusion, deduplicated by parent chunk,
and thresholded. Nothing is held in RAM between queries, so corpus size does
not affect proxy memory. Imported by guardrail_proxy.py.
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import logging
import re
import sqlite3
from pathlib import Path

import httpx

log = logging.getLogger("rag")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("ARCH_EMBED_MODEL", "embeddinggemma")
# Query embedding is one short text; on CPU it costs well under a second and keeps
# ~600 MB of VRAM free for the main model. Set ARCH_EMBED_ON_CPU=0 to use the GPU.
EMBED_ON_CPU = os.environ.get("ARCH_EMBED_ON_CPU", "1") == "1"

TOP_K_DENSE = 20
TOP_K_BM25 = 20
MAX_RESULTS = 5
# Cosine similarity floor for dense hits. embeddinggemma gives ~0.5-0.75 for relevant
# passages and ~0.2-0.4 for unrelated ones. Override with ARCH_RAG_MIN_SIM.
MIN_DENSE_SIM = float(os.environ.get("ARCH_RAG_MIN_SIM", "0.45"))
# RRF floor. A top-5 hit from a single retriever scores 1/(60+4) = 0.0156, so 0.015
# lets one strong signal through while rejecting rank-10+ noise found by one side only.
MIN_FUSED_SCORE = float(os.environ.get("ARCH_RAG_MIN_FUSED", "0.015"))
# A chunk found ONLY by keyword search must still be at least this semantically similar,
# otherwise "hello" matches every "Hello World" example. Override with ARCH_RAG_MIN_SIM_KW.
MIN_SIM_KEYWORD_ONLY = float(os.environ.get("ARCH_RAG_MIN_SIM_KW", "0.35"))
# Total retrieved text sent to the model, in characters (~4 chars/token). Keeps the prompt
# inside the model's context window. 10000 chars ~= 2500-2900 tokens.
MAX_CONTEXT_CHARS = int(os.environ.get("ARCH_RAG_MAX_CHARS", "10000"))
DEBUG = os.environ.get("ARCH_RAG_DEBUG", "0") == "1"
RRF_K = 60

# Cross-encoder reranker. Reads query + passage together and returns a relevance
# probability in [0, 1]. When it loads, it replaces the cosine/RRF thresholds as the
# relevance judge; when it is missing, retrieval falls back to the threshold logic above.
# Model files: run  python rag/download_reranker.py  once (needs internet), which saves
# them to models/reranker so the proxy works offline afterwards.
RERANKER_DIR = Path(os.environ.get("ARCH_RERANKER_DIR",
                                   str(Path(__file__).resolve().parent.parent / "models" / "reranker")))
RERANKER_NAME = os.environ.get("ARCH_RERANKER_NAME", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = 24          # how many fused candidates the reranker scores per query
# Minimum reranker probability to keep a passage. STARTING VALUE, NOT CALIBRATED: run the
# proxy with ARCH_RAG_DEBUG=1, look at the scores real questions produce, then adjust via
# ARCH_RAG_MIN_RERANK. Relevant passages typically score well above irrelevant ones, but
# the exact gap depends on the corpus.
MIN_RERANK = float(os.environ.get("ARCH_RAG_MIN_RERANK", "0.30"))
# Where the reranker runs. "auto" = GPU if PyTorch can see one, else CPU. On CPU the
# base model takes ~10 s for 24 candidates; on the GPU well under a second.
RERANKER_DEVICE = os.environ.get("ARCH_RERANKER_DEVICE", "auto")

# Messages that never need document retrieval: greetings, thanks, acknowledgements, chit-chat.
_SKIP = re.compile(r"^\W*(hi|hello|hey|hiya|yo|good (morning|afternoon|evening|night)|thanks?( you)?|"
                   r"thank you|ty|ok(ay)?|cool|great|nice|got it|bye|goodbye|see you|how are you|"
                   r"what'?s up|sup|test(ing)?|yes|no|yeah|nope|sure|please|sorry|lol|haha)\W*$", re.I)

_STOP = set("""a an and are as at be but by for from had has have he her his i if in into is it
its me my no not of on or our she so that the their them then there these they this to was we
were what when where which who will with would you your about also can could do does how more
than very""".split())


def _tok(s: str):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 1]


class Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.col = None
        self.fts_path = index_dir / "fts.sqlite"
        self.count = 0
        self.ready = False
        self.reranker = None
        self._load()
        self._load_reranker()

    # -------------------------------------------------------------- loading
    def _load(self):
        if not (self.index_dir / "chroma.sqlite3").exists():
            log.info("rag: no index at %s — retrieval disabled", self.index_dir)
            return
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.index_dir))
            self.col = client.get_or_create_collection("arch_docs", metadata={"hnsw:space": "cosine"})
            self.count = self.col.count()
            if self.count == 0:
                log.info("rag: index is empty — retrieval disabled")
                return
            if not self.fts_path.exists():
                log.warning("rag: keyword index missing (%s); run rag/ingest.py once to build it. "
                            "Dense search only until then.", self.fts_path)
            self.ready = True
            log.info("rag: %d chunks indexed at %s", self.count, self.index_dir)
        except Exception as e:
            log.error("rag: failed to load index: %s", e)

    def _load_reranker(self):
        if not self.ready:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            log.warning("rag: sentence-transformers not installed; reranker disabled, using similarity thresholds")
            return
        device = RERANKER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        try:
            if RERANKER_DIR.is_dir():
                self.reranker = CrossEncoder(str(RERANKER_DIR), device=device, local_files_only=True)
                log.info("rag: reranker loaded from %s on %s", RERANKER_DIR, device)
            else:
                log.warning("rag: no local reranker at %s; trying to download %s (run rag/download_reranker.py "
                            "to make this offline)", RERANKER_DIR, RERANKER_NAME)
                self.reranker = CrossEncoder(RERANKER_NAME, device=device)
                log.info("rag: reranker %s loaded on %s", RERANKER_NAME, device)
            if device == "cpu":
                log.warning("rag: reranker is on CPU (~10 s per query). Install the CUDA build of PyTorch "
                            "to run it on the GPU.")
        except Exception as e:
            log.warning("rag: reranker unavailable (%s); using similarity thresholds", str(e)[:200])
            self.reranker = None

    def _rerank_scores(self, query: str, passages: list[str]) -> list[float]:
        """Relevance probability per passage, in [0, 1]. Runs on CPU."""
        out = self.reranker.predict([(query, p) for p in passages], batch_size=16,
                                    show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in out]

    # -------------------------------------------------------------- search
    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{OLLAMA_URL}/api/embed",
                             json={"model": EMBED_MODEL, "input": ["task: search result | query: " + text],
                                   "keep_alive": "30m",
                                   "options": {"num_gpu": 0} if EMBED_ON_CPU else {}})
            r.raise_for_status()
            return r.json()["embeddings"][0]

    def _fts_search(self, query: str) -> list[str]:
        """Top chunk ids by BM25 from the FTS5 index."""
        if not self.fts_path.exists():
            return []
        toks = _tok(query)
        if not toks:
            return []
        match = " OR ".join(f'"{t}"' for t in toks)
        con = sqlite3.connect(f"file:{self.fts_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT id FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
                (match, TOP_K_BM25)).fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]

    async def search(self, query: str) -> list[dict]:
        """Return up to MAX_RESULTS parent chunks: [{title, page, heading, collection, text, score}]."""
        q = query.strip()
        if not self.ready or not q:
            return []
        if _SKIP.match(q) or not _tok(q):
            if DEBUG:
                log.info("rag: skipped retrieval for trivial message %r", q[:40])
            return []

        scores: dict[str, float] = {}
        sims: dict[str, float] = {}
        dense_ids: set[str] = set()
        qv = None

        # dense
        try:
            qv = await self._embed(q)
            res = self.col.query(query_embeddings=[qv], n_results=min(TOP_K_DENSE, self.count),
                                 include=["distances"])
            for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0])):
                sims[cid] = 1.0 - dist
                if sims[cid] < MIN_DENSE_SIM:
                    continue
                dense_ids.add(cid)
                scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        except Exception as e:
            log.error("rag: dense search failed: %s", e)

        # keyword
        kw_ids = []
        try:
            kw_ids = self._fts_search(q)
            for rank, cid in enumerate(kw_ids):
                scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + rank)
        except Exception as e:
            log.error("rag: keyword search failed: %s", e)

        if not scores:
            return []

        # similarity for keyword-only candidates, so weak lexical matches can be rejected
        need = [cid for cid in kw_ids if cid not in sims]
        if need and qv is not None:
            try:
                got = self.col.get(ids=need, include=["embeddings"])
                import numpy as np
                qn = np.asarray(qv); qn = qn / (np.linalg.norm(qn) or 1.0)
                for cid, emb in zip(got["ids"], got["embeddings"]):
                    e = np.asarray(emb); e = e / (np.linalg.norm(e) or 1.0)
                    sims[cid] = float(qn @ e)
            except Exception as e:
                log.error("rag: similarity lookup failed: %s", e)

        fused = sorted(scores.items(), key=lambda kv: -kv[1])
        final_score: dict[str, float] = {}

        if self.reranker is not None:
            # Reranker path: take the top fused candidates regardless of cosine floors
            # (the reranker is the relevance judge), score them, keep those above MIN_RERANK.
            cand = [cid for cid, _ in fused[:RERANK_CANDIDATES]]
            got = self.col.get(ids=cand, include=["documents", "metadatas"])
            by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
            cand = [c for c in cand if c in by_id]
            try:
                import asyncio
                rs = await asyncio.to_thread(self._rerank_scores, q, [by_id[c][0] for c in cand])
            except Exception as e:
                log.error("rag: rerank failed (%s); falling back to similarity thresholds", e)
                rs = None
            if rs is not None:
                for cid, r in zip(cand, rs):
                    if r >= MIN_RERANK:
                        final_score[cid] = r
                ranked = sorted(final_score, key=lambda c: -final_score[c])
                if DEBUG:
                    log.info("rag: %r rerank -> %s", q[:50], "; ".join(
                        f"{by_id[c][1]['title'][:24]} p{by_id[c][1].get('page', 0)} r={r:.2f} "
                        f"sim={sims.get(c, float('nan')):.2f} {'keep' if c in final_score else 'drop'}"
                        for c, r in sorted(zip(cand, rs), key=lambda t: -t[1])[:8]))
                if not ranked:
                    return []
                return self._assemble(ranked, by_id, final_score, sims)

        # Threshold path (no reranker): cosine floor for dense hits, similarity floor for keyword-only hits.
        ranked = []
        for cid, sc in fused:
            if sc < MIN_FUSED_SCORE:
                continue
            if cid not in dense_ids and sims.get(cid, 1.0) < MIN_SIM_KEYWORD_ONLY:
                continue          # keyword-only hit that is semantically unrelated
            ranked.append(cid)
            final_score[cid] = sc

        if DEBUG:
            log.info("rag: %r -> %s", q[:50], "; ".join(
                f"{cid.split(':')[0][:28]} sim={sims.get(cid, float('nan')):.2f} "
                f"{'D' if cid in dense_ids else ''}{'K' if cid in kw_ids else ''} "
                f"{'keep' if cid in ranked else 'drop'}" for cid, _ in fused[:8]))
        if not ranked:
            return []

        got = self.col.get(ids=ranked[:TOP_K_DENSE + TOP_K_BM25], include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
        return self._assemble(ranked, by_id, final_score, sims)

    def _assemble(self, ranked, by_id, final_score, sims):
        """Dedupe by parent, cap total context size, build result dicts."""

        out, seen_parents, used = [], set(), 0
        for cid in ranked:
            if cid not in by_id:
                continue
            doc, m = by_id[cid]
            text = m.get("parent") or doc
            key = (m["doc_id"], text[:200])
            if key in seen_parents:
                continue
            if used + len(text) > MAX_CONTEXT_CHARS:
                if not out:                       # always allow at least one, truncated
                    text = text[:MAX_CONTEXT_CHARS]
                else:
                    break
            seen_parents.add(key); used += len(text)
            out.append({"title": m["title"], "page": m.get("page", 0), "heading": m.get("heading", ""),
                        "collection": m.get("collection", ""), "text": text,
                        "score": round(final_score.get(cid, 0.0), 4), "sim": round(sims.get(cid, 0.0), 3)})
            if len(out) >= MAX_RESULTS:
                break
        return out


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks as a context block for the model."""
    lines = ["Reference material from ARCH documents. Use it when it is relevant to the question, "
             "cite sources by their [n] tag, and say so if it does not cover what was asked.\n"]
    for i, h in enumerate(hits, 1):
        where = f", p.{h['page']}" if h["page"] else ""
        sec = f" — {h['heading']}" if h["heading"] else ""
        lines.append(f"[{i}] {h['title']}{where}{sec}\n{h['text']}\n")
    return "\n".join(lines)
