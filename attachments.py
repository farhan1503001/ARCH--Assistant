"""
Files a student attaches to one chat (PDF, DOCX, TXT, MD).

On upload the file is parsed with the same parsers and chunker as the document library
(rag/ingest.py), embedded, and stored encrypted next to the chat:

    <data dir>/uploads/<chat id>/<file id>.meta.arch   small: name, pages, size
    <data dir>/uploads/<chat id>/<file id>.data.arch   chunks, parent passages, vectors

Nothing is added to the shared library index; a file belongs to its chat only and is
deleted with it.

On every turn of that chat, context_for() picks what the model sees:
  * all attached files together fit in the budget  -> their full text
  * a "summarize this paper" style request          -> excerpts spread evenly across each file
  * a trivial message ("thanks", "ok")              -> nothing (the file names are still listed)
  * anything else                                   -> the opening of each file (title, authors,
                                                       abstract) plus the passages that best match
                                                       the question, ranked by the reranker when
                                                       it is loaded, else by cosine similarity
"""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

import ingest                                   # rag/ingest.py (on sys.path via the proxy)
from retrieve import _SKIP, _tok                # trivial-message filter and tokenizer

log = logging.getLogger("uploads")

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_MB = int(os.environ.get("ARCH_UPLOAD_MAX_MB", "50"))
MAX_FILES_PER_CHAT = int(os.environ.get("ARCH_UPLOAD_MAX_FILES", "5"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("ARCH_UPLOAD_MAX_CHUNKS", "4000"))   # roughly 600-800 pages

# Characters of attached-file text sent to the model per turn. Shares the context window with
# the system prompt, library passages and conversation; see guardrail_proxy.py.
ATTACH_MAX_CHARS = int(os.environ.get("ARCH_UPLOAD_MAX_CHARS", "7000"))
ATTACH_MAX_PASSAGES = 6         # including the opening of each file
OVERVIEW_CHARS = 800             # opening of each file, always included for normal questions
SPREAD_PIECE_CHARS = 1200        # size of each excerpt when summarizing a long file
CANDIDATES_DENSE = 24
CANDIDATES_KEYWORD = 8
# Relevance floors for passages from attached files. Deliberately lower than the library's,
# because a student who attached a file is usually asking about it. STARTING VALUES, NOT
# CALIBRATED: check the scores with ARCH_RAG_DEBUG=1 and adjust.
ATTACH_MIN_RERANK = float(os.environ.get("ARCH_UPLOAD_MIN_RERANK", "0.05"))
ATTACH_MIN_SIM = float(os.environ.get("ARCH_UPLOAD_MIN_SIM", "0.30"))
DEBUG = os.environ.get("ARCH_RAG_DEBUG", "0") == "1"

QUERY_PREFIX = "task: search result | query: "    # same prefixes as the library index
_SUMMARY_RE = re.compile(
    r"\b(summar\w*|overview|outline|tl;?dr|gist|"
    r"main (idea|point|argument|finding|contribution|claim)s?|"
    r"key (idea|point|finding|takeaway|contribution)s?|"
    r"what('?s| is| are) (this|the|these) (\w+ )?(paper|file|document|doc|pdf|article|chapter|book|report|reading)s? about)\b",
    re.I)
_DOC_REF_RE = re.compile(
    r"\b(this|the|my|attached|uploaded|these)\s+(\w+\s+)?(paper|file|document|doc|pdf|article|chapter|book|"
    r"report|attachment|upload|text|reading|notes)s?\b|\b(it|this)\b", re.I)


class UploadError(Exception):
    """A problem the student should see, with an HTTP status for the API."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class AttachmentStore:
    def __init__(self, root: Path, protect, unprotect, embed, reranker=lambda: None):
        """embed: async fn(list[str]) -> list[list[float]] (prefixes already applied).
        reranker: fn returning a sentence-transformers CrossEncoder or None."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._protect, self._unprotect = protect, unprotect
        self._embed = embed
        self._reranker = reranker
        self._cache: OrderedDict = OrderedDict()     # (chat_id, file_id) -> loaded file
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------ files on disk
    def _dir(self, chat_id: str) -> Path:
        return self.root / chat_id

    def _write(self, p: Path, obj) -> None:
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(self._protect(json.dumps(obj, ensure_ascii=False).encode("utf-8")))
        tmp.replace(p)

    def _read(self, p: Path):
        return json.loads(self._unprotect(p.read_bytes()).decode("utf-8"))

    def list_files(self, chat_id: str) -> list[dict]:
        d = self._dir(chat_id)
        if not d.is_dir():
            return []
        out = []
        for f in d.glob("*.meta.arch"):
            try:
                out.append(self._read(f))
            except Exception as e:
                log.warning("skipping unreadable attachment %s: %s", f.name, e)
        out.sort(key=lambda m: m.get("created", 0))
        return out

    def delete(self, chat_id: str, file_id: str) -> bool:
        d = self._dir(chat_id)
        found = False
        for suffix in (".meta.arch", ".data.arch"):
            p = d / f"{file_id}{suffix}"
            if p.exists():
                p.unlink()
                found = True
        self._cache.pop((chat_id, file_id), None)
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
        return found

    def delete_chat(self, chat_id: str) -> None:
        for key in [k for k in self._cache if k[0] == chat_id]:
            self._cache.pop(key, None)
        shutil.rmtree(self._dir(chat_id), ignore_errors=True)

    # ------------------------------------------------------------ upload
    async def add(self, chat_id: str, filename: str, data: bytes) -> dict:
        name = Path(filename or "file").name.strip() or "file"
        ext = Path(name).suffix.lower()
        if ext == ".doc":
            raise UploadError("Old .doc files aren't supported. Save it as .docx or PDF and try again.")
        if ext not in SUPPORTED:
            raise UploadError("Only PDF, DOCX, TXT and MD files can be attached.")
        if not data:
            raise UploadError("The file is empty.")
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise UploadError(f"The file is larger than {MAX_UPLOAD_MB} MB.", 413)

        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:                       # one upload at a time per chat
            if len(self.list_files(chat_id)) >= MAX_FILES_PER_CHAT:
                raise UploadError(f"A chat can hold up to {MAX_FILES_PER_CHAT} files. "
                                  "Remove one or start a new chat.", 409)
            t0 = time.time()
            title, pages = await asyncio.to_thread(_parse, name, ext, data)
            children = ingest.build_chunks("upload", title, "upload", pages)
            if not children:
                raise UploadError("No readable text was found in this file. If it is a scanned PDF, "
                                  "run OCR on it first (for example with Adobe Acrobat).", 422)
            if len(children) > MAX_CHUNKS_PER_FILE:
                raise UploadError("This file is too long to attach. Split it into parts (for example "
                                  "one chapter per file) or add it to the document library instead.", 413)
            parents, parent_idx = [], {}
            for c in children:
                if c["parent"] not in parent_idx:
                    parent_idx[c["parent"]] = len(parents)
                    parents.append(c["parent"])
            t_parse = time.time() - t0
            vecs = await self._embed([ingest.doc_prefix(c["meta"]) + c["text"] for c in children])
            emb = np.asarray(vecs, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != len(children):
                raise UploadError("Embedding the file failed. Check that Ollama is running.", 502)

            file_id = uuid.uuid4().hex[:12]
            full_text = "\n\n".join(ingest.clean(t) for _, t in pages if t and t.strip())
            meta = {"id": file_id, "name": name, "title": title, "size": len(data),
                    "pages": len(pages) if ext == ".pdf" else None,
                    "chars": len(full_text), "chunks": len(children),
                    "created": int(time.time() * 1000)}
            payload = {
                "parents": parents,
                "children": [{"t": c["text"], "p": parent_idx[c["parent"]], "pg": c["meta"]["page"],
                              "h": c["meta"]["heading"]} for c in children],
                "full": full_text,
                "dim": int(emb.shape[1]),
                "emb": base64.b64encode(emb.tobytes()).decode("ascii"),
            }
            d = self._dir(chat_id)
            d.mkdir(parents=True, exist_ok=True)
            self._write(d / f"{file_id}.data.arch", payload)
            self._write(d / f"{file_id}.meta.arch", meta)   # written last: a file is listed only when complete
            log.info("upload %s: %s, %d chunks, %d chars, parse %.1fs, total %.1fs", chat_id, name,
                     len(children), len(full_text), t_parse, time.time() - t0)
            return meta

    # ------------------------------------------------------------ retrieval
    def _load(self, chat_id: str, meta: dict) -> dict:
        key = (chat_id, meta["id"])
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        raw = self._read(self._dir(chat_id) / f"{meta['id']}.data.arch")
        emb = np.frombuffer(base64.b64decode(raw["emb"]), dtype=np.float32).reshape(-1, raw["dim"])
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        loaded = {"meta": meta, "parents": raw["parents"], "children": raw["children"],
                  "full": raw.get("full", ""), "emb": emb / norms,
                  "toks": [set(_tok(c["t"])) for c in raw["children"]]}
        self._cache[key] = loaded
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return loaded

    async def context_for(self, chat_id: str, message: str, query: str,
                          budget: int = ATTACH_MAX_CHARS) -> tuple[list[dict], list[dict]]:
        """Passages from this chat's files for the current turn.

        message: the student's latest message, used to recognise summary requests.
        query:   what to search for (the proxy expands short follow-ups).
        Returns (files, hits); hits are dicts like the library's, with kind="file"."""
        files = self.list_files(chat_id)
        if not files:
            return [], []
        docs = [await asyncio.to_thread(self._load, chat_id, m) for m in files]
        msg = (message or "").strip()
        q = (query or msg).strip()

        if sum(len(d["full"]) for d in docs) <= budget:
            hits = [_hit(d, d["full"], 0, "full text") for d in docs if d["full"].strip()]
        elif not q or _SKIP.match(msg) or not _tok(q):
            hits = []
        elif _SUMMARY_RE.search(msg) and (_DOC_REF_RE.search(msg) or len(msg.split()) <= 4):
            hits = _spread(docs, budget)
        else:
            hits = await self._search(docs, q, budget)
        for h in hits:
            h.pop("_doc", None)
        return files, hits

    async def _search(self, docs: list[dict], q: str, budget: int) -> list[dict]:
        qv = np.asarray((await self._embed([QUERY_PREFIX + q]))[0], dtype=np.float32)
        qv /= (np.linalg.norm(qv) or 1.0)
        qtoks = set(_tok(q))

        cands = []                                      # (doc index, child index, cosine)
        for di, d in enumerate(docs):
            sims = d["emb"] @ qv
            picked = list(np.argsort(-sims)[:CANDIDATES_DENSE])
            overlap = [(len(qtoks & t), ci) for ci, t in enumerate(d["toks"])]
            picked += [ci for n, ci in sorted(overlap, reverse=True)[:CANDIDATES_KEYWORD] if n > 0]
            for ci in dict.fromkeys(int(c) for c in picked):
                cands.append((di, ci, float(sims[ci])))

        reranker = self._reranker()
        scored = None
        if reranker is not None and cands:
            try:
                texts = [docs[di]["children"][ci]["t"] for di, ci, _ in cands]
                rs = await asyncio.to_thread(
                    lambda: reranker.predict([(q, t) for t in texts], batch_size=16,
                                             show_progress_bar=False, convert_to_numpy=True))
                scored = [(float(r), di, ci, s) for r, (di, ci, s) in zip(rs, cands) if r >= ATTACH_MIN_RERANK]
            except Exception as e:
                log.error("uploads: rerank failed (%s); using similarity", e)
        if scored is None:
            scored = [(s, di, ci, s) for di, ci, s in cands if s >= ATTACH_MIN_SIM]
        scored.sort(key=lambda x: -x[0])
        if DEBUG:
            log.info("uploads: %r -> %s", q[:50], "; ".join(
                f"{docs[di]['meta']['name'][:20]} p{docs[di]['children'][ci]['pg']} score={sc:.2f} sim={s:.2f}"
                for sc, di, ci, s in scored[:8]))

        # opening of each file first (title, authors, abstract), then the best matches
        hits, used, seen = [], 0, set()
        per_overview = min(OVERVIEW_CHARS, budget // (2 * len(docs)))
        for di, d in enumerate(docs):
            if d["parents"]:
                text = _trim(d["parents"][0], per_overview)
                hits.append(_hit(d, text, d["children"][0]["pg"], "opening of the file", di))
                used += len(text)
        for sc, di, ci, s in scored:
            if len(hits) >= ATTACH_MAX_PASSAGES:
                break
            c = docs[di]["children"][ci]
            key = (di, c["p"])
            if key in seen:
                continue
            seen.add(key)
            text = docs[di]["parents"][c["p"]]
            if c["p"] == 0:                              # replace the short opening with the full passage
                hits = [h for h in hits if not (h["_doc"] == di and h["heading"] == "opening of the file")]
                used = sum(len(h["text"]) for h in hits)
            room = budget - used
            if room < 400:
                break
            if len(text) > room:
                text = _trim(text, room)
            hits.append(_hit(docs[di], text, c["pg"], c["h"], di))
            used += len(text)
        return hits


# ------------------------------------------------------------------ helpers
def _parse(name: str, ext: str, data: bytes):
    """Returns (title, pages). Runs in a worker thread."""
    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)                                   # Windows: close before another library opens it
    p = Path(tmp)
    try:
        p.write_bytes(data)
        try:
            if ext == ".pdf":
                pages = ingest.strip_repeated_lines(ingest.parse_pdf(p))
            elif ext == ".docx":
                pages = ingest.parse_docx(p)
            else:
                pages = ingest.parse_text(p)
        except Exception as e:
            log.warning("upload parse failed for %s: %s", name, e)
            raise UploadError("This file couldn't be read. It may be damaged or password-protected.", 422)
    finally:
        p.unlink(missing_ok=True)
    pages = [(n, t) for n, t in pages if t and t.strip()]
    fallback = Path(name).stem.replace("_", " ").replace("-", " ")
    return ingest.detect_title(pages, fallback) if pages else fallback, pages


def _hit(doc: dict, text: str, page, heading: str, di: int = -1) -> dict:
    return {"kind": "file", "title": doc["meta"]["name"], "page": page or 0, "heading": heading or "",
            "text": text, "_doc": di}


def _trim(text: str, limit: int) -> str:
    """Cut at a sentence or line end near the limit."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[:end + 1] if end > limit * 0.6 else cut).rstrip() + " …"


def _spread(docs: list[dict], budget: int) -> list[dict]:
    """Excerpts evenly spaced through each file, in document order, for summaries."""
    hits = []
    per_doc = budget // len(docs)
    for d in docs:
        n = len(d["parents"])
        if not n:
            continue
        k = max(1, min(n, per_doc // SPREAD_PIECE_CHARS))
        idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, k)))
        page_of = {}
        for c in d["children"]:
            page_of.setdefault(c["p"], (c["pg"], c["h"]))
        piece = per_doc // len(idx)
        for i in idx:
            pg, h = page_of.get(i, (0, ""))
            hits.append(_hit(d, _trim(d["parents"][i], piece), pg, h or f"excerpt {idx.index(i) + 1} of {len(idx)}"))
    return hits
