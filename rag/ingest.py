"""
ARCH Assistant — document ingestion for RAG.

Walks a docs folder (PDF, DOCX, MD, TXT, HTML) plus an optional urls.txt of
web pages, splits everything into chunks, embeds them with a local Ollama
embedding model, and stores them in a ChromaDB index the proxy reads at
query time. Idempotent: unchanged files are skipped, changed files are
re-indexed, deleted files are removed from the index.

Usage:
    python ingest.py                  # index ./docs
    python ingest.py --docs D:\arch   # a different folder
    python ingest.py --rebuild        # wipe and re-index everything
    python ingest.py --stats          # show what's indexed

Layout of the docs folder:
    docs/
      anything.pdf, anything.docx, anything.md, anything.txt, anything.html
      subfolders are fine; the top-level subfolder name becomes the "collection"
      urls.txt      one URL per line; fetched and indexed like a document
                    "site <start-url> <url-prefix> [max-pages]" crawls a whole site section:
                    starts at start-url, follows links that begin with url-prefix, stops at
                    max-pages (default 100). Example:
                    site https://huggingface.co/learn/llm-course/chapter1/1 https://huggingface.co/learn/llm-course/ 120
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ------------------------------------------------------------------ config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("ARCH_EMBED_MODEL", "embeddinggemma")

_default_data = (Path(os.environ["LOCALAPPDATA"]) / "ARCHAssistant"
                 if os.name == "nt" and "LOCALAPPDATA" in os.environ
                 else Path(__file__).resolve().parent.parent / "data")
DATA_DIR = Path(os.environ.get("ARCH_DATA_DIR", _default_data))
INDEX_DIR = DATA_DIR / "index"
MANIFEST = INDEX_DIR / "manifest.json"
FTS_DB = INDEX_DIR / "fts.sqlite"          # SQLite FTS5 keyword index (parallel to Chroma)

CHILD_TOKENS = 220          # what gets embedded and matched
PARENT_TOKENS = 900         # what gets sent to the model when a child matches
OVERLAP_TOKENS = 40
SUPPORTED = {".pdf", ".docx", ".md", ".mdx", ".txt", ".html", ".htm"}


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


# ------------------------------------------------------------------ parsing
def parse_pdf(p: Path):
    """Extract text per page, marking headings (lines set noticeably larger than body
    text, or bold and short) as '## ' lines so the chunker splits on them."""
    import pymupdf as fitz
    from collections import Counter

    with fitz.open(p) as doc:
        page_dicts = [page.get_text("dict") for page in doc]

    # body font size = the most common size across the document (by character count)
    size_chars = Counter()
    for pd in page_dicts:
        for b in pd["blocks"]:
            for l in b.get("lines", []):
                for sp in l["spans"]:
                    size_chars[round(sp["size"], 1)] += len(sp["text"])
    if not size_chars:
        print(f"  WARNING: {p.name} has no text layer. Skipping; OCR it first.")
        return []
    body = size_chars.most_common(1)[0][0]

    pages = []
    for i, pd in enumerate(page_dicts, 1):
        out = []
        for b in pd["blocks"]:
            for l in b.get("lines", []):
                spans = l["spans"]
                text = "".join(sp["text"] for sp in spans).strip()
                if not text:
                    continue
                size = max(sp["size"] for sp in spans)
                bold = all(("bold" in sp["font"].lower()) or (sp["flags"] & 16) for sp in spans)
                is_heading = (size >= body * 1.15 and len(text) < 120) or \
                             (bold and len(text) < 80 and not text.endswith((".", ",", ";")))
                # skip bare numbers (page numbers) and single symbols
                if re.fullmatch(r"[\d\s.\-]+", text):
                    continue
                out.append(f"\n## {text}\n" if is_heading else text)
        pages.append((i, "\n".join(out)))

    chars = sum(len(t) for _, t in pages)
    if chars < 50 * max(1, len(pages)):
        print(f"  WARNING: {p.name} looks scanned (almost no text layer). Skipping; OCR it first.")
        return []
    return pages


def parse_docx(p: Path):
    import docx
    d = docx.Document(p)
    parts = []
    for para in d.paragraphs:
        t = para.text.strip()
        if not t:
            continue
        style = (para.style.name or "").lower()
        if style == "title" or style == "heading 0":
            parts.append(f"# {t}")
        elif style.startswith("heading"):
            parts.append(f"\n## {t}\n")
        else:
            parts.append(t)
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [(None, "\n".join(parts))]


def parse_text(p: Path):
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".mdx":
        # drop JSX components and imports (<Tip>, <Youtube id=.../>, import ... from ...)
        text = re.sub(r"^import .*$", "", text, flags=re.M)
        text = re.sub(r"<[A-Z][A-Za-z]*[^>]*/>", "", text)
        text = re.sub(r"</?[A-Z][A-Za-z]*[^>]*>", "", text)
        text = re.sub(r"\{/\*.*?\*/\}", "", text, flags=re.S)
    # markdown headings: keep "# Title" for detect_title, turn "##"/"###" into chunk boundaries
    text = re.sub(r"^#{2,6}\s+(.+)$", r"## \1", text, flags=re.M)
    return [(None, text)]


def _parser():
    try:
        import lxml  # noqa: F401
        return "lxml"
    except ImportError:
        return "html.parser"


def html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, _parser())
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        t.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = []
    for el in main.descendants:
        if el.name in ("h1", "h2", "h3", "h4"):
            lines.append(f"\n## {el.get_text(' ', strip=True)}\n")
        elif el.name in ("p", "li", "td", "th"):
            t = el.get_text(" ", strip=True)
            if t:
                lines.append(t)
    text = "\n".join(lines)
    return title, text


def parse_html_file(p: Path):
    title, text = html_to_text(p.read_text(encoding="utf-8", errors="replace"))
    return [(None, text)]


_UA = {"User-Agent": "ARCH-Assistant-indexer/1.0 (local research index)"}


def fetch_url(url: str):
    """Fetch a URL. Returns (title, pages) where pages is [(page_no, text), ...].
    HTML is parsed to text; a PDF is downloaded and parsed like a local PDF."""
    import tempfile
    with httpx.Client(timeout=120, follow_redirects=True, headers=_UA) as c:
        with c.stream("GET", url) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            is_pdf = "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf")
            if is_pdf:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    for chunk in r.iter_bytes():
                        tmp.write(chunk)
                    tmp_path = Path(tmp.name)
                try:
                    pages = strip_repeated_lines(parse_pdf(tmp_path))
                finally:
                    tmp_path.unlink(missing_ok=True)
                name = url.rstrip("/").split("/")[-1].rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
                return detect_title(pages, name), pages
            if "html" not in ctype and "text" not in ctype:
                raise ValueError(f"unsupported content type {ctype or 'unknown'}")
            html = r.read().decode(r.encoding or "utf-8", errors="replace")
    title, text = html_to_text(html)
    return title, [(None, text)]


def _norm_url(u: str) -> str:
    from urllib.parse import urlsplit, urlunsplit
    p = urlsplit(u)
    path = re.sub(r"/index\.html?$", "", p.path).rstrip("/") or "/"
    return urlunsplit((p.scheme, p.netloc.lower(), path, "", ""))   # drop query, fragment, index.html


def crawl_site(start: str, prefix: str, max_pages: int = 100):
    """Breadth-first crawl within `prefix`. Yields (url, title, text). HTML pages only."""
    from urllib.parse import urljoin
    from bs4 import BeautifulSoup
    prefix_n = _norm_url(prefix)
    queue, seen, done = [_norm_url(start)], {_norm_url(start)}, 0
    with httpx.Client(timeout=30, follow_redirects=True, headers=_UA) as c:
        while queue and done < max_pages:
            url = queue.pop(0)
            print(f"  fetching {url} ...", end="", flush=True)
            try:
                r = c.get(url)
                if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                    print(f" skipped ({r.status_code}, {r.headers.get('content-type','?')[:30]})")
                    continue
            except Exception as e:
                print(f" error: {e}")
                continue
            html = r.text
            if len(html) > 6_000_000:
                print(f" skipped ({len(html)//1_000_000} MB page)")
                continue
            print(f" {len(html)//1000} KB")
            title, text = html_to_text(html)
            done += 1
            yield url, title, text
            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                href = a["href"]
                if href.startswith(("mailto:", "javascript:", "#")):
                    continue
                nu = _norm_url(urljoin(url, href))
                if nu.startswith(prefix_n) and nu not in seen and not nu.lower().endswith(
                        (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ipynb", ".csv")):
                    seen.add(nu)
                    queue.append(nu)
            time.sleep(0.4)      # be polite


# ------------------------------------------------------------------ cleaning
def clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop lines that are just page numbers
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


def strip_repeated_lines(pages):
    """Remove header/footer lines that appear on most pages."""
    if len(pages) < 4:
        return pages
    from collections import Counter
    counts = Counter()
    for _, t in pages:
        for line in {l.strip() for l in t.splitlines() if 3 < len(l.strip()) < 80}:
            counts[line] += 1
    repeated = {l for l, c in counts.items() if c >= 0.6 * len(pages)}
    out = []
    for n, t in pages:
        kept = [l for l in t.splitlines() if l.strip() not in repeated]
        out.append((n, "\n".join(kept)))
    return out


# ------------------------------------------------------------------ chunking
def split_recursive(text: str, target: int, overlap: int):
    """Split on headings, then paragraphs, then sentences, aiming for `target` tokens."""
    seps = ["\n## ", "\n\n", "\n", ". ", " "]

    def _split(s, level):
        if approx_tokens(s) <= target or level >= len(seps):
            return [s]
        parts = s.split(seps[level])
        sep = seps[level]
        out, buf = [], ""
        for part in parts:
            piece = (sep + part) if buf else part
            if approx_tokens(buf + piece) > target and buf:
                out.append(buf)
                buf = part
            else:
                buf += piece
        if buf:
            out.append(buf)
        final = []
        for o in out:
            final.extend(_split(o, level + 1) if approx_tokens(o) > target else [o])
        return final

    chunks = [c.strip() for c in _split(text, 0) if c.strip()]
    if overlap and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i > 0:
                tail = chunks[i - 1][-overlap * 4:]
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                c = tail + " " + c
            overlapped.append(c)
        chunks = overlapped
    return chunks


def heading_for(parent: str, text_before: str) -> str:
    """First heading inside the chunk, else the last heading seen before it."""
    inside = re.findall(r"^## (.+)$", parent, flags=re.M)
    if inside:
        return inside[0].strip()
    before = re.findall(r"^## (.+)$", text_before, flags=re.M)
    return before[-1].strip() if before else ""


def detect_title(pages, fallback: str) -> str:
    """Use the document's own title line when it has one."""
    first = (pages[0][1] if pages else "")[:600]
    for m in re.finditer(r"^#\s+(.+)$|^## (.+)$", first, flags=re.M):
        cand = (m.group(1) or m.group(2)).strip()
        # skip section numbers ("1.2 Degree"), chapter labels, and very short lines
        if re.match(r"^(\d+(\.\d+)*|chapter|section|part|appendix)\b", cand, flags=re.I):
            continue
        if 4 <= len(cand) <= 120:
            return cand
    return fallback


def build_chunks(doc_id: str, title: str, collection: str, pages):
    """Return list of child chunks, each carrying its parent text."""
    children = []
    for page_no, raw in pages:
        text = clean(raw)
        if not text:
            continue
        parents = split_recursive(text, PARENT_TOKENS, 0)
        seen = ""
        for pi, parent in enumerate(parents):
            heading = heading_for(parent, seen)
            seen += parent
            kids = split_recursive(parent, CHILD_TOKENS, OVERLAP_TOKENS)
            for ki, kid in enumerate(kids):
                children.append({
                    "id": f"{doc_id}:{page_no or 0}:{pi}:{ki}",
                    "text": kid,
                    "parent": parent,
                    "meta": {
                        "doc_id": doc_id, "title": title, "collection": collection,
                        "page": page_no or 0, "heading": heading,
                    },
                })
    return children


# ------------------------------------------------------------------ embedding
def embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    out = []
    B = 32
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(texts), B):
            batch = [prefix + t for t in texts[i:i + B]]
            r = c.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": batch})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
            print(f"  embedded {min(i + B, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def doc_prefix(meta: dict) -> str:
    h = f" Section: {meta['heading']}." if meta.get("heading") else ""
    return f"title: {meta['title']}{h} | text: "


# ------------------------------------------------------------------ store
def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_or_create_collection(
        "arch_docs", metadata={"hnsw:space": "cosine"})


def fts_conn():
    import sqlite3
    con = sqlite3.connect(FTS_DB)
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                "id UNINDEXED, doc_id UNINDEXED, text, tokenize='porter unicode61')")
    return con


def fts_delete_doc(con, doc_id: str):
    con.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    con.commit()


def fts_add(con, children):
    con.executemany("INSERT INTO chunks(id, doc_id, text) VALUES (?, ?, ?)",
                    [(c["id"], c["meta"]["doc_id"], c["text"]) for c in children])
    con.commit()


def fts_rebuild_from_chroma(con, col):
    """Populate FTS from an existing Chroma index (no re-embedding)."""
    con.execute("DELETE FROM chunks")
    total = col.count()
    page = 2000
    for off in range(0, total, page):
        d = col.get(include=["documents", "metadatas"], limit=page, offset=off)
        con.executemany("INSERT INTO chunks(id, doc_id, text) VALUES (?, ?, ?)",
                        [(i, m["doc_id"], t) for i, t, m in zip(d["ids"], d["documents"], d["metadatas"])])
        print(f"  keyword index: {min(off + page, total)}/{total}", end="\r")
    con.commit()
    print()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=1))


def file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def remove_doc(col, doc_id: str):
    col.delete(where={"doc_id": doc_id})


def index_children(col, children):
    if not children:
        return
    vecs = embed([doc_prefix(c["meta"]) + c["text"] for c in children])
    col.upsert(
        ids=[c["id"] for c in children],
        embeddings=vecs,
        documents=[c["text"] for c in children],
        metadatas=[{**c["meta"], "parent": c["parent"]} for c in children],
    )


# ------------------------------------------------------------------ main
def run(docs_dir: Path, rebuild: bool):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild and INDEX_DIR.exists():
        import shutil
        for child in INDEX_DIR.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    col = get_collection()
    fts = fts_conn()
    n_fts = fts.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if col.count() > 0 and n_fts != col.count():
        print(f"Building keyword index for {col.count()} existing chunks ...")
        fts_rebuild_from_chroma(fts, col)
    manifest = load_manifest()
    seen_ids = set()
    listed_urls, listed_prefixes = set(), []     # what urls.txt currently covers
    t0 = time.time()

    # files
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED and p.name != "urls.txt"]
    print(f"Found {len(files)} documents in {docs_dir}")
    for p in sorted(files):
        rel = p.relative_to(docs_dir)
        doc_id = "file:" + str(rel).replace("\\", "/")
        seen_ids.add(doc_id)
        h = file_hash(p)
        if manifest.get(doc_id, {}).get("hash") == h:
            continue
        print(f"- {rel}")
        try:
            ext = p.suffix.lower()
            if ext == ".pdf":
                pages = strip_repeated_lines(parse_pdf(p))
            elif ext == ".docx":
                pages = parse_docx(p)
            elif ext in (".html", ".htm"):
                pages = parse_html_file(p)
            else:
                pages = parse_text(p)
        except Exception as e:
            print(f"  ERROR parsing {rel}: {e}")
            continue
        collection = rel.parts[0] if len(rel.parts) > 1 else "general"
        title = detect_title(pages, p.stem.replace("_", " ").replace("-", " "))
        children = build_chunks(doc_id, title, collection, pages)
        remove_doc(col, doc_id); fts_delete_doc(fts, doc_id)
        index_children(col, children); fts_add(fts, children)
        manifest[doc_id] = {"hash": h, "chunks": len(children), "title": title,
                            "indexed": int(time.time())}
        save_manifest(manifest)
        print(f"  {len(children)} chunks")

    # urls
    def index_page(url: str, title: str, pages, collection: str):
        doc_id = "url:" + url
        seen_ids.add(doc_id)
        text_all = "\n".join(t for _, t in pages)
        if len(text_all) < 200:
            print(f"  {url}: too little text, skipped")
            return
        h = hashlib.sha256(text_all.encode()).hexdigest()[:16]
        if manifest.get(doc_id, {}).get("hash") == h:
            return
        children = build_chunks(doc_id, title or url, collection, pages)
        remove_doc(col, doc_id); fts_delete_doc(fts, doc_id)
        index_children(col, children); fts_add(fts, children)
        manifest[doc_id] = {"hash": h, "chunks": len(children), "title": title or url,
                            "indexed": int(time.time())}
        save_manifest(manifest)
        print(f"  {url}: {len(children)} chunks")

    urls_file = docs_dir / "urls.txt"
    if urls_file.exists():
        lines = [u.strip() for u in urls_file.read_text(encoding="utf-8").splitlines()
                 if u.strip() and not u.strip().startswith("#")]
        print(f"Found {len(lines)} URL entries")
        for line in lines:
            parts = line.split()
            if parts[0].lower() == "site" and len(parts) >= 3:
                start, prefix = parts[1], parts[2]
                listed_prefixes.append(_norm_url(prefix))
                max_pages = int(parts[3]) if len(parts) > 3 else 100
                from urllib.parse import urlsplit
                collection = "web-" + urlsplit(start).netloc.replace("www.", "").split(".")[0]
                print(f"- crawling {start} (within {prefix}, max {max_pages} pages)")
                n = 0
                for url, title, text in crawl_site(start, prefix, max_pages):
                    index_page(url, title, [(None, text)], collection)
                    n += 1
                print(f"  {n} pages visited")
            else:
                url = parts[0]
                listed_urls.add(url)
                print(f"- {url}")
                try:
                    title, pages = fetch_url(url)
                except Exception as e:
                    print(f"  ERROR fetching: {e}")
                    continue
                index_page(url, title, pages, "web")

    # deletions
    # A local file is removed from the index when the file is gone. A web page is removed
    # only when urls.txt no longer lists it (or the site it was crawled from): a failed
    # download, an offline run, or a crawl that reached fewer pages must never delete
    # content that was indexed earlier.
    kept_web = 0
    for doc_id in list(manifest):
        if doc_id in seen_ids:
            continue
        if doc_id.startswith("url:"):
            url = doc_id[4:]
            if url in listed_urls or any(_norm_url(url).startswith(pfx) for pfx in listed_prefixes):
                kept_web += 1
                continue
            reason = "no longer in urls.txt"
        else:
            reason = "file gone"
        print(f"- removing {doc_id} ({reason})")
        remove_doc(col, doc_id); fts_delete_doc(fts, doc_id)
        del manifest[doc_id]
    if kept_web:
        print(f"Kept {kept_web} web page(s) that could not be re-fetched this run.")

    save_manifest(manifest)
    total = sum(m["chunks"] for m in manifest.values())
    print(f"\nDone in {time.time() - t0:.0f}s. {len(manifest)} documents, {total} chunks, index at {INDEX_DIR}")
    print("Restart the proxy to pick up the new index.")


def stats():
    m = load_manifest()
    if not m:
        print("Index is empty."); return
    for doc_id, info in sorted(m.items()):
        print(f"{info['chunks']:5d}  {info['title']}   ({doc_id})")
    print(f"\n{len(m)} documents, {sum(i['chunks'] for i in m.values())} chunks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(Path(__file__).resolve().parent.parent / "docs"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.stats:
        stats(); sys.exit(0)
    d = Path(a.docs)
    if not d.is_dir():
        print(f"Docs folder not found: {d}\nCreate it and put PDF/DOCX/MD/TXT/HTML files in it, or pass --docs.")
        sys.exit(1)
    run(d, a.rebuild)
