"""
Regression tests for the proxy and the ingest script. No Ollama, GPU or internet needed:
the model backend is replaced by a stand-in, and everything is written to a temporary folder.

Run from the project folder, with the venv active:
    python tests\\test_fixes.py
"""
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="arch-test-"))
os.environ["ARCH_DATA_DIR"] = str(TMP / "data")
os.environ["ARCH_RERANKER_DIR"] = str(TMP / "no-reranker")
os.chdir(TMP)                                   # guardrail.log goes to the temp folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))

import httpx
from fastapi.testclient import TestClient

import guardrail_proxy as gp

MARK = "FORBIDDENMARK"                          # the stand-in classifier flags text containing this
state = {"reply": "A normal answer."}


def fake_ollama(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content or b"{}")
    path = request.url.path
    if path == "/api/generate":                 # ShieldGemma stand-in
        return httpx.Response(200, json={"response": "Yes" if MARK in body["prompt"] else "No"})
    if path == "/api/show":
        return httpx.Response(200, json={"parameters": "num_ctx 8192", "system": "x" * 11000})
    if path == "/api/chat":
        return httpx.Response(200, json={"message": {"role": "assistant", "content": state["reply"]}})
    return httpx.Response(200, json={"models": []})


gp.client = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(fake_ollama))
web = TestClient(gp.app, base_url="http://localhost:8080")

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"\n        {detail}"))


def ask(text, **headers):
    r = web.post("/v1/chat/completions", json={"model": "cvex-ai", "messages": [{"role": "user", "content": text}]},
                 headers=headers)
    return r


# ---------------------------------------------------------------- 1. chats are never pruned
for i in range(205):
    web.put(f"/api/chats/chat{i:04d}", json={"title": f"t{i}", "messages": []})
check("1. all 205 saved chats are still there", len(web.get("/api/chats").json()) == 205,
      len(web.get("/api/chats").json()))

# ---------------------------------------------------------------- 2. whole text is screened
long_clean = "word " * 1500                                             # 7500 chars
check("2a. windows cover every character of a long text",
      "".join(w[:gp.SHIELD_WINDOW_CHARS - gp.SHIELD_OVERLAP_CHARS] for w in gp.shield_windows(long_clean)[:-1])
      + gp.shield_windows(long_clean)[-1] == long_clean)
state["reply"] = long_clean + MARK                                      # violation at the very end
r = ask("hello there, explain something long").json()
check("2b. violation at the end of a long reply is blocked",
      r["choices"][0]["message"]["content"] == gp.POLICY["output_refusal_message"], r)
state["reply"] = "A normal answer."
r = ask(long_clean + MARK).json()
check("2c. violation at the end of a long question is blocked",
      r["choices"][0]["message"]["content"] == gp.POLICY["refusal_message"], r)
r = ask("a normal question").json()
check("2d. normal exchange still passes", r["choices"][0]["message"]["content"] == "A normal answer.", r)

# ---------------------------------------------------------------- 4. startup logging order
src = (ROOT / "guardrail_proxy.py").read_text(encoding="utf-8")
check("4. logging is configured before the document index loads",
      src.index("logging.basicConfig(") < src.index("Retriever(DATA_DIR"))

# ---------------------------------------------------------------- 5. only local pages may call the proxy
check("5a. request for another hostname is refused", web.get("/api/chats", headers={"host": "evil.example"}).status_code == 403)
check("5b. request sent by another website is refused", ask("hi", origin="http://evil.example").status_code == 403)
check("5c. localhost and 127.0.0.1 are accepted",
      web.get("/api/chats").status_code == 200
      and web.get("/api/chats", headers={"host": "127.0.0.1:8080"}).status_code == 200
      and ask("hi", origin="http://localhost:8080").status_code == 200)

# ---------------------------------------------------------------- 6. requirements
req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("6. requirements: unused package removed, GPU PyTorch step documented",
      "rank-bm25" not in req and "torch.cuda.is_available()" in req)

# ---------------------------------------------------------------- 3. ingest keeps web pages it cannot re-fetch
import ingest


def fake_embed(texts, prefix=""):
    out = []
    for t in texts:
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        out.append([x / n for x in v])
    return out


PAGE = "Graph neural networks pass messages between neighbouring nodes. " * 10
online = {"up": True}


def fake_fetch(url):
    if not online["up"]:
        raise ConnectionError("offline")
    return "A web page", [(None, PAGE)]


ingest.embed = fake_embed
ingest.fetch_url = fake_fetch
docs = TMP / "docs"
docs.mkdir()
(docs / "note.md").write_text("# Note\n\n" + "Trees are connected acyclic graphs. " * 10, encoding="utf-8")
(docs / "urls.txt").write_text("https://example.org/page\n", encoding="utf-8")

ingest.run(docs, rebuild=False)
m = ingest.load_manifest()
check("3a. file and web page indexed", set(m) == {"file:note.md", "url:https://example.org/page"}, set(m))

online["up"] = False
ingest.run(docs, rebuild=False)
m = ingest.load_manifest()
check("3b. web page survives a failed download", "url:https://example.org/page" in m, set(m))

(docs / "urls.txt").write_text("# nothing listed\n", encoding="utf-8")
(docs / "note.md").unlink()
ingest.run(docs, rebuild=False)
m = ingest.load_manifest()
check("3c. page removed once dropped from urls.txt; deleted file removed too", m == {}, set(m))
n_fts = ingest.fts_conn().execute("SELECT count(*) FROM chunks").fetchone()[0]
check("3d. keyword index emptied to match", n_fts == 0 and ingest.get_collection().count() == 0, n_fts)

asyncio.run(gp.client.aclose())
print(f"\n{sum(results)} passed, {len(results) - sum(results)} failed")
sys.exit(0 if all(results) else 1)
