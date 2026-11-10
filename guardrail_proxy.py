"""
CVEX guardrail proxy.

Sits between the chat UI and Ollama. Exposes an OpenAI-compatible
/v1/chat/completions endpoint so any ChatGPT-style UI (Open WebUI, etc.)
can talk to it.

Per request:
  1. Regex rules on the user message      (policy.json -> blocked_input_patterns)
  2. ShieldGemma on the user message       (policy.json -> shieldgemma)
  2b. Retrieve reference passages (RAG), pick the learning mode (modes.json), trim old
      turns to fit the context window, and prepend an application block to the latest
      user turn carrying the mode and the passages
  3. Forward to the main model via Ollama
  4. ShieldGemma on the model reply
  5. Regex rules on the model reply        (policy.json -> blocked_output_patterns)

Every decision is written to guardrail.log.

Run:  python guardrail_proxy.py
"""

import os
# Cap BLAS threads BEFORE numpy loads (via chromadb / rank_bm25). On Windows, OpenBLAS
# otherwise allocates a large buffer per core at startup and can fail with
# "Memory allocation still failed after 10 retries" when RAM is tight.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json
import logging
import re
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Logging is configured before anything else so that startup messages (index size,
# reranker device, config problems) are not lost.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("guardrail.log", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("guardrail")

# ---------------------------------------------------------------- config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAIN_MODEL = "cvex-ai"          # created with: ollama create cvex-ai -f Modelfile
LISTEN_HOST = "127.0.0.1"       # local machine only; never expose on the network
LISTEN_PORT = 8080
POLICY_PATH = Path(__file__).with_name("policy.json")
KEEP_ALIVE = "30m"              # keep models loaded between requests
SHIELD_ON_CPU = True            # ShieldGemma 2B on CPU: ~1-2s per check, frees ~1.7 GB VRAM for the main model.
                                # Set False on a 16 GB+ card where everything fits.
THINK = False                   # Gemma 4 thinking mode off: faster, and avoids empty replies

# Learning modes. Behaviour is defined in the Modelfile (LEARNING MODES); modes.json holds
# the ids plus the text the UI shows. The proxy only accepts ids listed there.
MODES_PATH = Path(__file__).with_name("modes.json")
try:
    _modes_cfg = json.loads(MODES_PATH.read_text(encoding="utf-8"))
except Exception as _e:
    _modes_cfg = {"default": "explain", "modes": [{"id": "explain", "label": "Explain"}]}
    logging.getLogger("guardrail").warning("modes.json unreadable (%s); only Explain mode available", _e)
MODES = {m["id"]: m for m in _modes_cfg.get("modes", [])}
DEFAULT_MODE = _modes_cfg.get("default", "explain") if _modes_cfg.get("default") in MODES else next(iter(MODES))

# Context-window budget. The model's num_ctx and system prompt are read from Ollama on the
# first request (falls back to these values). Old turns are dropped, oldest first, so that
# system prompt + retrieved passages + conversation + room for the reply fit in num_ctx.
FALLBACK_NUM_CTX = 8192
FALLBACK_SYSTEM_CHARS = 11700
REPLY_RESERVE_TOKENS = 1536     # room left for the answer
CHARS_PER_TOKEN = 3.5           # conservative for English with some LaTeX/code
MIN_HISTORY_CHARS = 1500        # never squeeze the conversation below this

# Where chat history is stored. Override with the ARCH_DATA_DIR environment
# variable. Default on Windows: %LOCALAPPDATA%\ARCHAssistant ; elsewhere ./data
_default_data = (Path(os.environ["LOCALAPPDATA"]) / "ARCHAssistant"
                 if os.name == "nt" and "LOCALAPPDATA" in os.environ
                 else Path(__file__).parent / "data")
DATA_DIR = Path(os.environ.get("ARCH_DATA_DIR", _default_data))
CHATS_DIR = DATA_DIR / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

# RAG (optional). Index is built by rag/ingest.py. If no index exists, the
# assistant simply runs without document retrieval.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent / "rag"))
try:
    from retrieve import Retriever, format_context
    RAG = Retriever(DATA_DIR / "index")
except Exception as _e:          # missing deps etc. — run without RAG
    RAG = None
    logging.getLogger("rag").warning("rag disabled: %s", _e)

    def format_context(hits):    # never called without hits, kept so app_block always resolves
        return ""
# ---------------------------------------------------------------- at-rest encryption
# On Windows, chat files are encrypted with DPAPI (CryptProtectData), keyed to
# the logged-in Windows user. Other accounts on the same PC, or anyone who copies
# the folder elsewhere, cannot read them. On other platforms (dev only) files are
# stored as plain JSON and a warning is logged.
CHAT_EXT = ".arch"
_DPAPI_DESCRIPTION = "ARCH Assistant chat"

if os.name == "nt":
    import ctypes
    import ctypes.wintypes as _wt

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", _wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.windll.crypt32
    _kernel32 = ctypes.windll.kernel32
    _CRYPTPROTECT_UI_FORBIDDEN = 0x01

    def _dpapi(data: bytes, encrypt: bool) -> bytes:
        buf = ctypes.create_string_buffer(data, len(data))
        inp = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        out = _DATA_BLOB()
        if encrypt:
            ok = _crypt32.CryptProtectData(ctypes.byref(inp), _DPAPI_DESCRIPTION, None, None, None,
                                           _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
        else:
            ok = _crypt32.CryptUnprotectData(ctypes.byref(inp), None, None, None, None,
                                             _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
        if not ok:
            raise OSError(f"DPAPI {'encrypt' if encrypt else 'decrypt'} failed: {ctypes.GetLastError()}")
        try:
            return ctypes.string_at(out.pbData, out.cbData)
        finally:
            _kernel32.LocalFree(out.pbData)

    def protect(data: bytes) -> bytes:
        return _dpapi(data, True)

    def unprotect(data: bytes) -> bytes:
        return _dpapi(data, False)

    ENCRYPTED = True
else:
    def protect(data: bytes) -> bytes:
        return data

    def unprotect(data: bytes) -> bytes:
        return data

    ENCRYPTED = False


def read_chat_file(p: Path) -> dict:
    return json.loads(unprotect(p.read_bytes()).decode("utf-8"))


def write_chat_file(p: Path, d: dict) -> None:
    raw = json.dumps(d, ensure_ascii=False).encode("utf-8")
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(protect(raw))
    tmp.replace(p)                       # atomic on the same filesystem


POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
INPUT_RULES = [(r["name"], re.compile(r["regex"])) for r in POLICY["blocked_input_patterns"]]
OUTPUT_RULES = [(r["name"], re.compile(r["regex"])) for r in POLICY["blocked_output_patterns"]]
SG = POLICY["shieldgemma"]

app = FastAPI(title="ARCH Assistant Guardrail Proxy")
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=300)

# Only this computer's own browser may talk to the proxy. Binding to 127.0.0.1 keeps other
# machines out; these checks keep other WEBSITES out. A hostile page open in the same browser
# can aim requests at localhost (DNS rebinding, cross-site form posts); such requests carry
# the attacker's hostname in Host or Origin, so they are refused here.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


def _is_local(hostport: str) -> bool:
    host = hostport.strip().lower()
    host = host.rsplit(":", 1)[0] if (":" in host and not host.endswith("]")) else host
    return host in _LOCAL_HOSTS


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    origin_host = re.sub(r"^[a-z]+://", "", origin, flags=re.I) if origin else None
    if not _is_local(host) or (origin_host is not None and not _is_local(origin_host)):
        log.warning("refused non-local request: host=%r origin=%r path=%s", host, origin, request.url.path)
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return await call_next(request)


# ---------------------------------------------------------------- guardrails
def regex_check(text: str, rules) -> str | None:
    """Return the name of the first matching rule, or None."""
    for name, pattern in rules:
        if pattern.search(text):
            return name
    return None


def build_shieldgemma_prompt(text: str, role: str) -> str:
    """Official ShieldGemma prompt format. role = 'input' or 'output'."""
    policies = "\n".join(f"* {p}" for p in SG["policies"])
    if role == "input":
        header = ("You are a policy expert trying to help determine whether a user "
                  "prompt is in violation of the defined safety policies.")
        body = f"<start_of_turn>\nHuman Question: {text}\n<end_of_turn>"
        question = "Does the human question violate the above principle?"
    else:
        header = ("You are a policy expert trying to help determine whether an AI "
                  "response to a prompt is in violation of the defined safety policies.")
        body = f"<start_of_turn>\nChatbot Response: {text}\n<end_of_turn>"
        question = "Does the Chatbot Response violate the above principle?"
    exemption = SG.get("academic_exemption", "")
    exemption_block = f"\n\n{exemption}" if exemption else ""
    return (f"{header}\n\n{body}\n\nOur safety principle is defined in the below:\n\n"
            f"{policies}{exemption_block}\n\n{question} Your answer must start with 'Yes' or 'No'. "
            "And then walk through step by step to be sure we answer correctly.\n")


SHIELD_WINDOW_CHARS = 3500      # ShieldGemma sees this much text per call
SHIELD_OVERLAP_CHARS = 300      # windows overlap so nothing is split out of view


def shield_windows(text: str) -> list[str]:
    """Split text into overlapping windows that together cover all of it."""
    if len(text) <= SHIELD_WINDOW_CHARS:
        return [text]
    step = SHIELD_WINDOW_CHARS - SHIELD_OVERLAP_CHARS
    return [text[i:i + SHIELD_WINDOW_CHARS] for i in range(0, len(text) - SHIELD_OVERLAP_CHARS, step)]


async def shieldgemma_flags(text: str, role: str) -> bool:
    """True if ShieldGemma says any part of the text violates policy. Long text is checked
    window by window, stopping at the first violation."""
    if not SG.get("enabled", True):
        return False
    if not text.strip():
        return False
    for window in shield_windows(text):
        if await _shieldgemma_flags_one(window, role):
            return True
    return False


async def _shieldgemma_flags_one(text: str, role: str) -> bool:
    payload = {
        "model": SG["model"],
        "prompt": build_shieldgemma_prompt(text, role),
        "raw": True,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 4,
                    **({"num_gpu": 0} if SHIELD_ON_CPU else {})},
    }
    try:
        r = await client.post("/api/generate", json=payload)
        r.raise_for_status()
        answer = r.json().get("response", "").strip().lower()
        return answer.startswith("yes")
    except httpx.HTTPStatusError as e:          # fail closed on classifier errors
        log.error("shieldgemma error (%s) %s: %s", role, e.response.status_code, e.response.text[:300])
        return True
    except Exception as e:
        log.error("shieldgemma error (%s): %s", role, e)
        return True


# ---------------------------------------------------------------- helpers
def _text(c) -> str:
    if isinstance(c, list):                    # multimodal content array
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _text(m.get("content", ""))
    return ""


SHORT_FOLLOWUP_WORDS = 6


def retrieval_query(messages: list[dict], mode: str = "explain") -> str:
    """What to search for.

    A substantial message is searched as-is. A short follow-up ("yes go ahead", "B",
    "why?") carries no topic of its own, so the query is built from:
      * the most recent earlier user message that does carry a topic (more than
        SHORT_FOLLOWUP_WORDS words), else the first user message of the chat;
      * in Quiz and Tutor modes, the end of the assistant's last reply, which is where
        the question being answered sits;
      * the follow-up itself.
    """
    users = [_text(m.get("content", "")).strip() for m in messages if m.get("role") == "user"]
    if not users:
        return ""
    latest = users[-1]
    if len(users) == 1 or len(latest.split()) > SHORT_FOLLOWUP_WORDS:
        return latest
    earlier = users[:-1]
    anchor = next((u for u in reversed(earlier) if len(u.split()) > SHORT_FOLLOWUP_WORDS),
                  earlier[0] if earlier else "")
    parts = [anchor]
    if mode in ("quiz", "tutor"):
        last_bot = next((_text(m.get("content", "")) for m in reversed(messages)
                         if m.get("role") == "assistant"), "")
        parts.append(last_bot[-400:])
    parts.append(latest)
    return "\n".join(p for p in parts if p and p.strip())


# ---------------------------------------------------------------- learning modes + context assembly
_MARKER_RE = re.compile(r"\[\s*(ARCH app|Student message)\s*\]", re.I)


def defuse_markers(text: str) -> str:
    """Stop a student from forging the application block by typing its markers."""
    return _MARKER_RE.sub(lambda m: f"({m.group(1)})", text)


def app_block(mode: str, hits: list[dict]) -> str:
    """The application block prepended to the latest user turn (see Modelfile: APPLICATION BLOCK)."""
    label = MODES.get(mode, {}).get("label", mode.title())
    parts = ["[ARCH app]", f"Mode: {label}"]
    if hits:
        parts += ["", format_context(hits).rstrip()]
    parts.append("[Student message]")
    return "\n".join(parts) + "\n"


def trim_history(messages: list[dict], budget_chars: int) -> tuple[list[dict], int]:
    """Keep the newest turns that fit in budget_chars. Always keeps the latest user turn and the
    assistant turn just before it (a quiz question or tutor hint), then adds older turns while
    they fit. Returns (kept, number_dropped)."""
    if not messages:
        return messages, 0
    kept = [messages[-1]]
    used = len(_text(messages[-1].get("content", "")))
    i = len(messages) - 2
    if i >= 0 and messages[i].get("role") == "assistant":
        kept.insert(0, messages[i])
        used += len(_text(messages[i].get("content", "")))
        i -= 1
    while i >= 0:
        n = len(_text(messages[i].get("content", "")))
        if used + n > budget_chars:
            break
        kept.insert(0, messages[i])
        used += n
        i -= 1
    while len(kept) > 2 and kept[0].get("role") != "user":   # start on a user turn
        kept.pop(0)
    return kept, len(messages) - len(kept)


_budget_cache: dict = {}


async def model_budget() -> tuple[int, int]:
    """(num_ctx, system_prompt_chars) of MAIN_MODEL, read once from Ollama /api/show."""
    if _budget_cache:
        return _budget_cache["num_ctx"], _budget_cache["system_chars"]
    num_ctx, system_chars = FALLBACK_NUM_CTX, FALLBACK_SYSTEM_CHARS
    try:
        r = await client.post("/api/show", json={"model": MAIN_MODEL})
        r.raise_for_status()
        d = r.json()
        m = re.search(r"num_ctx\s+(\d+)", d.get("parameters", "") or "")
        if m:
            num_ctx = int(m.group(1))
        system = d.get("system")
        if not system:
            m = re.search(r'SYSTEM\s+"""(.*?)"""', d.get("modelfile", "") or "", re.S)
            system = m.group(1) if m else None
        if system:
            system_chars = len(system)
        _budget_cache.update(num_ctx=num_ctx, system_chars=system_chars)
        log.info("context budget: num_ctx=%d, system prompt=%d chars", num_ctx, system_chars)
    except Exception as e:
        log.warning("could not read model settings from Ollama (%s); assuming num_ctx=%d", e, num_ctx)
    return num_ctx, system_chars


def history_budget(num_ctx: int, system_chars: int, block_chars: int) -> int:
    room_tokens = num_ctx - system_chars / CHARS_PER_TOKEN - REPLY_RESERVE_TOKENS
    return max(MIN_HISTORY_CHARS, int(room_tokens * CHARS_PER_TOKEN) - block_chars)


def build_model_messages(messages: list[dict], mode: str, hits: list[dict],
                         num_ctx: int, system_chars: int) -> tuple[list[dict], int]:
    """Messages actually sent to the model: defused, trimmed to fit, with the application
    block on the latest user turn. The stored chat is never modified."""
    clean = []
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue                                   # the UI sends none; ignore anything else
        c = _text(m.get("content", ""))
        clean.append({"role": m["role"], "content": defuse_markers(c) if m["role"] == "user" else c})
    if not clean:
        return clean, 0
    block = app_block(mode, hits) if (mode != DEFAULT_MODE or hits) else ""
    kept, dropped = trim_history(clean, history_budget(num_ctx, system_chars, len(block)))
    if block and kept and kept[-1]["role"] == "user":
        kept[-1] = {"role": "user", "content": block + kept[-1]["content"]}
    return kept, dropped


def openai_response(text: str, model: str, stream: bool, sources: list | None = None,
                    mode: str | None = None):
    """Wrap text in OpenAI chat-completion format (streamed or not)."""
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    now = int(time.time())
    if not stream:
        body = {
            "id": rid, "object": "chat.completion", "created": now, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        }
        if sources:
            body["sources"] = sources
        if mode:
            body["mode"] = mode
        return JSONResponse(body)

    async def gen():
        chunk = {"id": rid, "object": "chat.completion.chunk", "created": now,
                 "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                              "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        done = {"id": rid, "object": "chat.completion.chunk", "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- routes
@app.get("/")
async def index():
    """Serve the chat UI."""
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- chat history
_ID_OK = re.compile(r"^[A-Za-z0-9_-]{4,40}$")


def _chat_path(chat_id: str) -> Path | None:
    if not _ID_OK.match(chat_id):
        return None
    return CHATS_DIR / f"{chat_id}{CHAT_EXT}"


@app.get("/api/chats")
async def list_chats():
    """Newest first. Returns id, title, created, updated only (no messages)."""
    items = []
    for f in CHATS_DIR.glob(f"*{CHAT_EXT}"):
        try:
            d = read_chat_file(f)
            items.append({"id": d["id"], "title": d.get("title", "Untitled"),
                          "created": d.get("created", 0), "updated": d.get("updated", 0)})
        except Exception as e:
            log.warning("skipping unreadable chat file %s: %s", f.name, e)
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    p = _chat_path(chat_id)
    if p is None or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return read_chat_file(p)


@app.put("/api/chats/{chat_id}")
async def save_chat(chat_id: str, request: Request):
    p = _chat_path(chat_id)
    if p is None:
        return JSONResponse({"error": "bad id"}, status_code=400)
    d = await request.json()
    d["id"] = chat_id
    d["updated"] = int(time.time() * 1000)
    d.setdefault("created", d["updated"])
    write_chat_file(p, d)        # chats are kept until the user deletes them
    return {"ok": True, "updated": d["updated"]}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    p = _chat_path(chat_id)
    if p is None or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    p.unlink()
    return {"ok": True}


@app.get("/api/modes")
async def list_modes():
    """Learning modes for the UI: id, label, description, placeholder, chips."""
    return {"default": DEFAULT_MODE,
            "modes": [{k: m.get(k) for k in ("id", "label", "description", "placeholder", "chips")}
                      for m in MODES.values()]}


@app.get("/v1/models")
async def list_models():
    return {"object": "list",
            "data": [{"id": MAIN_MODEL, "object": "model", "owned_by": "cvex"}]}


@app.get("/health")
async def health():
    try:
        r = await client.get("/api/tags")
        return {"proxy": "ok", "ollama": "ok" if r.status_code == 200 else "down",
                "rag_chunks": RAG.count if (RAG is not None and RAG.ready) else 0}
    except Exception:
        return {"proxy": "ok", "ollama": "down"}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    user_text = last_user_message(messages)
    req_id = uuid.uuid4().hex[:8]
    mode = body.get("mode") or DEFAULT_MODE
    if mode not in MODES:
        log.warning("[%s] unknown mode %r, using %s", req_id, mode, DEFAULT_MODE)
        mode = DEFAULT_MODE

    # 1. regex on input
    hit = regex_check(user_text, INPUT_RULES)
    if hit:
        log.warning("[%s] BLOCK input regex=%s", req_id, hit)
        return openai_response(POLICY["refusal_message"], MAIN_MODEL, stream)

    # 2. ShieldGemma on input
    t0 = time.time()
    if SG.get("check_input", True) and await shieldgemma_flags(user_text, "input"):
        log.warning("[%s] BLOCK input shieldgemma", req_id)
        return openai_response(POLICY["refusal_message"], MAIN_MODEL, stream)

    t_in = time.time() - t0

    # 2b. retrieve documents, apply the learning mode, fit the context window
    sources, hits = [], []
    if RAG is not None and RAG.ready:
        t0 = time.time()
        try:
            hits = await RAG.search(retrieval_query(messages, mode))
        except Exception as e:
            log.error("[%s] rag error: %s", req_id, e)
            hits = []
        t_rag = time.time() - t0
        if hits:
            sources = [{"n": i + 1, "title": h["title"], "page": h["page"],
                        "heading": h["heading"]} for i, h in enumerate(hits)]
            log.info("[%s] RAG %d sources in %.1fs: %s", req_id, len(hits), t_rag,
                     "; ".join(f"{h['title']} p.{h['page']}" for h in hits))
    num_ctx, system_chars = await model_budget()
    model_messages, dropped = build_model_messages(messages, mode, hits, num_ctx, system_chars)
    if dropped:
        log.info("[%s] trimmed %d older message(s) to fit num_ctx=%d", req_id, dropped, num_ctx)

    # 3. forward to main model (non-streaming so the output can be checked)
    t0 = time.time()
    payload = {"model": MAIN_MODEL, "messages": model_messages, "stream": False,
               "keep_alive": KEEP_ALIVE, "think": THINK}
    try:
        r = await client.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        reply = (data.get("message") or {}).get("content", "") or ""
        if not reply.strip():
            # log enough to diagnose: done_reason, token counts, whether thinking consumed the output
            msg = data.get("message") or {}
            log.error("[%s] EMPTY reply from model: done_reason=%s eval_count=%s prompt_eval=%s "
                      "thinking_len=%d keys=%s", req_id, data.get("done_reason"), data.get("eval_count"),
                      data.get("prompt_eval_count"), len(msg.get("thinking") or ""), list(data.keys()))
            reply = ("I wasn't able to produce an answer to that. Please try rephrasing, "
                     "or start a new chat if this keeps happening.")
    except httpx.HTTPStatusError as e:
        log.error("[%s] ollama error %s: %s", req_id, e.response.status_code, e.response.text[:500])
        return JSONResponse({"error": "model backend unavailable"}, status_code=502)
    except Exception as e:
        log.error("[%s] ollama error: %s", req_id, e)
        return JSONResponse({"error": "model backend unavailable"}, status_code=502)

    t_gen = time.time() - t0

    # 4. ShieldGemma on output
    t0 = time.time()
    if SG.get("check_output", True) and await shieldgemma_flags(reply, "output"):
        log.warning("[%s] BLOCK output shieldgemma", req_id)
        return openai_response(POLICY["output_refusal_message"], MAIN_MODEL, stream)

    # 5. regex on output
    hit = regex_check(reply, OUTPUT_RULES)
    if hit:
        log.warning("[%s] BLOCK output regex=%s", req_id, hit)
        return openai_response(POLICY["output_refusal_message"], MAIN_MODEL, stream)

    t_out = time.time() - t0
    log.info("[%s] ALLOW mode=%s len_in=%d len_out=%d | shield_in=%.1fs gen=%.1fs shield_out=%.1fs",
             req_id, mode, len(user_text), len(reply), t_in, t_gen, t_out)
    return openai_response(reply, MAIN_MODEL, stream, sources=sources, mode=mode)


if __name__ == "__main__":
    import uvicorn
    log.info("chat history dir: %s (encrypted: %s)", CHATS_DIR, ENCRYPTED)
    log.info("rag: %s", f"{RAG.count} chunks indexed" if (RAG is not None and RAG.ready) else "no index (run rag/ingest.py)")
    if not ENCRYPTED:
        log.warning("not on Windows: chat files are NOT encrypted at rest")
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT)
