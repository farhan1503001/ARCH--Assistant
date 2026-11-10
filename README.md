<div align="center">

# 🏛️ ARCH Assistant

**A private, local AI assistant for the Center for Advanced Research in Computational Humanities, SIU Carbondale**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-000000)
![Model](https://img.shields.io/badge/Model-Gemma%204%2012B-4285F4)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white)
![Privacy](https://img.shields.io/badge/Data-stays%20on%20your%20machine-2EA44F)

[✨ Features](#features) · [📋 Requirements](#requirements) · [🚀 Setup](#setup) · [▶️ Run](#run) · [⚙️ Configuration](#configuration) · [🧪 Tests](#tests) · [🗂️ Layout](#layout)

</div>

ARCH was formerly named CVEX (Center for Virtual Expression). Everything runs on the user's own computer, so no chat text or documents are sent to an outside service.

---

<a id="features"></a>
## ✨ Features

| | Feature | What it does |
|---|---|---|
| 🧠 | **Local model** | Gemma 4 12B served by [Ollama](https://ollama.com), with a custom system prompt |
| 🛡️ | **Guardrails** | Every message and reply is checked by regex rules and ShieldGemma 2B, with an exemption for scholarly discussion |
| 💬 | **Chat interface** | ChatGPT-style web page with a history sidebar and math rendering |
| 🔐 | **Protected history** | One encrypted file per chat (Windows DPAPI), readable only by the same Windows user |
| 🎓 | **Learning modes** | Explain, Tutor, Quiz and Summarize |
| 📚 | **Document search (RAG)** | Answers can draw on your books, papers and web pages, with sources shown under each answer |

<details>
<summary>🎓 <b>What each learning mode does</b></summary>

| Mode | Behaviour |
|---|---|
| **Explain** | Clear, complete explanations, concept first |
| **Tutor** | Guides with questions and hints instead of giving the answer |
| **Quiz** | One question at a time, graded, with a running score |
| **Summarize** | Structured summary of a topic, paper or pasted text |

</details>

<details>
<summary>🔎 <b>How document search works</b></summary>

Documents are split into small chunks for matching and larger passages for context. A question is matched by vector search and keyword search, the results are merged, and a reranker (`BAAI/bge-reranker-base`) picks the best passages. Greetings and other trivial messages skip the search.

</details>

---

<a id="requirements"></a>
## 📋 Requirements

- 🪟 Windows 10/11
- 🐍 Python 3.10 or newer (64-bit)
- 🦙 [Ollama](https://ollama.com)
- 🎮 NVIDIA GPU with 12 GB or more of VRAM (developed on an RTX 4070 Ti)

---

<a id="setup"></a>
## 🚀 Setup

Run these in the project folder.

<details open>
<summary><b>1️⃣ Pull the models</b></summary>

```
ollama pull gemma4:12b
ollama pull shieldgemma:2b
ollama pull embeddinggemma
ollama create cvex-ai -f Modelfile
```

Run the last command again whenever `Modelfile` changes.

</details>

<details>
<summary><b>2️⃣ Create a virtual environment and install packages</b></summary>

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then install the GPU build of PyTorch (the default install is CPU-only and makes the reranker slow). The steps are in the comments of `requirements.txt`. Confirm with:

```
python -c "import torch; print(torch.cuda.is_available())"
```

It should print `True`.

</details>

<details>
<summary><b>3️⃣ Download the reranker</b></summary>

```
python rag\download_reranker.py
```

</details>

<details>
<summary><b>4️⃣ Add documents (optional)</b></summary>

Put PDF, DOCX, Markdown, MDX, TXT or HTML files in `docs\`. List web pages to crawl in `docs\urls.txt`. Then build the index:

```
python rag\ingest.py
```

Other options: `--rebuild` starts the index from scratch, and `--stats` shows what is indexed.

</details>

---

<a id="run"></a>
## ▶️ Run

Make sure Ollama is running, then:

```
python guardrail_proxy.py
```

Open **http://localhost:8080** in a browser.

---

<a id="configuration"></a>
## ⚙️ Configuration

| File | What it controls |
|---|---|
| 📝 `Modelfile` | System prompt, context size, temperature, answer length |
| 🛡️ `policy.json` | Refusal messages, regex rules, ShieldGemma policies |
| 🎓 `modes.json` | Names, descriptions and example prompts for the learning modes |
| 🌐 `docs/urls.txt` | Web pages and sites to crawl into the index |

Restart the proxy after editing any of these. Chats and the document index are stored in `%LOCALAPPDATA%\ARCHAssistant`.

---

<a id="tests"></a>
## 🧪 Tests

```
python tests\test_fixes.py
python test_guardrail.py
python test_modes_live.py
```

`tests\test_fixes.py` needs no model. The other two need the proxy and Ollama running.

---

<a id="layout"></a>
## 🗂️ Layout

<details>
<summary><b>Show project structure</b></summary>

```
guardrail_proxy.py   web server, guardrails, chat storage, model calls
Modelfile            system prompt and model settings
policy.json          guardrail rules
modes.json           learning mode text for the UI
static/              web interface and logos
rag/                 document ingestion, retrieval, reranker download
tests/               automated checks
docs/                your documents (not tracked in git)
```

</details>

> 📌 The books and papers in `docs/` and the downloaded reranker in `models/` are not stored in this repository. Only `docs/urls.txt` is tracked.
