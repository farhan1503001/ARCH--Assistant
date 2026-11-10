"""
Download the reranker model once (needs internet) and save it locally so the proxy
runs offline afterwards. Saves to models/reranker next to the project folder.

    python rag/download_reranker.py
"""
import os
from pathlib import Path

NAME = os.environ.get("ARCH_RERANKER_NAME", "BAAI/bge-reranker-base")
DEST = Path(os.environ.get("ARCH_RERANKER_DIR",
                           str(Path(__file__).resolve().parent.parent / "models" / "reranker")))

from sentence_transformers import CrossEncoder

print(f"Downloading {NAME} ...")
m = CrossEncoder(NAME, device="cpu")
DEST.mkdir(parents=True, exist_ok=True)
m.save_pretrained(str(DEST))
print(f"Saved to {DEST}")

# sanity check: a relevant pair should score clearly higher than an irrelevant one
pairs = [("what is a graph neural network",
          "Graph neural networks operate on graph-structured data by passing messages between neighbouring nodes."),
         ("what is a graph neural network",
          "To create an HTML file with Python, write a string containing <html><body>Hello World</body></html>.")]
rel, irr = [float(x) for x in m.predict(pairs)]
print(f"Check: relevant={rel:.3f}  irrelevant={irr:.3f}  ({'OK' if rel > irr else 'UNEXPECTED - tell Claude'})")
print("Use these two numbers to sanity-check ARCH_RAG_MIN_RERANK: it should sit between them.")
