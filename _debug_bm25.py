import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import BM25Retriever, _tokenize

vs = VectorStore("vector_db")
docs = vs.get_all_documents()
bm25 = BM25Retriever()
bm25.build_index(docs)

for q in ["govind", "who is govind", "Govind Krishnan", "govind krishnan radhakrishnan"]:
    print("\nQUERY:", q, "tokens:", _tokenize(q))
    hits = bm25.retrieve(q, top_k=5)
    for i, h in enumerate(hits, 1):
        f = h["metadata"].get("source_file")
        print(f"  {i}. {h.get('bm25_score'):.3f}  {f}")
        # show if govind in text
        print("     has_govind=", "govind" in h["text"].lower(), "snip=", h["text"][:90].replace("\n"," "))
