import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag.vector_store import VectorStore
from src.rag.hybrid_retriever import BM25Retriever
from src.ollama import OllamaClient
from src.config import get_settings

get_settings.cache_clear()
s = get_settings()

vs = VectorStore(s.VECTOR_DB_PATH)
docs = vs.get_all_documents()
bm25 = BM25Retriever()
bm25.build_index(docs)

q = "who is govind"
hits = bm25.retrieve(q, top_k=5)
print("BM25 top hits for:", q)
for i, h in enumerate(hits, 1):
    meta = h.get("metadata", {})
    print(f"{i}. score={h.get('bm25_score'):.3f} file={meta.get('source_file')}")
    print("   ", h["text"][:180].replace("\n", " "))

context = "\n\n".join(
    f"[Source {i} | {h['metadata'].get('source_file')}]\n{h['text']}"
    for i, h in enumerate(hits[:3], 1)
)
prompt = f"""You are G-RAG, a document Q&A assistant.
Use the context below to answer the question.
If the context contains the person's name, role, skills, experience, or location, use those facts.
Do NOT refuse when the answer is present in the context.

Context:
{context}

Question: {q}

Answer:"""

client = OllamaClient(host=s.OLLAMA_HOST, model=s.OLLAMA_MODEL)
print("\nMODEL", s.OLLAMA_MODEL)
print("\nANSWER:")
print(client.generate(prompt, temperature=0.2, num_predict=200))
