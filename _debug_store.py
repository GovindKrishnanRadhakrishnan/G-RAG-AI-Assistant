from src.rag.vector_store import VectorStore

vs = VectorStore("vector_db")
print("count", vs.count())
docs = vs.get_all_documents()
print("docs", len(docs))
for d in docs:
    meta = d.get("metadata", {})
    text = d.get("text", "")
    print("=" * 60)
    print("file=", meta.get("source_file"), "page=", meta.get("page"), "len=", len(text))
    print(text[:800])
    print()
