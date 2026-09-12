import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag.vector_store import VectorStore

vs = VectorStore("vector_db")
docs = vs.get_all_documents()
print("count", len(docs))

files = {}
for d in docs:
    f = d.get("metadata", {}).get("source_file", "?")
    files[f] = files.get(f, 0) + 1
print("files:")
for f, n in files.items():
    print(f"  {n:4d}  {f}")

print("\n--- chunks containing 'govind' ---")
hits = 0
for d in docs:
    text = d.get("text", "")
    if "govind" in text.lower():
        hits += 1
        meta = d.get("metadata", {})
        print("FILE", meta.get("source_file"), "PAGE", meta.get("page"))
        print(text[:600].replace("\n", " "))
        print("---")
print("hits", hits)

print("\n--- first 200 chars of each unique filename first chunk ---")
seen = set()
for d in docs:
    f = d.get("metadata", {}).get("source_file", "?")
    if f in seen:
        continue
    seen.add(f)
    print("FILE", f)
    print(d.get("text", "")[:250].replace("\n", " "))
    print("---")
