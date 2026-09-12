from src.config import get_settings

get_settings.cache_clear()
s = get_settings()
print("model", s.OLLAMA_MODEL)
print("eval", s.RUN_EVALUATION)

from src.rag.retriever import AdvancedRetriever

r = AdvancedRetriever(
    persist_directory=s.VECTOR_DB_PATH,
    ollama_host=s.OLLAMA_HOST,
    ollama_model=s.OLLAMA_MODEL,
    top_k_retrieval=s.TOP_K_RETRIEVAL,
    top_k_final=s.TOP_K_FINAL,
    compression_threshold=s.COMPRESSION_THRESHOLD,
    memory_turns=s.MEMORY_TURNS,
    cross_encoder_model=s.CROSS_ENCODER_MODEL,
    embed_model_name=s.EMBED_MODEL,
    run_evaluation=False,
)
print("chunks in store", r.vector_store.count())
docs = r.vector_store.get_all_documents()
print("--- texts containing govind ---")
for d in docs:
    t = d["text"].lower()
    if "govind" in t:
        print("FILE", d["metadata"].get("source_file"), "PAGE", d["metadata"].get("page"))
        print(d["text"][:500].replace("\n", " "))
        print("---")

q = "who is govind"
rewritten = r.rewriter.rewrite(q)
print("\nREWRITTEN:", rewritten)
cands = r.hybrid.retrieve(rewritten)
print("hybrid candidates", len(cands))
for i, c in enumerate(cands[:5], 1):
    print(
        f"\nCAND {i} rrf={c.get('rrf_score')} file={c.get('metadata', {}).get('source_file')} page={c.get('metadata', {}).get('page')}"
    )
    print(c["text"][:400].replace("\n", " "))

reranked = r.reranker.rerank(rewritten, cands, top_n=s.TOP_K_FINAL)
compressed = r.compressor.compress(rewritten, reranked)
print("\n=== COMPRESSED CONTEXT ===")
for i, c in enumerate(compressed, 1):
    print(f"\nSRC {i} file={c.get('metadata', {}).get('source_file')}")
    print("COMP:", c["text"][:450].replace("\n", " "))

result = r.query(q)
print("\n=== ANSWER ===")
print(result["answer"])
print("sources", len(result["sources"]))
for ssrc in result["sources"]:
    print(
        "-",
        ssrc.get("source_file"),
        "p",
        ssrc.get("page"),
        "snip:",
        (ssrc.get("snippet") or "")[:150].replace("\n", " "),
    )
