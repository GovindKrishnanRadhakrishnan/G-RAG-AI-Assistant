try:
    from prometheus_client import Histogram

    # Histogram for tracking RAG retrieval and compression time
    RAG_RETRIEVAL_DURATION = Histogram(
        "rag_retrieval_duration_seconds",
        "Time spent retrieving and compressing context from vector database",
        buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, float("inf"))
    )

    # Histogram for tracking LLM generation time via Ollama
    LLM_INFERENCE_DURATION = Histogram(
        "llm_inference_duration_seconds",
        "Time spent on LLM text generation via local Ollama",
        buckets=(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, float("inf"))
    )

except ImportError:
    # prometheus_client is an optional monitoring dependency.
    # When not installed, provide no-op stubs so the pipeline still runs.
    class _NoOpHistogram:
        """No-op stand-in for prometheus_client.Histogram."""
        def observe(self, *args, **kwargs):
            pass

    RAG_RETRIEVAL_DURATION = _NoOpHistogram()
    LLM_INFERENCE_DURATION = _NoOpHistogram()

