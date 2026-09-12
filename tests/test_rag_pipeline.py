import sys
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import pytest

# --- Mock sentence_transformers globally before any other imports ---
class MockSentenceTransformer:
    def __init__(self, model_name=None, *args, **kwargs):
        self.model_name = model_name

    def encode(self, sentences, *args, **kwargs):
        if isinstance(sentences, str):
            return (np.ones(128, dtype=np.float32) * 0.1).tolist()
        # Returns shape (len(sentences), 128)
        return np.ones((len(sentences), 128), dtype=np.float32) * 0.1

class MockCrossEncoder:
    def __init__(self, model_name=None, *args, **kwargs):
        self.model_name = model_name

    def predict(self, pairs, *args, **kwargs):
        scores = []
        for q, p in pairs:
            if "B" in p:
                scores.append(0.9)
            elif "A" in p:
                scores.append(0.5)
            else:
                scores.append(0.1)
        return scores

mock_st = MagicMock()
mock_st.SentenceTransformer = MockSentenceTransformer
mock_st.CrossEncoder = MockCrossEncoder
sys.modules['sentence_transformers'] = mock_st

# --- Imports from project ---
from src.rag.vector_store import VectorStore
from src.rag.retriever import AdvancedRetriever
from src.rag.reranker import CrossEncoderReranker

# --- Mock Response class for requests.post ---
class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        pass

@pytest.fixture(autouse=True)
def mock_ollama_api():
    def mock_post(url, json, timeout=None):
        prompt = json.get("prompt", "")
        if "relevance" in prompt.lower() or "faithfulness" in prompt.lower():
            return MockResponse({"response": "0.95"})
        elif "rewritten query" in prompt.lower() or "rewrite" in prompt.lower():
            return MockResponse({"response": "mocked search query"})
        else:
            return MockResponse({"response": "This is a mocked RAG response based on the context."})

    with patch("requests.post", side_effect=mock_post) as mock:
        yield mock

# --- Tests ---

def test_chromadb_connects():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        vs = VectorStore(persist_directory=tmp_dir)
        try:
            assert vs.client is not None
            assert vs.collection is not None
            assert vs.collection.name == "research_papers"
        finally:
            vs.client._system.stop()

def test_retrieval_returns_chunks():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        dummy_embed_fn = lambda text: [0.1] * 128
        retriever = AdvancedRetriever(
            persist_directory=tmp_dir,
            embed_fn=dummy_embed_fn,
            run_evaluation=False,
        )
        try:
            test_docs = [
                {"text": "Sample document about artificial intelligence and modern deep learning.", "metadata": {"chunk_hash": "hash1", "source_file": "doc1.pdf", "page": 1}},
                {"text": "Another paper on transformer architecture and attention mechanisms.", "metadata": {"chunk_hash": "hash2", "source_file": "doc2.pdf", "page": 2}},
                {"text": "Information regarding vector stores like ChromaDB and dense embeddings search.", "metadata": {"chunk_hash": "hash3", "source_file": "doc3.pdf", "page": 1}},
                {"text": "RAG systems combine retrieval models with large language models.", "metadata": {"chunk_hash": "hash4", "source_file": "doc4.pdf", "page": 1}},
            ]
            retriever.ingest(test_docs)
            
            chunks = retriever.hybrid.retrieve("artificial intelligence and RAG systems")
            assert len(chunks) >= 3
        finally:
            retriever.vector_store.client._system.stop()

def test_reranker_works():
    reranker = CrossEncoderReranker()
    chunks = [
        {"text": "passage A"},
        {"text": "passage B"},
        {"text": "passage C"},
    ]
    sorted_chunks = reranker.rerank("query", chunks, top_n=3)
    assert len(sorted_chunks) == 3
    # B should be first (score 0.9), then A (score 0.5), then C (score 0.1)
    assert sorted_chunks[0]["text"] == "passage B"
    assert sorted_chunks[1]["text"] == "passage A"
    assert sorted_chunks[2]["text"] == "passage C"
    assert sorted_chunks[0]["cross_encoder_score"] == 0.9

def test_end_to_end_query():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        dummy_embed_fn = lambda text: [0.1] * 128
        retriever = AdvancedRetriever(
            persist_directory=tmp_dir,
            embed_fn=dummy_embed_fn,
            run_evaluation=True,
        )
        try:
            test_docs = [
                {"text": "Sample document about artificial intelligence and modern deep learning.", "metadata": {"chunk_hash": "hash1", "source_file": "doc1.pdf", "page": 1}},
                {"text": "Another paper on transformer architecture and attention mechanisms.", "metadata": {"chunk_hash": "hash2", "source_file": "doc2.pdf", "page": 2}},
            ]
            retriever.ingest(test_docs)
            
            result = retriever.query("Explain artificial intelligence")
            
            assert isinstance(result, dict)
            assert "answer" in result
            assert isinstance(result["answer"], str)
            assert len(result["answer"].strip()) > 0
            assert "evaluation" in result
            assert result["evaluation"]["faithfulness"] == 0.95
        finally:
            retriever.vector_store.client._system.stop()
