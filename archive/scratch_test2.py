import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tempfile
from unittest.mock import MagicMock, patch

# --- Mock sentence_transformers globally ---
class MockSentenceTransformer:
    def __init__(self, model_name=None, *args, **kwargs):
        self.model_name = model_name

    def encode(self, sentences, *args, **kwargs):
        if isinstance(sentences, str):
            return (np.ones(128, dtype=np.float32) * 0.1).tolist()
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

from src.rag.vector_store import VectorStore
from src.rag.retriever import AdvancedRetriever
from src.rag.reranker import CrossEncoderReranker

print("Imports done.")

# Run test_chromadb_connects
print("Running test_chromadb_connects...")
with tempfile.TemporaryDirectory() as tmp_dir:
    vs = VectorStore(persist_directory=tmp_dir)
    try:
        assert vs.client is not None
        assert vs.collection is not None
        assert vs.collection.name == "research_papers"
        print("Asserts passed.")
    finally:
        print("Stopping vs client...")
        vs.client._system.stop()
        print("vs client stopped.")
print("test_chromadb_connects finished.")
