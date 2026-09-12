import time

def log_time(label):
    print(f"[{time.time() - start_time:.2f}s] {label}")

start_time = time.time()
log_time("Start script")

import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
log_time("Sys path modified")

import numpy as np
log_time("numpy imported")
import tempfile
log_time("tempfile imported")
from unittest.mock import MagicMock, patch
log_time("unittest/mock imported")

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
        return [0.5] * len(pairs)

mock_st = MagicMock()
mock_st.SentenceTransformer = MockSentenceTransformer
mock_st.CrossEncoder = MockCrossEncoder
sys.modules['sentence_transformers'] = mock_st
log_time("sentence_transformers mocked")

from src.rag.vector_store import VectorStore
log_time("VectorStore imported")
from src.rag.retriever import AdvancedRetriever
log_time("AdvancedRetriever imported")
from src.rag.reranker import CrossEncoderReranker
log_time("CrossEncoderReranker imported")

print("Running test_chromadb_connects...")
with tempfile.TemporaryDirectory() as tmp_dir:
    log_time("Temp directory created")
    vs = VectorStore(persist_directory=tmp_dir)
    log_time("VectorStore initialized")
    assert vs.client is not None
    assert vs.collection is not None
    assert vs.collection.name == "research_papers"
    log_time("Asserts passed")
    vs.client._system.stop()
    log_time("vs client stopped")
print("Done!")
