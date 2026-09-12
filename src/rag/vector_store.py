import chromadb
from typing import List, Dict


class VectorStore:
    """
    Thin wrapper around ChromaDB using the modern PersistentClient API.

    New helpers added for Advanced RAG:
      - get_all_documents()  : returns every stored chunk (needed for BM25 corpus)
      - distance_search()    : similarity search that also returns raw distances
                               so HybridRetriever can do Reciprocal Rank Fusion.
    """

    def __init__(self, persist_directory: str):
        # PersistentClient replaces the deprecated Client(Settings(...)) in v0.4+
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("research_papers")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        ids = [doc["metadata"]["chunk_hash"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read — basic
    # ------------------------------------------------------------------

    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Return top-k most similar chunks (no distances exposed)."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    # ------------------------------------------------------------------
    # Read — advanced helpers
    # ------------------------------------------------------------------

    def distance_search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """
        Same as similarity_search but includes the raw L2 distance so that
        HybridRetriever can compute Reciprocal Rank Fusion ranks.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_all_documents(self) -> List[Dict]:
        """
        Return every chunk stored in the collection.
        Used by BM25Retriever to rebuild its corpus index.
        """
        results = self.collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            return []
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents by recreating an empty collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)
