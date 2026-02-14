"""
Middle-Term Memory - RAG-based retrieval for ongoing conversations.
Data persists in vector database until trained into LoRA.
"""
class MiddleTermMemory:
    """
    Wraps the vector store for middle-term memory operations.
    This is the "working memory" that accumulates until training threshold.
    """
    def __init__(self, vector_store: ChromaStore):
        self.store = vector_store
    def remember(self, text: str, metadata: dict = None) -> dict:
        return self.store.add_texts([text], [metadata] if metadata else None)
    def recall(self, query: str, k: int = 5) -> list:
        return self.store.query(query, k)
    def forget_all(self):
        """Clear all middle-term memory."""
        return self.store.move_to_archive("data/trained_archive")
