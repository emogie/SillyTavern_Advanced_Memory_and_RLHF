"""
ChromaDB Vector Store - Open source vector database for RAG.
Handles storage, retrieval, and management of embedded text chunks.
"""
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
logger = logging.getLogger("ChromaStore")

class ChromaStore:
    """
    Manages a ChromaDB vector database for middle-term memory.
    Uses sentence-transformers for open-source embeddings.
    """
    def __init__(self, config: dict):
        self.config = config
        self.db_path = Path(config.get('vector_db_path', 'data/vector_db'))
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = config.get('collection_name', 'sillytavern_memory')
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        # Initialize embedding model
        model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized: {self.collection.count()} documents")
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None,
                  ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add texts to the vector store with automatic chunking."""
        all_chunks = []
        all_metas = []
        all_ids = []
        for i, text in enumerate(texts):
            chunks = self._chunk_text(text)
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            for j, chunk in enumerate(chunks):
                chunk_id = ids[i] if ids and i < len(ids) else None
                if not chunk_id:
                    chunk_id = hashlib.md5(
                        f"{text[:100]}_{i}_{j}".encode()
                    ).hexdigest()
                else:
                    chunk_id = f"{chunk_id}_chunk{j}"
                all_chunks.append(chunk)
                chunk_meta = {**meta, "chunk_index": j, "total_chunks": len(chunks)}
                # ChromaDB requires string values in metadata
                all_metas.append({
                    k: str(v) for k, v in chunk_meta.items()
                })
                all_ids.append(chunk_id)
        if not all_chunks:
            return {"added": 0}
        # Generate embeddings
        embeddings = self.embedding_model.encode(all_chunks).tolist()
        # Add to ChromaDB (in batches to avoid size limits)
        batch_size = 500
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            self.collection.upsert(
                documents=all_chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=all_metas[start:end],
                ids=all_ids[start:end]
            )
        logger.info(f"Added {len(all_chunks)} chunks from {len(texts)} texts")
        return {"added": len(all_chunks), "source_texts": len(texts)}
    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        output = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                output.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        return output
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents for training data export."""
        count = self.collection.count()
        if count == 0:
            return []
        # Fetch in batches
        all_docs = []
        batch_size = 1000
        offset = 0
        while offset < count:
            results = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            for i, doc in enumerate(results['documents']):
                meta = results['metadatas'][i] if results['metadatas'] else {}
                all_docs.append({
                    "id": results['ids'][i],
                    "text": doc,
                    "metadata": meta
                })
            offset += batch_size
        return all_docs
    def get_total_size_mb(self) -> float:
        """Estimate total data size in the vector database."""
        docs = self.get_all_documents()
        total_bytes = sum(len(d['text'].encode('utf-8')) for d in docs)
        # Add ~30% overhead for metadata and embeddings
        return (total_bytes * 1.3) / (1024 * 1024)
    def get_document_count(self) -> int:
        """Get total number of chunks in the store."""
        return self.collection.count()
    def move_to_archive(self, archive_path: str):
        """Move current data to archive after training."""
        import shutil
        archive = Path(archive_path)
        archive.mkdir(parents=True, exist_ok=True)
        # Export all data
        docs = self.get_all_documents()
        timestamp = __import__('time').strftime('%Y%m%d_%H%M%S')
        export_file = archive / f"training_data_{timestamp}.json"
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)
        logger.info(f"Archived {len(docs)} documents to {export_file}")
        # Clear the collection but keep it
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return {"archived": len(docs), "archive_file": str(export_file)}
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                for sep in ['. ', '! ', '? ', '\n', '; ', ', ']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks
    def close(self):
        """Clean shutdown."""
        logger.info("ChromaDB store closed.")

    def delete_by_ids(self, ids: list) -> int:
        """Delete documents by their IDs."""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0

    def clear(self):
        """Delete all documents from the collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data and all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")