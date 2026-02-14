"""
Memory Manager - Orchestrates middle-term (RAG) and long-term (LoRA) memory.
Stores chat content with embeddings for retrieval-augmented generation.
Provides browsing, deletion, and cleanup capabilities.
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional

from modules.vectordb.chroma_store import ChromaStore

logger = logging.getLogger("MemoryManager")


class MemoryManager:
    """
    Manages the memory lifecycle:
    - Middle-term: Stored in vector DB for RAG retrieval
    - Long-term: Trained into LoRA adapter
    - Handles data collection threshold for training readiness
    - Provides browsing, deletion, and cleanup of stored data
    """

    def __init__(self, config: dict, vector_store: ChromaStore):
        self.config = config
        self.vector_store = vector_store
        self.min_training_size_mb = config.get('min_training_size_mb', 50)

    def store(self, character: str, messages: List[Dict], auto_stored: bool = False) -> Dict:
        """Store chat messages into the vector database."""
        texts = []
        metadatas = []

        for msg in messages:
            content = msg.get('content', '').strip()
            if not content:
                continue

            role = msg.get('role', 'unknown')
            name = msg.get('name', role)
            images = msg.get('images', [])

            # Format message for storage
            formatted = f"[{name}] ({role}): {content}"
            if images:
                formatted += f" [has {len(images)} image(s)]"

            texts.append(formatted)
            metadatas.append({
                "character": character,
                "role": role,
                "name": name,
                "has_images": str(len(images) > 0),
                "auto_stored": str(auto_stored),
                "timestamp": str(msg.get('timestamp', int(time.time() * 1000)))
            })

        if not texts:
            return {"stored": 0, "message": "No content to store"}

        result = self.vector_store.add_texts(texts, metadatas)
        status = self.get_status()

        return {
            "stored": result['added'],
            "total_documents": status['document_count'],
            "total_size_mb": status['total_size_mb'],
            "training_ready": status['training_ready']
        }

    def query(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """
        Query middle-term memory for relevant context.

        Args:
            query: The search query text
            k: Maximum number of results to return
            min_score: Minimum relevance score (0.0 to 1.0).
                       Results below this threshold are filtered out.

        Returns:
            List of matching documents with text, score, and metadata.
        """
        results = self.vector_store.query(query, k)

        # Filter by minimum relevance score if specified
        if min_score > 0:
            results = [r for r in results if r.get('score', 0) >= min_score]

        return results

    def browse(self, character: str = None, offset: int = 0, limit: int = 50,
               sort_order: str = "newest") -> Dict:
        """
        Browse stored memories with pagination and filtering.

        Args:
            character: Filter by character name (None = all)
            offset: Pagination offset
            limit: Number of results per page
            sort_order: "newest" or "oldest"

        Returns:
            Dict with documents, total count, and pagination info.
        """
        all_docs = self.vector_store.get_all_documents()

        # Filter by character if specified
        if character:
            all_docs = [
                d for d in all_docs
                if d.get('metadata', {}).get('character', '') == character
            ]

        # Sort by timestamp
        def get_timestamp(doc):
            try:
                return int(doc.get('metadata', {}).get('timestamp', '0'))
            except (ValueError, TypeError):
                return 0

        reverse = sort_order == "newest"
        all_docs.sort(key=get_timestamp, reverse=reverse)

        total = len(all_docs)

        # Paginate
        paginated = all_docs[offset:offset + limit]

        # Format for display
        formatted_docs = []
        for doc in paginated:
            metadata = doc.get('metadata', {})
            ts = get_timestamp(doc)
            formatted_docs.append({
                "id": doc.get('id', ''),
                "text": doc.get('text', doc.get('document', '')),
                "character": metadata.get('character', 'Unknown'),
                "role": metadata.get('role', 'unknown'),
                "name": metadata.get('name', ''),
                "auto_stored": metadata.get('auto_stored', 'False') == 'True',
                "timestamp": ts,
                "timestamp_formatted": self._format_timestamp(ts),
                "text_preview": self._truncate(doc.get('text', doc.get('document', '')), 200)
            })

        return {
            "documents": formatted_docs,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total
        }

    def delete(self, doc_ids: List[str] = None, character: str = None,
               before_timestamp: int = None, auto_stored_only: bool = False) -> Dict:
        """
        Delete memories by various criteria.

        Args:
            doc_ids: Specific document IDs to delete
            character: Delete all documents for this character
            before_timestamp: Delete documents before this timestamp
            auto_stored_only: Only delete auto-stored documents (not manually stored)

        Returns:
            Dict with deletion results.
        """
        deleted_count = 0

        if doc_ids:
            # Delete specific documents by ID
            deleted_count = self.vector_store.delete_by_ids(doc_ids)
            logger.info(f"Deleted {deleted_count} documents by ID")

        elif character or before_timestamp or auto_stored_only:
            # Get all documents and filter
            all_docs = self.vector_store.get_all_documents()
            ids_to_delete = []

            for doc in all_docs:
                metadata = doc.get('metadata', {})
                should_delete = True

                if character:
                    if metadata.get('character', '') != character:
                        should_delete = False

                if before_timestamp and should_delete:
                    try:
                        doc_ts = int(metadata.get('timestamp', '0'))
                        if doc_ts >= before_timestamp:
                            should_delete = False
                    except (ValueError, TypeError):
                        should_delete = False

                if auto_stored_only and should_delete:
                    if metadata.get('auto_stored', 'False') != 'True':
                        should_delete = False

                if should_delete:
                    doc_id = doc.get('id', '')
                    if doc_id:
                        ids_to_delete.append(doc_id)

            if ids_to_delete:
                deleted_count = self.vector_store.delete_by_ids(ids_to_delete)
                logger.info(f"Deleted {deleted_count} documents by filter criteria")

        status = self.get_status()
        return {
            "deleted": deleted_count,
            "remaining_documents": status['document_count'],
            "remaining_size_mb": status['total_size_mb']
        }

    def clear_all(self) -> Dict:
        """Delete ALL stored memories. Use with caution."""
        count_before = self.vector_store.get_document_count()
        self.vector_store.clear()
        logger.warning(f"Cleared all memory: {count_before} documents deleted")
        return {
            "deleted": count_before,
            "message": f"All {count_before} documents have been deleted"
        }

    def get_characters(self) -> List[Dict]:
        """Get a list of all characters with document counts."""
        all_docs = self.vector_store.get_all_documents()
        char_counts = {}

        for doc in all_docs:
            char = doc.get('metadata', {}).get('character', 'Unknown')
            if char not in char_counts:
                char_counts[char] = {
                    "name": char,
                    "total": 0,
                    "auto_stored": 0,
                    "manual": 0,
                    "oldest_timestamp": float('inf'),
                    "newest_timestamp": 0
                }

            char_counts[char]["total"] += 1

            if doc.get('metadata', {}).get('auto_stored', 'False') == 'True':
                char_counts[char]["auto_stored"] += 1
            else:
                char_counts[char]["manual"] += 1

            try:
                ts = int(doc.get('metadata', {}).get('timestamp', '0'))
                if ts < char_counts[char]["oldest_timestamp"]:
                    char_counts[char]["oldest_timestamp"] = ts
                if ts > char_counts[char]["newest_timestamp"]:
                    char_counts[char]["newest_timestamp"] = ts
            except (ValueError, TypeError):
                pass

        # Format timestamps
        result = []
        for char_data in char_counts.values():
            if char_data["oldest_timestamp"] == float('inf'):
                char_data["oldest_timestamp"] = 0
            char_data["oldest_formatted"] = self._format_timestamp(char_data["oldest_timestamp"])
            char_data["newest_formatted"] = self._format_timestamp(char_data["newest_timestamp"])
            result.append(char_data)

        # Sort by total documents descending
        result.sort(key=lambda x: x["total"], reverse=True)
        return result

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent memory history for review."""
        all_docs = self.vector_store.get_all_documents()

        # Sort by timestamp descending (newest first)
        def get_timestamp(doc):
            try:
                return int(doc.get('metadata', {}).get('timestamp', '0'))
            except (ValueError, TypeError):
                return 0

        all_docs.sort(key=get_timestamp, reverse=True)

        # Limit results
        recent = all_docs[:limit]

        history = []
        for doc in recent:
            metadata = doc.get('metadata', {})
            ts = get_timestamp(doc)
            history.append({
                "id": doc.get('id', ''),
                "text_preview": self._truncate(doc.get('text', doc.get('document', '')), 300),
                "full_text": doc.get('text', doc.get('document', '')),
                "character": metadata.get('character', 'Unknown'),
                "role": metadata.get('role', 'unknown'),
                "name": metadata.get('name', ''),
                "auto_stored": metadata.get('auto_stored', 'False') == 'True',
                "timestamp": ts,
                "timestamp_formatted": self._format_timestamp(ts)
            })

        return history

    def get_status(self) -> Dict:
        """Get current memory status including training readiness."""
        size_mb = self.vector_store.get_total_size_mb()
        doc_count = self.vector_store.get_document_count()

        return {
            "document_count": doc_count,
            "total_size_mb": round(size_mb, 2),
            "min_training_size_mb": self.min_training_size_mb,
            "training_ready": size_mb >= self.min_training_size_mb,
            "progress_to_training": min(100, round(
                (size_mb / self.min_training_size_mb) * 100, 1
            ))
        }

    def get_training_data(self) -> List[Dict]:
        """Export all data from vector store for training."""
        docs = self.vector_store.get_all_documents()
        logger.info(f"Prepared {len(docs)} documents for training")
        return docs

    def archive_data(self) -> Dict:
        """Move trained data to archive."""
        archive_path = self.config.get('trained_archive_path', 'data/trained_archive')
        result = self.vector_store.move_to_archive(archive_path)
        return result

    @staticmethod
    def _format_timestamp(ts: int) -> str:
        """Format a millisecond timestamp to readable string."""
        if not ts or ts <= 0:
            return "Unknown"
        try:
            import datetime
            dt = datetime.datetime.fromtimestamp(ts / 1000)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError, OverflowError):
            return "Unknown"

    @staticmethod
    def _truncate(text: str, max_length: int = 200) -> str:
        """Truncate text with ellipsis."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."