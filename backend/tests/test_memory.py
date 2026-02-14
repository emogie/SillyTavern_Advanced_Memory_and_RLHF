"""Tests for memory management."""
import pytest
import tempfile
import os
def test_chroma_store():
    """Test ChromaDB vector store operations."""
    from modules.vectordb.chroma_store import ChromaStore
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'vector_db_path': os.path.join(tmpdir, 'test_db'),
            'collection_name': 'test_collection',
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': 256,
            'chunk_overlap': 25
        }
        store = ChromaStore(config)
        # Add texts
        result = store.add_texts(
            ["Hello world, this is a test message about memory.",
             "Another message about training LoRA models."],
            [{"role": "user"}, {"role": "assistant"}]
        )
        assert result['added'] >= 2
        # Query
        results = store.query("memory test", k=2)
        assert len(results) > 0
        assert results[0]['score'] > 0
        # Count
        assert store.get_document_count() >= 2
        # Size
        size = store.get_total_size_mb()
        assert size >= 0
        store.close()

def test_memory_manager():
    """Test memory manager store and query."""
    from modules.vectordb.chroma_store import ChromaStore
    from modules.memory.memory_manager import MemoryManager
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_config = {
            'vector_db_path': os.path.join(tmpdir, 'test_db'),
            'trained_archive_path': os.path.join(tmpdir, 'archive'),
            'collection_name': 'test_mem',
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': 256,
            'chunk_overlap': 25,
            'min_training_size_mb': 0.001
        }
        store = ChromaStore(mem_config)
        manager = MemoryManager(mem_config, store)
        result = manager.store("TestChar", [
            {"role": "user", "name": "User", "content": "Tell me about dragons"},
            {"role": "assistant", "name": "Bot", "content": "Dragons are mythical creatures"}
        ])
        assert result['stored'] > 0
        results = manager.query("dragons", 3)
        assert len(results) > 0
        status = manager.get_status()
        assert status['document_count'] > 0
        store.close()
