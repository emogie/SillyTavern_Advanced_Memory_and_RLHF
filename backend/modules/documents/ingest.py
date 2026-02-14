"""
Document Ingestion Pipeline - Processes uploaded documents into the vector store.
"""
import logging
from pathlib import Path
from typing import Dict, Any
from modules.documents.parsers import DocumentParser
from modules.vectordb.chroma_store import ChromaStore
logger = logging.getLogger("DocumentIngestor")

class DocumentIngestor:
    """Ingests documents of various formats into the vector database."""
    def __init__(self, config: dict, vector_store: ChromaStore):
        self.config = config
        self.vector_store = vector_store
        self.supported = config.get('supported_import', [])
    def ingest(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Process and store a document."""
        ext = Path(filename).suffix.lower()
        if ext not in self.supported and self.supported:
            return {"status": "error", "message": f"Unsupported format: {ext}"}
        try:
            text = DocumentParser.parse(filename, content)
            if not text.strip():
                return {"status": "warning", "message": "No text extracted from document"}
            result = self.vector_store.add_texts(
                [text],
                [{"source": filename, "type": "document", "format": ext}]
            )
            logger.info(f"Ingested {filename}: {result['added']} chunks")
            return {
                "status": "ok",
                "filename": filename,
                "chunks_added": result['added'],
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {e}")
            return {"status": "error", "message": str(e)}
