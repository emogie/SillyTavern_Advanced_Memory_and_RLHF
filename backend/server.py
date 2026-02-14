"""
FastAPI backend server for the Advanced Memory & RLHF Plugin.
Provides REST API endpoints for all plugin functionality.
"""
import os
import yaml
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import our modules
from modules.module_manager import ModuleManager
from modules.memory.memory_manager import MemoryManager
from modules.training.device_manager import DeviceManager
from modules.training.lora_trainer import LoRATrainer
from modules.training.progress_tracker import ProgressTracker
from modules.rlhf.feedback_collector import FeedbackCollector
from modules.documents.ingest import DocumentIngestor
from modules.documents.export import DocumentExporter
from modules.vectordb.chroma_store import ChromaStore
from fastapi import FastAPI
app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemoryPlugin")

# ===================== CONFIGURATION =====================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# ===================== GLOBAL STATE =====================
module_manager: Optional[ModuleManager] = None
memory_manager: Optional[MemoryManager] = None
device_manager: Optional[DeviceManager] = None
lora_trainer: Optional[LoRATrainer] = None
progress_tracker: Optional[ProgressTracker] = None
feedback_collector: Optional[FeedbackCollector] = None
doc_ingestor: Optional[DocumentIngestor] = None
doc_exporter: Optional[DocumentExporter] = None
vector_store: Optional[ChromaStore] = None

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down all components."""
    global module_manager, memory_manager, device_manager
    global lora_trainer, progress_tracker, feedback_collector
    global doc_ingestor, doc_exporter, vector_store

    logger.info("Initializing plugin components...")

    # Device detection
    device_manager = DeviceManager()
    logger.info(f"Detected device: {device_manager.device_name} ({device_manager.device_type})")

    # Vector store
    vector_store = ChromaStore(config['memory'])

    # Memory manager
    memory_manager = MemoryManager(config['memory'], vector_store)

    # Training
    progress_tracker = ProgressTracker()
    lora_trainer = LoRATrainer(config['training'], device_manager, progress_tracker)

    # RLHF
    feedback_collector = FeedbackCollector(config['rlhf'])

    # Documents
    doc_ingestor = DocumentIngestor(config['documents'], vector_store)
    doc_exporter = DocumentExporter(config['documents'])

    # Module manager
    module_manager = ModuleManager(config['modules'])
    module_manager.register_module('memory', 'Memory Management',
        'Middle-term RAG and long-term LoRA memory', memory_manager)
    module_manager.register_module('rlhf', 'RLHF Feedback',
        'Reinforcement Learning from Human Feedback', feedback_collector)
    module_manager.register_module('training', 'LoRA Training',
        'Fine-tune LoRA adapters from collected data', lora_trainer)
    module_manager.register_module('documents', 'Document Processing',
        'Import, export, and process documents', doc_ingestor)
    module_manager.register_module('vectordb', 'Vector Database',
        'ChromaDB vector storage for RAG', vector_store)

    # Auto-load LoRA if available
    lora_trainer.auto_load_lora()

    logger.info("All components initialized.")
    yield
    logger.info("Shutting down plugin components...")
    vector_store.close()

# ===================== APP =====================
app = FastAPI(
    title="SillyTavern Memory & RLHF Plugin",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config['server'].get('cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== REQUEST/RESPONSE MODELS =====================
class MemoryStoreRequest(BaseModel):
    character: str = "Unknown"
    messages: List[Dict[str, Any]]
    auto_stored: bool = False
    timestamp: Optional[int] = None

class MemoryQueryRequest(BaseModel):
    query: str
    k: int = 5
    min_score: float = 0.0

class MemoryDeleteRequest(BaseModel):
    doc_ids: List[str] = []
    character: Optional[str] = None
    before_timestamp: Optional[int] = None
    auto_stored_only: bool = False

class TrainingConfig(BaseModel):
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    batch_size: int = 4
    base_model: Optional[str] = None

class FeedbackRequest(BaseModel):
    message_id: str
    rating: str  # 'positive', 'negative', 'excellent'
    response_text: str
    prompt_text: str = ""
    character: str = "Unknown"
    timestamp: Optional[int] = None

class ExportRequest(BaseModel):
    format: str
    chat_data: Dict[str, Any]

class SummaryRequest(BaseModel):
    chat_data: Dict[str, Any]

class ModuleToggleRequest(BaseModel):
    name: str
    enabled: bool

# ===================== HEALTH =====================
@app.get("/health")
async def health_check():
    return {"status": "ok", "plugin": "AdvancedMemoryRLHF", "version": "1.0.0"}

# ===================== DEVICE =====================
@app.get("/device/info")
async def get_device_info():
    return device_manager.get_device_info()

# ===================== I18N =====================
@app.get("/i18n/{language_code}")
async def get_translation(language_code: str):
    import json
    
    # Sanitize
    safe_code = ''.join(c for c in language_code if c.isalnum() or c in ('-', '_'))
    if safe_code != language_code or '..' in language_code:
        raise HTTPException(status_code=400, detail="Invalid language code")
    
    i18n_dir = BASE_DIR / "data" / "i18n"
    lang_file = i18n_dir / f"{safe_code}.json"
    
    logger.info(f"[i18n] Request for '{safe_code}', looking at: {lang_file}")
    
    if not lang_file.exists():
        available = [f.name for f in i18n_dir.glob("*.json")] if i18n_dir.exists() else []
        logger.warning(f"[i18n] Not found: {lang_file}, available: {available}")
        raise HTTPException(
            status_code=404,
            detail=f"Language '{safe_code}' not found. Available: {available}"
        )
    
    try:
        with open(lang_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        meta = data.pop('_meta', None)
        key_count = len(data)
        logger.info(f"[i18n] Serving '{safe_code}': {key_count} translation keys")
        
        # Return in the format plugin.js expects
        return {
            "translations": data,
            "meta": meta
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {safe_code}.json: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================== I18N =====================
@app.get("/i18n")
async def list_translations():
    import json
    i18n_dir = BASE_DIR / "data" / "i18n"
    
    logger.info(f"[i18n] Scanning: {i18n_dir}")
    logger.info(f"[i18n] Exists: {i18n_dir.exists()}")
    
    languages = []
    if i18n_dir.exists():
        json_files = sorted(i18n_dir.glob("*.json"))
        logger.info(f"[i18n] Found {len(json_files)} files")
        
        for f in json_files:
            if f.name == "template.json":
                continue
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    meta = data.get('_meta', {})
                    languages.append({
                        "code": f.stem,
                        "name": meta.get('language', f.stem),
                        "native_name": meta.get('native_name', f.stem),
                        "author": meta.get('author', 'Unknown'),
                        "version": meta.get('version', '1.0.0'),
                        "file": f.name
                    })
            except Exception as e:
                logger.warning(f"[i18n] Could not read {f.name}: {e}")
                languages.append({
                    "code": f.stem,
                    "name": f.stem,
                    "native_name": f.stem,
                    "file": f.name
                })
    
    logger.info(f"[i18n] Available languages: {[l['code'] for l in languages]}")
    
    return {
        "available_languages": languages,
        "i18n_path": str(i18n_dir),
        "default_language": "en",
        "data_dir_exists": i18n_dir.exists()
    }

# ===================== MEMORY ENDPOINTS =====================
@app.get("/memory/status")
async def memory_status():
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    return memory_manager.get_status()

@app.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    result = memory_manager.store(
        character=request.character,
        messages=request.messages,
        auto_stored=request.auto_stored
    )
    return result

@app.post("/memory/query")
async def query_memory(request: MemoryQueryRequest):
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    results = memory_manager.query(request.query, request.k, request.min_score)
    return {"results": results}

@app.get("/memory/browse")
async def browse_memory(character: str = None, offset: int = 0, limit: int = 50, sort_order: str = "newest"):
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    results = memory_manager.browse(
        character=character,
        offset=offset,
        limit=limit,
        sort_order=sort_order
    )
    return results

@app.post("/memory/delete")
async def delete_memory(request: MemoryDeleteRequest):
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    result = memory_manager.delete(
        doc_ids=request.doc_ids,
        character=request.character,
        before_timestamp=request.before_timestamp,
        auto_stored_only=request.auto_stored_only
    )
    return result

@app.post("/memory/clear-all")
async def clear_all_memory():
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    result = memory_manager.clear_all()
    return result

@app.get("/memory/characters")
async def list_characters():
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    return {"characters": memory_manager.get_characters()}

@app.get("/memory/history")
async def memory_history(limit: int = 100):
    if not module_manager.is_enabled('memory'):
        raise HTTPException(status_code=503, detail="Memory module disabled")
    return {"history": memory_manager.get_history(limit=limit)}

# ===================== TRAINING ENDPOINTS =====================
@app.post("/training/start")
async def start_training(config_req: TrainingConfig):
    if not module_manager.is_enabled('training'):
        raise HTTPException(status_code=503, detail="Training module disabled")

    # Validate base model
    base_model = config_req.base_model
    if not base_model:
        base_model = config.get('training', {}).get('default_base_model', '')
    if not base_model:
        raise HTTPException(
            status_code=400,
            detail="No base model specified. Please detect or enter a model path."
        )

    # Check memory size
    status = memory_manager.get_status()
    min_size = config['memory'].get('min_training_size_mb', 50)
    if status['total_size_mb'] < min_size:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data: {status['total_size_mb']:.1f}MB / {min_size}MB minimum required"
        )

    if progress_tracker.is_running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    training_data = memory_manager.get_training_data()
    feedback_data = feedback_collector.get_feedback_data() if module_manager.is_enabled('rlhf') else []

    # Pass the validated config with base_model
    train_config = config_req.dict()
    train_config['base_model'] = base_model

    asyncio.get_event_loop().run_in_executor(
        None,
        lora_trainer.train,
        training_data,
        feedback_data,
        train_config
    )

    return {
        "status": "training_started",
        "message": "LoRA training initiated",
        "base_model": base_model
    }

@app.get("/training/progress")
async def training_progress():
    return progress_tracker.get_progress()

@app.post("/training/cancel")
async def cancel_training():
    progress_tracker.cancel()
    return {"status": "cancellation_requested"}

@app.get("/training/lora-status")
async def lora_status():
    return lora_trainer.get_lora_status()

# ===================== RLHF ENDPOINTS =====================
@app.post("/rlhf/feedback")
async def submit_feedback(request: FeedbackRequest):
    if not module_manager.is_enabled('rlhf'):
        raise HTTPException(status_code=503, detail="RLHF module disabled")
    result = feedback_collector.store_feedback(request.dict())
    return result

@app.get("/rlhf/stats")
async def feedback_stats():
    if not module_manager.is_enabled('rlhf'):
        return {"positive": 0, "negative": 0, "excellent": 0, "total": 0}
    return feedback_collector.get_stats()

# ===================== DOCUMENT ENDPOINTS =====================
@app.post("/documents/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not module_manager.is_enabled('documents'):
        raise HTTPException(status_code=503, detail="Documents module disabled")
    contents = await file.read()
    result = doc_ingestor.ingest(file.filename, contents)
    return result

@app.post("/documents/export")
async def export_chat(request: ExportRequest):
    if not module_manager.is_enabled('documents'):
        raise HTTPException(status_code=503, detail="Documents module disabled")
    result = doc_exporter.export(request.format, request.chat_data)
    return result

@app.post("/documents/summary")
async def generate_summary(request: SummaryRequest):
    if not module_manager.is_enabled('documents'):
        raise HTTPException(status_code=503, detail="Documents module disabled")
    summary = doc_exporter.generate_summary(request.chat_data)
    return {"summary": summary}

@app.get("/documents/download/{filename}")
async def download_file(filename: str):
    filepath = BASE_DIR / "data" / "exports" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)

# ===================== MODULE MANAGEMENT =====================
@app.get("/modules/list")
async def list_modules():
    return {"modules": module_manager.list_modules()}

@app.post("/modules/toggle")
async def toggle_module(request: ModuleToggleRequest):
    result = module_manager.toggle_module(request.name, request.enabled)
    return result