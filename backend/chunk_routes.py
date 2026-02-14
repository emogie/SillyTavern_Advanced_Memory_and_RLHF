"""
API routes for the Chunk & Versioning System
Converted from Flask Blueprint to FastAPI Router
"""

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

chunk_router = APIRouter(prefix="/chunks", tags=["chunks"])

# This will be set during initialization
chunk_manager = None


def init_chunk_routes(app, manager):
    """Initialize chunk routes with the ChunkManager instance"""
    global chunk_manager
    chunk_manager = manager
    app.include_router(chunk_router)


# ==================== MODEL ENDPOINTS ====================

@chunk_router.post('/model/detect')
async def detect_model(request: Request):
    data = await request.json()
    model_path = data.get('model_path', '')
    if not model_path:
        return JSONResponse(status_code=400, content={'error': 'model_path required'})
    try:
        result = chunk_manager.detect_model_change(model_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Model detection failed: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/model/register')
async def register_model(request: Request):
    data = await request.json()
    model_path = data.get('model_path', '')
    friendly_name = data.get('friendly_name', None)
    if not model_path:
        return JSONResponse(status_code=400, content={'error': 'model_path required'})
    try:
        result = chunk_manager.register_model(model_path, friendly_name)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/model/switch')
async def switch_model(request: Request):
    data = await request.json()
    model_path = data.get('model_path', '')
    friendly_name = data.get('friendly_name', None)
    if not model_path:
        return JSONResponse(status_code=400, content={'error': 'model_path required'})
    try:
        result = chunk_manager.handle_model_switch(model_path, friendly_name)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/model/known')
async def known_models():
    try:
        models = chunk_manager.get_known_models()
        return JSONResponse(content={'models': models})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# ==================== CHUNK ENDPOINTS (fixed paths FIRST) ====================

@chunk_router.post('/create')
async def create_chunk(request: Request):
    data = await request.json()
    documents = data.get('documents', [])
    character = data.get('character', None)
    metadata = data.get('metadata', None)
    if not documents:
        return JSONResponse(status_code=400, content={'error': 'documents required'})
    try:
        chunk = chunk_manager.create_chunk(documents, character, metadata)
        return JSONResponse(content=chunk.to_dict())
    except Exception as e:
        logger.error(f"Chunk creation failed: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/list')
async def list_chunks(
    status: str = Query(default=None),
    model: str = Query(default=None)
):
    try:
        chunks = chunk_manager.get_chunks_by_status(status, model)
        return JSONResponse(content={'chunks': chunks, 'total': len(chunks)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/untrained')
async def untrained_chunks():
    try:
        chunks = chunk_manager.get_untrained_chunks()
        return JSONResponse(content={'chunks': chunks, 'total': len(chunks)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/restorable')
async def restorable_chunks(model: str = Query(default=None)):
    try:
        chunks = chunk_manager.get_restorable_chunks(model)
        return JSONResponse(content={
            'chunks': chunks,
            'total': len(chunks),
            'total_documents': sum(c.get('document_count', 0) for c in chunks)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/restore')
async def restore_chunks(request: Request):
    data = await request.json()
    chunk_ids = data.get('chunk_ids', None)
    try:
        result = chunk_manager.restore_chunks_to_rag(chunk_ids)
        return JSONResponse(content={
            'restored_chunks': result['restored_chunks'],
            'failed_chunks': result['failed_chunks'],
            'total_documents': result['total_documents'],
            'documents': result['documents'],
            'message': f"Restored {len(result['restored_chunks'])} chunks with {result['total_documents']} documents"
        })
    except Exception as e:
        logger.error(f"Chunk restore failed: {e}")
        return JSONResponse(status_code=500, content={'error': str(e)})


# ==================== OVERVIEW & HISTORY (before catch-all!) ====================

@chunk_router.get('/overview')
async def overview():
    try:
        data = chunk_manager.get_overview()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/history')
async def operation_history(limit: int = Query(default=100)):
    try:
        history = chunk_manager.get_operation_history(limit)
        return JSONResponse(content={'history': history, 'total': len(history)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# ==================== LORA ENDPOINTS (before catch-all!) ====================

@chunk_router.get('/lora/list')
async def list_loras(include_deleted: str = Query(default='false')):
    include = include_deleted.lower() == 'true'
    try:
        loras = chunk_manager.get_all_loras(include)
        return JSONResponse(content={'loras': loras, 'total': len(loras)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/lora/compatible')
async def compatible_loras(model: str = Query(default=None)):
    try:
        loras = chunk_manager.get_compatible_loras(model)
        return JSONResponse(content={
            'loras': loras,
            'total': len(loras),
            'current_model': chunk_manager.current_model
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/lora/select')
async def select_lora(request: Request):
    data = await request.json()
    lora_id = data.get('lora_id', '')
    if not lora_id:
        return JSONResponse(status_code=400, content={'error': 'lora_id required'})
    try:
        result = chunk_manager.select_lora(lora_id)
        if result is None:
            return JSONResponse(status_code=404, content={'error': 'LoRA not found'})
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/lora/{lora_id}/delete')
async def delete_lora(lora_id: str, request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    delete_files = data.get('delete_files', False)
    try:
        success = chunk_manager.delete_lora(lora_id, delete_files)
        if success:
            return JSONResponse(content={'message': f'LoRA {lora_id} deleted', 'files_deleted': delete_files})
        return JSONResponse(status_code=404, content={'error': 'LoRA not found'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.post('/lora/{lora_id}/unusable')
async def mark_unusable(lora_id: str, request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    reason = data.get('reason', 'Marked by user')
    try:
        chunk_manager.mark_lora_unusable(lora_id, reason)
        return JSONResponse(content={'message': f'LoRA {lora_id} marked as unusable'})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# ==================== CATCH-ALL ROUTES (MUST BE LAST!) ====================

@chunk_router.get('/{chunk_id}')
async def get_chunk(chunk_id: str):
    try:
        chunk = chunk_manager.get_chunk(chunk_id)
        if not chunk:
            return JSONResponse(status_code=404, content={'error': 'Chunk not found'})
        return JSONResponse(content=chunk)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@chunk_router.get('/{chunk_id}/documents')
async def get_chunk_documents(chunk_id: str):
    try:
        documents = chunk_manager.get_chunk_documents(chunk_id)
        if documents is None:
            return JSONResponse(status_code=404, content={'error': 'Documents not found'})
        return JSONResponse(content={
            'chunk_id': chunk_id,
            'documents': documents,
            'count': len(documents)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})