"""
Auto-start script for the Memory Plugin backend.
Called automatically when SillyTavern loads the plugin.
"""
import os
import sys
import subprocess
import signal
import logging
import traceback

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

from collections import defaultdict
import time
import re
import uvicorn

# --- Patch for Windows: torch.distributed may be incomplete ---
def patch_torch_distributed():
    """Fix missing torch.distributed functions on Windows."""
    try:
        import torch.distributed
        if not hasattr(torch.distributed, 'is_initialized'):
            torch.distributed.is_initialized = lambda: False
        if not hasattr(torch.distributed, 'get_rank'):
            torch.distributed.get_rank = lambda: 0
    except ImportError:
        pass

patch_torch_distributed()
# --- End patch ---

def ensure_dependencies():
    """Install missing Python dependencies."""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-r', requirements_file, '-q'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print("[MemoryPlugin] Warning: Some dependencies may not be installed.")

def create_data_dirs():
    """Ensure all data directories exist."""
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        'data/vector_db',
        'data/trained_archive',
        'data/lora_models',
        'data/lora_backups',
        'data/feedback',
        'data/exports',
        'data/reward_model',
        'data/i18n',
    ]
    for d in dirs:
        full_path = os.path.join(base, d)
        os.makedirs(full_path, exist_ok=True)
    print(f"[MemoryPlugin] Data directories verified at: {base}")

# Global rate limit variables
_rate_limits = defaultdict(list)
RATE_LIMIT_WINDOW = 5
RATE_LIMIT_MAX = 10

def main():
    print("[MemoryPlugin] Starting backend server...")
    print(f"[MemoryPlugin] Python: {sys.executable}")
    print(f"[MemoryPlugin] Script: {os.path.abspath(__file__)}")
    print(f"[MemoryPlugin] CWD: {os.getcwd()}")

    create_data_dirs()
    ensure_dependencies()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("[MemoryPlugin] Shutting down backend...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Import everything inside try/except so errors are visible
    try:
        print("[MemoryPlugin] Importing server...")
        from server import app
        print("[MemoryPlugin] Server imported OK")
    except Exception as e:
        print(f"[MemoryPlugin] FATAL: Failed to import server: {e}")
        traceback.print_exc()
        sys.exit(1)

    try:
        print("[MemoryPlugin] Importing chunk system...")
        from chunk_manager import ChunkManager
        from chunk_routes import init_chunk_routes
        print("[MemoryPlugin] Chunk system imported OK")
    except Exception as e:
        print(f"[MemoryPlugin] WARNING: Chunk system not available: {e}")
        traceback.print_exc()
        ChunkManager = None
        init_chunk_routes = None

    try:
        print("[MemoryPlugin] Importing i18n routes...")
        # from i18n_routes import i18n_router
        print("[MemoryPlugin] i18n routes imported OK")
    except Exception as e:
        print(f"[MemoryPlugin] WARNING: i18n routes not available: {e}")
        traceback.print_exc()
        i18n_router = None

    # ---------- CORS ----------
    try:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost",
                "http://127.0.0.1",
                "http://0.0.0.0",
            ],
            allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("[MemoryPlugin] CORS middleware added")
    except Exception as e:
        print(f"[MemoryPlugin] WARNING: CORS setup failed: {e}")
        traceback.print_exc()

    # ---------- Security Middleware ----------
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def validate_and_rate_limit(request: Request, call_next):
            # Path traversal validation
            if request.headers.get("content-type", "").startswith("application/json"):
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        for key in ['model_path', 'path', 'file_path']:
                            if key in body and isinstance(body[key], str):
                                path = body[key]
                                if '..' in path or path.startswith('/etc') or path.startswith('/proc'):
                                    return JSONResponse(
                                        status_code=400,
                                        content={'error': 'Invalid path: potential traversal detected'}
                                    )
                                if re.search(r'[;&|`\)]', path):
                                    return JSONResponse(
                                        status_code=400,
                                        content={'error': 'Invalid path: forbidden characters'}
                                    )
                except Exception:
                    pass

            # Rate limiting for store endpoints
            url_path = request.url.path
            if 'store' in url_path:
                client = request.client.host if request.client else "unknown"
                now = time.time()
                _rate_limits[client] = [t_val for t_val in _rate_limits[client] if now - t_val < RATE_LIMIT_WINDOW]
                if len(_rate_limits[client]) >= RATE_LIMIT_MAX:
                    return JSONResponse(
                        status_code=429,
                        content={'error': 'Rate limited. Please slow down.'}
                    )
                _rate_limits[client].append(now)

            response = await call_next(request)
            return response

        print("[MemoryPlugin] Security middleware added")
    except Exception as e:
        print(f"[MemoryPlugin] WARNING: Security middleware failed: {e}")
        traceback.print_exc()

    # ---------- Register i18n routes ----------
    #if i18n_router is not None:
    #    try:
    #        app.include_router(i18n_router)
    #        print("[MemoryPlugin] i18n routes registered")
    #    except Exception as e:
    #        print(f"[MemoryPlugin] WARNING: i18n route registration failed: {e}")
    #        traceback.print_exc()

    # ---------- Initialize chunk system ----------
    if ChunkManager is not None and init_chunk_routes is not None:
        try:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            manager = ChunkManager(data_dir)
            init_chunk_routes(app, manager)
            print("[MemoryPlugin] Chunk system initialized")
        except Exception as e:
            print(f"[MemoryPlugin] WARNING: Chunk system init failed: {e}")
            traceback.print_exc()

    # ---------- List all registered routes ----------
    print("[MemoryPlugin] Registered routes:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            print(f"  {', '.join(route.methods):10s} {route.path}")

    print("[MemoryPlugin] Starting server on http://127.0.0.1:5125")
    print("[MemoryPlugin] Press Ctrl+C to stop")

    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=5125,
            log_level="info",
            access_log=False,
        )
    except Exception as e:
        print(f"[MemoryPlugin] FATAL: Server failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[MemoryPlugin] FATAL UNHANDLED ERROR: {e}")
        traceback.print_exc()
        # Keep window open on Windows so you can read the error
        input("\nPress Enter to exit...")
        sys.exit(1)