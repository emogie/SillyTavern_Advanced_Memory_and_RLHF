# SillyTavern Advanced Memory & RLHF Plugin

I was asked to uplad this, so I did. I can't really support this as I'm not good in programming.

A comprehensive modular plugin for SillyTavern that adds:

- **Middle-Term Memory**: RAG-based retrieval using ChromaDB vector database
- **Long-Term Memory**: LoRA adapter training from accumulated data
- **RLHF**: Reinforcement Learning from Human Feedback via click-based ratings
- **Document Processing**: Import/export in 15+ formats
- **Chunk & Versioning System**: Track training data, manage LoRA adapters across model switches
- **Multi-GPU Support**: NVIDIA CUDA, AMD ROCm/HIP, Apple Metal, CPU fallback
- **Multi-Language UI**: 22 languages with auto-detection from SillyTavern
- **Modular Architecture**: Enable/disable any component independently

---

## Architecture

┌──────────────────────────────────────────────────────┐
│ SillyTavern UI │
│ ┌─────────────┐ ┌──────────┐ ┌────────────────────┐│
│ │ Memory Panel │ │ RLHF 👍👎│ │ Training + Chunks ││
│ │ Query/Store │ │ on msgs │ │ Progress + Model ││
│ └──────┬───────┘ └────┬─────┘ └─────┬──────────────┘│
│ │ │ │ │
│ ┌──────┴──────────────┴─────────────┴────────────┐ │
│ │ Plugin.js (Frontend + i18n) │ │
│ └───────────────────────┬────────────────────────┘ │
└──────────────────────────┼───────────────────────────┘
│ REST API (HTTP)
┌──────────────────────────┼───────────────────────────┐
│ FastAPI Backend (Python) │
│ ┌───────────────────────────────────────────────┐ │
│ │ Module Manager │ │
│ │ ┌─────────┐ ┌──────┐ ┌────────┐ ┌────────┐ │ │
│ │ │ Memory │ │ RLHF │ │Training│ │ Docs │ │ │
│ │ │ Manager │ │ │ │ LoRA │ │ Import │ │ │
│ │ └────┬────┘ └──┬───┘ └───┬────┘ │ Export │ │ │
│ │ │ │ │ └───┬────┘ │ │
│ │ ┌────┴─────────┴─────────┴──────────┴────┐ │ │
│ │ │ ChromaDB Vector Store │ │ │
│ │ └────────────────────────────────────────┘ │ │
│ └───────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────┐ │
│ │ Chunk Manager (data tracking + versioning) │ │
│ └───────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────┐ │
│ │ Device Manager (CUDA / MPS / ROCm / CPU) │ │
│ └───────────────────────────────────────────────┘ │
│ ┌───────────────────────────────────────────────┐ │
│ │ i18n Routes (22 languages) │ │
│ └───────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘

---

## Installation

### 1. Clone into SillyTavern's extensions directory

```
cd SillyTavern/public/scripts/extensions/third-party/
git clone [<repo-url>](https://github.com/emogie/SillyTavern_Advanced_Memory_and_RLHF) Advanced_Memory_and_RLHF

2. Install npm and Python dependencies

cd Advanced_Memory_and_RLHF
npm install

cd Advanced_Memory_and_RLHF/backend
pip install -r requirements.txt

3. Start SillyTavern

The backend starts automatically when SillyTavern loads the plugin (if you drop plugins folder SillyTavern/).

If auto-start fails, start manually:

cd Advanced_Memory_and_RLHF/backend
py start_backend.py

The backend runs on http://127.0.0.1:5125.
Features
Middle-Term Memory (RAG)

    Chat messages are auto-stored in ChromaDB vector database
    Relevant context is automatically injected into prompts via RAG
    Configurable relevance threshold (Min score) and context size (Max chars)
    Upload documents (PDF, DOCX, TXT, etc.) to expand knowledge base
    Per-character memory isolation
    Memory browser with search, pagination, and bulk operations

Long-Term Memory (LoRA)

    When 50+ MB of data accumulates, LoRA training becomes available
    Auto-detects base model from SillyTavern's active API connection
    Existing LoRA is backed up before each training run
    New LoRA auto-loads for future conversations
    Training data (feedback) is archived after successful training
    Chunks are marked as "trained" for tracking
    Supports 4-bit quantization on NVIDIA GPUs for memory efficiency

RLHF Feedback

    Toggle feedback mode to show 👍 👎 ⭐ buttons on AI messages
    Single-click rating collection
    Positive and excellent feedback used as training data alongside memory
    Statistics tracked independently from raw data
    Feedback files archived after training, stats persist as historical record

Chunk & Versioning System

    Data Tracking: Every stored message creates a trackable chunk
    Model Registration: Register which base model you're using
    Model Switching: When you switch models, LoRAs are marked as incompatible
    Data Restoration: Trained data can be restored back to RAG if needed
    LoRA Management: View, select, and delete LoRA adapters per model
    Operation History: Full audit trail of all operations

Document Support
Direction	Formats
Import	TXT, JSON, PDF, XML, DOC, DOCX, ODT, ODS, ODP, XLS, XLSX, PPTX, PNG, JPG, GIF, WEBP
Export	TXT, JSON, PDF, XML, DOCX, ODT, HTML
Other	Print to browser, Generate summary
Multi-Language Support (i18n)

The plugin UI supports 22 languages with automatic detection from SillyTavern's locale:

ar cs de en es fr hi id it ja ko nl no pl pt ru sv th tr uk vi zh

    Language auto-detected from SillyTavern settings
    Manual override available in plugin settings
    Hardcoded English fallback ensures UI is never blank
    Translation coverage indicator shows completion percentage
    Translation files stored in backend/data/i18n/*.json

GPU Support
GPU Type	Framework	Status
NVIDIA	CUDA	Full support + 4-bit quantization
AMD	ROCm/HIP	Full support
Apple	Metal/MPS	Full support
CPU	PyTorch	Fallback (slower)
Module Management

All features can be independently enabled/disabled via the Module Manager panel in the UI.
```

Configuration

Configuration is stored in backend/config.yaml:

server:
  host: "127.0.0.1"
  port: 5125
  cors_origins: ["*"]

memory:
  vector_db_path: "data/vector_db"
  trained_archive_path: "data/trained_archive"
  embedding_model: "all-MiniLM-L6-v2"
  collection_name: "sillytavern_memory"
  min_training_size_mb: 50        # Minimum data before training is available
  chunk_size: 512
  chunk_overlap: 50
  auto_store: true

training:
  lora_models_path: "data/lora_models"
  lora_backups_path: "data/lora_backups"
  trained_archive_path: "data/trained_archive"
  default_base_model: ""          # Auto-detected from SillyTavern
  default_epochs: 3
  default_learning_rate: 0.0002
  default_lora_rank: 16
  default_lora_alpha: 32
  default_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  warmup_ratio: 0.03
  save_steps: 100
  fp16: true

rlhf:
  feedback_path: "data/feedback"
  min_feedback_samples: 100
  reward_model_path: "data/reward_model"

documents:
  exports_path: "data/exports"
  supported_import: [".txt", ".json", ".pdf", ...]
  supported_export: ["txt", "json", "pdf", ...]

modules:
  enabled:
    - "memory"
    - "rlhf"
    - "training"
    - "documents"
    - "vectordb"

API Endpoints
Core
Endpoint	Method	Description
/health	GET	Backend health check
/device/info	GET	GPU/device information
Memory
Endpoint	Method	Description
/memory/status	GET	Memory size and training readiness
/memory/store	POST	Store chat messages
/memory/query	POST	Query vector database
/memory/browse	GET	Browse stored documents with pagination
/memory/delete	POST	Delete specific documents
/memory/clear-all	POST	Clear all memory data
/memory/characters	GET	List characters with memory data
/memory/history	GET	Recent memory operations
Training
Endpoint	Method	Description
/training/start	POST	Start LoRA training (requires base model)
/training/progress	GET	Training progress + ETA
/training/cancel	POST	Cancel running training
/training/lora-status	GET	LoRA adapter status
RLHF
Endpoint	Method	Description
/rlhf/feedback	POST	Submit feedback rating
/rlhf/stats	GET	Feedback statistics
Documents
Endpoint	Method	Description
/documents/ingest	POST	Upload and ingest document
/documents/export	POST	Export chat in selected format
/documents/summary	POST	Generate chat summary
/documents/download/{filename}	GET	Download exported file
Chunks & Versioning
Endpoint	Method	Description
/chunks/overview	GET	System overview (chunks, LoRAs, model)
/chunks/history	GET	Operation history
/chunks/create	POST	Create a data chunk
/chunks/list	GET	List chunks with filtering
/chunks/untrained	GET	Get untrained chunks
/chunks/restorable	GET	Get chunks that can be restored to RAG
/chunks/restore	POST	Restore chunk data back to RAG
/chunks/{chunk_id}	GET	Get chunk details
/chunks/{chunk_id}/documents	GET	Get chunk's preserved documents
/chunks/model/detect	POST	Detect model changes
/chunks/model/register	POST	Register a model
/chunks/model/switch	POST	Handle model switch
/chunks/model/known	GET	List known models
/chunks/lora/list	GET	List all LoRA adapters
/chunks/lora/compatible	GET	List LoRAs compatible with current model
/chunks/lora/select	POST	Select a LoRA for use
/chunks/lora/{lora_id}/delete	POST	Delete a LoRA adapter
/chunks/lora/{lora_id}/unusable	POST	Mark LoRA as unusable
Internationalization
Endpoint	Method	Description
/i18n	GET	List available languages
/i18n/{language_code}	GET	Get translation file for a language
Modules
Endpoint	Method	Description
/modules/list	GET	List all modules
/modules/toggle	POST	Enable/disable a module
Data Flow

User Chat ──► Auto-Store ──► ChromaDB Vector DB (Middle-Term Memory)
                                      │
                    [< 50 MB? Keep collecting]
                                      │
                    [≥ 50 MB? Training available]
                                      │
User clicks "Detect Model" ──► Reads model from SillyTavern API
                                      │
User clicks "Start Training" ──► Backup existing LoRA
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                    Memory Data             RLHF Feedback
                  (from VectorDB)          (👍 👎 ⭐ ratings)
                          │                       │
                          └───────────┬───────────┘
                                      │
                              LoRA Fine-Tuning
                           [Progress Bar + ETA]
                                      │
                          Save New LoRA Adapter
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                   │
             Archive Feedback   Mark Chunks as      Save Training
             (move to archive)  "trained"           Manifest
                    │                 │                   │
                    └─────────────────┼─────────────────┘
                                      │
                          Auto-Load LoRA Adapter
                                      │
                    Next chat uses both:
                    • RAG from new VectorDB data (middle-term)
                    • LoRA adapter knowledge (long-term)
                    • Fresh feedback collection begins

Directory Structure

Advanced_Memory_and_RLHF/
├── plugin.js                          # SillyTavern plugin entry point (frontend)
├── manifest.json                      # Plugin manifest for SillyTavern
├── package.json                       # Node.js dependencies
├── style.css                          # Plugin UI styles
├── README.md                          # This file
│
├── backend/
│   ├── start_backend.py              # Backend auto-starter (called by ST)
│   ├── server.py                      # FastAPI backend server + routes
│   ├── config.yaml                    # Configuration
│   ├── requirements.txt              # Python dependencies
│   ├── chunk_manager.py              # Chunk & versioning system
│   ├── chunk_routes.py               # Chunk API routes (FastAPI Router)
│   ├── i18n_routes.py                # i18n API routes (FastAPI Router)
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── module_manager.py         # Module loader/unloader
│   │   │
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── memory_manager.py     # Memory orchestrator
│   │   │   ├── middle_term.py        # Middle-term memory (RAG + Vector DB)
│   │   │   └── long_term.py          # Long-term memory (LoRA)
│   │   │
│   │   ├── rlhf/
│   │   │   ├── __init__.py
│   │   │   ├── feedback_collector.py # Collects user feedback ratings
│   │   │   ├── reward_model.py       # Reward model training
│   │   │   └── rlhf_trainer.py       # RLHF training pipeline
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── lora_trainer.py       # LoRA fine-tuning + archiving
│   │   │   ├── device_manager.py     # GPU/CPU detection
│   │   │   └── progress_tracker.py   # Training progress tracking
│   │   │
│   │   ├── documents/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py             # Document ingestion pipeline
│   │   │   ├── parsers.py            # File format parsers
│   │   │   └── export.py             # Export/print functionality
│   │   │
│   │   └── vectordb/
│   │       ├── __init__.py
│   │       └── chroma_store.py       # ChromaDB vector store
│   │
│   ├── data/
│   │   ├── vector_db/                # ChromaDB database files
│   │   ├── trained_archive/          # Archived feedback after training
│   │   ├── lora_models/              # Current LoRA adapters
│   │   ├── lora_backups/             # LoRA backups before retraining
│   │   ├── feedback/                 # Active RLHF feedback data
│   │   ├── exports/                  # Exported documents
│   │   ├── reward_model/             # Reward model data
│   │   ├── chunks/                   # Chunk tracking data
│   │   ├── models/                   # Model registry data
│   │   ├── loras/                    # LoRA registry data
│   │   ├── history/                  # Operation history
│   │   └── i18n/                     # Translation files (22 languages)
│   │       ├── en.json
│   │       ├── de.json
│   │       ├── fr.json
│   │       ├── es.json
│   │       ├── ja.json
│   │       ├── ... (22 total)
│   │       └── template.json         # Template for new translations
│   │
│   └── tests/
│       ├── test_memory.py
│       ├── test_training.py
│       └── test_documents.py
│
└── node_modules/                      # Node.js dependencies (auto-generated)

Security

    CORS: Restricted to localhost origins only
    Path Validation: All file paths checked for traversal attacks (.., shell injection)
    Rate Limiting: Store endpoints limited to 10 requests per 5 seconds
    Input Sanitization: Language codes and file paths sanitized before use
    Local Only: Backend binds to 127.0.0.1 — not accessible from network

Troubleshooting
Backend won't start

Check the SillyTavern console for error messages. Common issues:

    Missing Python: Install Python 3.10+ and ensure py or python is in PATH
    Missing dependencies: Run pip install -r backend/requirements.txt
    Port conflict: Another service is using port 5125

Start manually for detailed errors:

cd backend
py start_backend.py

Translations not loading

    Check that backend/data/i18n/ contains .json files
    Verify JSON syntax: py -c "import json; json.load(open('data/i18n/en.json')); print('OK')"
    Check backend console for [i18n] log messages
    Test directly: open http://127.0.0.1:5125/i18n in browser

Training fails

    "No base model": Click "Detect Model" or enter model path manually
    "Insufficient data": Need 50+ MB in vector database before training
    Cloud API models (GPT, Claude): LoRA training requires local model weights
    Out of memory: Reduce batch size, enable 4-bit quantization (NVIDIA only)

Chunk routes return 404

Ensure chunk_routes.py has catch-all routes (/{chunk_id}) at the bottom of the file, after all fixed-path routes like /overview, /history, /lora/*.
Adding Translations

    Copy backend/data/i18n/template.json to backend/data/i18n/<code>.json
    Fill in translations for all keys
    Update the _meta section with language name and author
    Restart the backend — new language appears automatically

Adding Modules

    Create a new directory under backend/modules/
    Implement the module with an __init__.py
    Register it in server.py's lifespan function
    Add API endpoints as needed
    Add the module name to config.yaml under modules.enabled

This README reflects all the changes we made including:

- FastAPI backend (not Flask)
- Chunk & versioning system
- i18n system with 22 languages
- Model auto-detection from SillyTavern
- Training data archiving after successful training
- Security middleware (CORS, path validation, rate limiting)
- Updated directory structure with all actual files
- Troubleshooting for common issues we encountered
- Correct installation path (`extensions/third-party/` not `plugins/`)

if you make changes to Python files, delete the __pycache__ folders to avoid stale cached code causing confusion:
for /d /r "C:\SillyTavern-Launcher\SillyTavern\public\scripts\extensions\third-party\Advanced_Memory_and_RLHF\backend" %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

And if you edit any .json translation files, you can quickly validate them all at once:
cd backend\data\i18n
for %f in (*.json) do @py -c "import json; json.load(open('%f',encoding='utf-8')); print('OK: %f')" 2>&1 | findstr /v "OK:" && echo FAIL: %f

License
MIT License

Copyright (c) 2026 emogie
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO 
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
