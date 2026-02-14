"""
Chunk & Versioning System for Model-Aware Memory Management
Handles model switching, LoRA lifecycle, and RAG data restoration
"""

import os
import json
import hashlib
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class ModelFingerprint:
    """Identifies a model uniquely using multiple methods"""

    @staticmethod
    def compute_file_checksum(filepath: str, algorithm: str = 'sha256', chunk_size: int = 8192 * 1024) -> str:
        """Compute checksum of a model file (reads in chunks for large files)"""
        h = hashlib.new(algorithm)
        file_size = os.path.getsize(filepath)

        # For very large files (>10GB), use partial hashing for speed
        if file_size > 10 * 1024 * 1024 * 1024:
            return ModelFingerprint._compute_partial_checksum(filepath, algorithm, file_size)

        with open(filepath, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()

    @staticmethod
    def _compute_partial_checksum(filepath: str, algorithm: str, file_size: int) -> str:
        """For very large files: hash beginning, middle, end + file size"""
        h = hashlib.new(algorithm)
        sample_size = 64 * 1024 * 1024  # 64MB samples

        with open(filepath, 'rb') as f:
            # Beginning
            h.update(f.read(sample_size))
            # Middle
            f.seek(file_size // 2)
            h.update(f.read(sample_size))
            # End
            f.seek(max(0, file_size - sample_size))
            h.update(f.read(sample_size))

        # Include file size in hash
        h.update(str(file_size).encode())
        return 'partial_' + h.hexdigest()

    @staticmethod
    def compute_config_fingerprint(config_path: str) -> str:
        """Compute fingerprint from model config (architecture, vocab size, etc.)"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Extract architecture-defining fields
            relevant_keys = [
                'model_type', 'architectures', 'vocab_size', 'hidden_size',
                'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
                'max_position_embeddings', 'num_key_value_heads'
            ]

            fingerprint_data = {}
            for key in relevant_keys:
                if key in config:
                    fingerprint_data[key] = config[key]

            data_str = json.dumps(fingerprint_data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()[:32]
        except Exception as e:
            logger.warning(f"Could not compute config fingerprint: {e}")
            return None

    @staticmethod
    def identify_model(model_path: str) -> Dict[str, Any]:
        """
        Create a complete model identity using multiple signals.
        Returns a dict with all available identification info.
        """
        identity = {
            'path': str(model_path),
            'name': os.path.basename(model_path),
            'detected_at': datetime.now().isoformat(),
            'file_checksum': None,
            'config_fingerprint': None,
            'file_size': None,
            'model_type': None,
            'architecture': None,
            'identity_hash': None  # Combined unique identifier
        }

        model_path = Path(model_path)

        # Method 1: Config-based fingerprint (fast, architecture-aware)
        config_candidates = [
            model_path / 'config.json',
            model_path.parent / 'config.json'
        ]
        for config_path in config_candidates:
            if config_path.exists():
                identity['config_fingerprint'] = ModelFingerprint.compute_config_fingerprint(str(config_path))
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    identity['model_type'] = config.get('model_type', 'unknown')
                    identity['architecture'] = config.get('architectures', ['unknown'])[0] if 'architectures' in config else 'unknown'
                except:
                    pass
                break

        # Method 2: File-based checksum (slow but definitive)
        # Look for the main model file
        model_files = []
        if model_path.is_file():
            model_files = [model_path]
        elif model_path.is_dir():
            for pattern in ['*.safetensors', '*.bin', '*.gguf', '*.ggml', '*.pt']:
                model_files.extend(model_path.glob(pattern))

        if model_files:
            main_file = sorted(model_files, key=lambda f: f.stat().st_size, reverse=True)[0]
            identity['file_size'] = main_file.stat().st_size
            identity['file_checksum'] = ModelFingerprint.compute_file_checksum(str(main_file))

        # Create combined identity hash
        id_components = []
        if identity['config_fingerprint']:
            id_components.append(identity['config_fingerprint'])
        if identity['file_checksum']:
            id_components.append(identity['file_checksum'])
        if identity['file_size']:
            id_components.append(str(identity['file_size']))
        if identity['name']:
            id_components.append(identity['name'])

        if id_components:
            combined = '|'.join(id_components)
            identity['identity_hash'] = hashlib.sha256(combined.encode()).hexdigest()[:16]
        else:
            # Fallback: use path + name
            identity['identity_hash'] = hashlib.sha256(str(model_path).encode()).hexdigest()[:16]

        return identity


class ChunkStatus:
    """Possible states for a data chunk"""
    PENDING = 'pending'           # In RAG, not yet trained
    TRAINING = 'training'         # Currently being trained
    TRAINED = 'trained'           # Successfully trained into LoRA
    FAILED = 'failed'             # Training failed, data preserved in RAG
    RESTORED = 'restored'         # Was trained, but restored to RAG (model changed)
    ARCHIVED = 'archived'         # Old, no longer active


class DataChunk:
    """Represents a batch of data that can be tracked through the training pipeline"""

    def __init__(self, chunk_id: str, data: dict = None):
        self.chunk_id = chunk_id
        self.created_at = datetime.now().isoformat()
        self.status = ChunkStatus.PENDING
        self.model_identity_hash = None
        self.lora_id = None
        self.document_ids = []
        self.document_count = 0
        self.character = None
        self.metadata = data or {}
        self.history = []

    def to_dict(self) -> dict:
        return {
            'chunk_id': self.chunk_id,
            'created_at': self.created_at,
            'status': self.status,
            'model_identity_hash': self.model_identity_hash,
            'lora_id': self.lora_id,
            'document_ids': self.document_ids,
            'document_count': self.document_count,
            'character': self.character,
            'metadata': self.metadata,
            'history': self.history
        }

    @staticmethod
    def from_dict(data: dict) -> 'DataChunk':
        chunk = DataChunk(data['chunk_id'])
        chunk.created_at = data.get('created_at', datetime.now().isoformat())
        chunk.status = data.get('status', ChunkStatus.PENDING)
        chunk.model_identity_hash = data.get('model_identity_hash')
        chunk.lora_id = data.get('lora_id')
        chunk.document_ids = data.get('document_ids', [])
        chunk.document_count = data.get('document_count', 0)
        chunk.character = data.get('character')
        chunk.metadata = data.get('metadata', {})
        chunk.history = data.get('history', [])
        return chunk

    def add_history(self, action: str, details: str = ''):
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })


class LoRARecord:
    """Tracks a LoRA adapter and its relationship to chunks and models"""

    def __init__(self, lora_id: str):
        self.lora_id = lora_id
        self.created_at = datetime.now().isoformat()
        self.model_identity_hash = None
        self.model_name = None
        self.model_type = None
        self.chunk_ids = []
        self.path = None
        self.status = 'active'  # active, unusable, deleted
        self.training_config = {}
        self.metrics = {}
        self.notes = ''

    def to_dict(self) -> dict:
        return {
            'lora_id': self.lora_id,
            'created_at': self.created_at,
            'model_identity_hash': self.model_identity_hash,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'chunk_ids': self.chunk_ids,
            'path': self.path,
            'status': self.status,
            'training_config': self.training_config,
            'metrics': self.metrics,
            'notes': self.notes
        }

    @staticmethod
    def from_dict(data: dict) -> 'LoRARecord':
        record = LoRARecord(data['lora_id'])
        record.created_at = data.get('created_at', datetime.now().isoformat())
        record.model_identity_hash = data.get('model_identity_hash')
        record.model_name = data.get('model_name')
        record.model_type = data.get('model_type')
        record.chunk_ids = data.get('chunk_ids', [])
        record.path = data.get('path')
        record.status = data.get('status', 'active')
        record.training_config = data.get('training_config', {})
        record.metrics = data.get('metrics', {})
        record.notes = data.get('notes', '')
        return record


class ChunkManager:
    """
    Manages the lifecycle of data chunks, LoRA adapters, and model switching.

    Directory structure:
    data/
    ├── chunks/
    │   ├── chunk_registry.json          # Master registry of all chunks
    │   ├── chunk_001/
    │   │   ├── manifest.json            # Chunk metadata
    │   │   └── documents.json           # Preserved document data
    │   ├── chunk_002/
    │   │   └── ...
    ├── loras/
    │   ├── lora_registry.json           # Master registry of all LoRAs
    │   ├── model_abc123/                # Organized by model identity
    │   │   ├── lora_001/
    │   │   │   ├── adapter_model.safetensors
    │   │   │   └── adapter_config.json
    │   │   └── lora_002/
    │   └── model_def456/
    │       └── ...
    ├── models/
    │   └── model_registry.json          # Known models and their identities
    └── history/
        └── operations.jsonl             # Operation log
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.chunks_dir = self.data_dir / 'chunks'
        self.loras_dir = self.data_dir / 'loras'
        self.models_dir = self.data_dir / 'models'
        self.history_dir = self.data_dir / 'history'

        # Create directories
        for d in [self.chunks_dir, self.loras_dir, self.models_dir, self.history_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load registries
        self.chunk_registry = self._load_registry(self.chunks_dir / 'chunk_registry.json')
        self.lora_registry = self._load_registry(self.loras_dir / 'lora_registry.json')
        self.model_registry = self._load_registry(self.models_dir / 'model_registry.json')

        # Current model tracking
        self.current_model = self.model_registry.get('current_model', None)
        self._chunk_counter = self.chunk_registry.get('_counter', 0)
        self._lora_counter = self.lora_registry.get('_counter', 0)

    def _load_registry(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_registry(self, path: Path, data: dict):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_chunk_registry(self):
        self.chunk_registry['_counter'] = self._chunk_counter
        self._save_registry(self.chunks_dir / 'chunk_registry.json', self.chunk_registry)

    def _save_lora_registry(self):
        self.lora_registry['_counter'] = self._lora_counter
        self._save_registry(self.loras_dir / 'lora_registry.json', self.lora_registry)

    def _save_model_registry(self):
        self._save_registry(self.models_dir / 'model_registry.json', self.model_registry)

    def _log_operation(self, operation: str, details: dict):
        """Append to operation history log"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        log_file = self.history_dir / 'operations.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    # ==================== MODEL MANAGEMENT ====================

    def register_model(self, model_path: str, friendly_name: str = None) -> Dict[str, Any]:
        """
        Register or identify a model. Returns model identity info.
        If model is already known, returns existing record.
        If model is new, creates new record and triggers model-switch logic.
        """
        identity = ModelFingerprint.identify_model(model_path)
        identity_hash = identity['identity_hash']

        if friendly_name:
            identity['friendly_name'] = friendly_name

        known_models = self.model_registry.get('known_models', {})
        is_new = identity_hash not in known_models

        # Check if model changed
        previous_model = self.current_model
        model_changed = previous_model and previous_model != identity_hash

        # Store model info
        if is_new:
            known_models[identity_hash] = {
                'identity': identity,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'friendly_name': friendly_name or identity['name'],
                'times_used': 1,
                'lora_ids': [],
                'compatible_lora_ids': []
            }
            logger.info(f"New model registered: {identity['name']} (hash: {identity_hash})")
        else:
            known_models[identity_hash]['last_seen'] = datetime.now().isoformat()
            known_models[identity_hash]['times_used'] = known_models[identity_hash].get('times_used', 0) + 1

        self.model_registry['known_models'] = known_models
        self.model_registry['current_model'] = identity_hash
        self.current_model = identity_hash
        self._save_model_registry()

        result = {
            'identity_hash': identity_hash,
            'identity': identity,
            'is_new': is_new,
            'model_changed': model_changed,
            'previous_model': previous_model,
            'friendly_name': known_models[identity_hash].get('friendly_name', identity['name'])
        }

        self._log_operation('model_registered', {
            'identity_hash': identity_hash,
            'name': identity['name'],
            'is_new': is_new,
            'model_changed': model_changed,
            'previous_model': previous_model
        })

        return result

    def detect_model_change(self, model_path: str) -> Dict[str, Any]:
        """
        Quick check if the model has changed since last time.
        Returns change info without full registration.
        """
        identity = ModelFingerprint.identify_model(model_path)
        identity_hash = identity['identity_hash']

        changed = self.current_model is not None and self.current_model != identity_hash
        known = identity_hash in self.model_registry.get('known_models', {})

        return {
            'changed': changed,
            'known': known,
            'current_hash': self.current_model,
            'new_hash': identity_hash,
            'new_name': identity['name']
        }

    def get_known_models(self) -> List[Dict]:
        """Get list of all known models"""
        known = self.model_registry.get('known_models', {})
        models = []
        for hash_id, info in known.items():
            model_info = {
                'identity_hash': hash_id,
                'name': info.get('friendly_name', info.get('identity', {}).get('name', 'Unknown')),
                'model_type': info.get('identity', {}).get('model_type', 'unknown'),
                'first_seen': info.get('first_seen'),
                'last_seen': info.get('last_seen'),
                'times_used': info.get('times_used', 0),
                'lora_count': len(info.get('lora_ids', [])),
                'is_current': hash_id == self.current_model
            }
            models.append(model_info)
        return sorted(models, key=lambda m: m.get('last_seen', ''), reverse=True)

    def get_compatible_loras(self, model_identity_hash: str = None) -> List[Dict]:
        """Get LoRAs compatible with the specified (or current) model"""
        target_hash = model_identity_hash or self.current_model
        if not target_hash:
            return []

        loras = []
        for lora_id, lora_data in self.lora_registry.items():
            if lora_id.startswith('_'):
                continue
            if isinstance(lora_data, dict) and lora_data.get('model_identity_hash') == target_hash:
                if lora_data.get('status') != 'deleted':
                    loras.append(lora_data)

        return sorted(loras, key=lambda l: l.get('created_at', ''), reverse=True)

    # ==================== CHUNK MANAGEMENT ====================

    def create_chunk(self, documents: List[dict], character: str = None,
                     metadata: dict = None) -> DataChunk:
        """
        Create a new data chunk from documents.
        Documents are preserved so they can be restored to RAG if needed.
        """
        self._chunk_counter += 1
        chunk_id = f"chunk_{self._chunk_counter:04d}"

        chunk = DataChunk(chunk_id)
        chunk.character = character
        chunk.document_ids = [doc.get('id', str(i)) for i, doc in enumerate(documents)]
        chunk.document_count = len(documents)
        chunk.model_identity_hash = self.current_model
        chunk.metadata = metadata or {}
        chunk.add_history('created', f'{len(documents)} documents')

        # Save chunk data
        chunk_dir = self.chunks_dir / chunk_id
        chunk_dir.mkdir(exist_ok=True)

        # Save manifest
        with open(chunk_dir / 'manifest.json', 'w') as f:
            json.dump(chunk.to_dict(), f, indent=2, default=str)

        # Preserve document data for potential restoration
        with open(chunk_dir / 'documents.json', 'w') as f:
            json.dump(documents, f, indent=2, default=str)

        # Update registry
        self.chunk_registry[chunk_id] = chunk.to_dict()
        self._save_chunk_registry()

        self._log_operation('chunk_created', {
            'chunk_id': chunk_id,
            'document_count': len(documents),
            'character': character,
            'model': self.current_model
        })

        logger.info(f"Created chunk {chunk_id} with {len(documents)} documents")
        return chunk

    def mark_chunk_training(self, chunk_id: str):
        """Mark a chunk as currently being trained"""
        self._update_chunk_status(chunk_id, ChunkStatus.TRAINING, 'Training started')

    def mark_chunk_trained(self, chunk_id: str, lora_id: str):
        """Mark a chunk as successfully trained"""
        if chunk_id in self.chunk_registry:
            self.chunk_registry[chunk_id]['lora_id'] = lora_id
        self._update_chunk_status(chunk_id, ChunkStatus.TRAINED, f'Trained into LoRA {lora_id}')

    def mark_chunk_failed(self, chunk_id: str, error: str = ''):
        """Mark a chunk as failed training - data remains available"""
        self._update_chunk_status(chunk_id, ChunkStatus.FAILED, f'Training failed: {error}')

    def mark_chunk_restored(self, chunk_id: str):
        """Mark a chunk as restored to RAG"""
        self._update_chunk_status(chunk_id, ChunkStatus.RESTORED, 'Data restored to RAG')

    def _update_chunk_status(self, chunk_id: str, status: str, details: str):
        if chunk_id not in self.chunk_registry:
            logger.warning(f"Chunk {chunk_id} not found in registry")
            return

        self.chunk_registry[chunk_id]['status'] = status
        history = self.chunk_registry[chunk_id].get('history', [])
        history.append({
            'timestamp': datetime.now().isoformat(),
            'action': status,
            'details': details
        })
        self.chunk_registry[chunk_id]['history'] = history
        self._save_chunk_registry()

        # Update manifest file
        chunk_dir = self.chunks_dir / chunk_id
        manifest_path = chunk_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'w') as f:
                json.dump(self.chunk_registry[chunk_id], f, indent=2, default=str)

        self._log_operation('chunk_status_changed', {
            'chunk_id': chunk_id,
            'new_status': status,
            'details': details
        })

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get chunk info"""
        return self.chunk_registry.get(chunk_id)

    def get_chunk_documents(self, chunk_id: str) -> Optional[List[dict]]:
        """Get preserved documents from a chunk"""
        doc_path = self.chunks_dir / chunk_id / 'documents.json'
        if doc_path.exists():
            with open(doc_path, 'r') as f:
                return json.load(f)
        return None

    def get_chunks_by_status(self, status: str = None, model_hash: str = None) -> List[dict]:
        """Get chunks filtered by status and/or model"""
        chunks = []
        for chunk_id, chunk_data in self.chunk_registry.items():
            if chunk_id.startswith('_'):
                continue
            if not isinstance(chunk_data, dict):
                continue
            if status and chunk_data.get('status') != status:
                continue
            if model_hash and chunk_data.get('model_identity_hash') != model_hash:
                continue
            chunk_data['chunk_id'] = chunk_id
            chunks.append(chunk_data)

        return sorted(chunks, key=lambda c: c.get('created_at', ''), reverse=True)

    def get_untrained_chunks(self) -> List[dict]:
        """Get all chunks that haven't been trained yet (pending or failed)"""
        result = []
        for chunk_id, chunk_data in self.chunk_registry.items():
            if chunk_id.startswith('_'):
                continue
            if not isinstance(chunk_data, dict):
                continue
            status = chunk_data.get('status')
            if status in [ChunkStatus.PENDING, ChunkStatus.FAILED, ChunkStatus.RESTORED]:
                chunk_data['chunk_id'] = chunk_id
                result.append(chunk_data)
        return result

    def get_restorable_chunks(self, model_identity_hash: str = None) -> List[dict]:
        """
        Get chunks whose data can be restored to RAG.
        These are chunks that were trained for a different model,
        or chunks that failed training.
        """
        target = model_identity_hash or self.current_model
        result = []

        for chunk_id, chunk_data in self.chunk_registry.items():
            if chunk_id.startswith('_'):
                continue
            if not isinstance(chunk_data, dict):
                continue

            status = chunk_data.get('status')
            chunk_model = chunk_data.get('model_identity_hash')

            # Restorable if:
            # 1. Trained for a different model (LoRA not compatible)
            # 2. Training failed
            # 3. Already restored but could be re-loaded
            can_restore = False

            if status == ChunkStatus.TRAINED and chunk_model != target:
                can_restore = True  # Trained for wrong model
            elif status == ChunkStatus.FAILED:
                can_restore = True  # Failed, data still available
            elif status == ChunkStatus.RESTORED:
                can_restore = True  # Already restored, can re-restore

            if can_restore:
                # Check if documents.json still exists
                doc_path = self.chunks_dir / chunk_id / 'documents.json'
                if doc_path.exists():
                    chunk_data['chunk_id'] = chunk_id
                    chunk_data['restore_reason'] = 'model_mismatch' if status == ChunkStatus.TRAINED else status
                    result.append(chunk_data)

        return result

    # ==================== LORA MANAGEMENT ====================

    def register_lora(self, chunk_ids: List[str], lora_path: str,
                      training_config: dict = None, metrics: dict = None) -> LoRARecord:
        """Register a new LoRA adapter"""
        self._lora_counter += 1
        lora_id = f"lora_{self._lora_counter:04d}"

        record = LoRARecord(lora_id)
        record.model_identity_hash = self.current_model
        record.chunk_ids = chunk_ids
        record.training_config = training_config or {}
        record.metrics = metrics or {}

        # Get model info
        known_models = self.model_registry.get('known_models', {})
        if self.current_model in known_models:
            model_info = known_models[self.current_model]
            record.model_name = model_info.get('friendly_name', '')
            record.model_type = model_info.get('identity', {}).get('model_type', 'unknown')

        # Move/copy LoRA to organized directory
        model_lora_dir = self.loras_dir / f"model_{self.current_model}" / lora_id
        model_lora_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(lora_path)
        if source_path.is_dir():
            for item in source_path.iterdir():
                shutil.copy2(str(item), str(model_lora_dir / item.name))
        elif source_path.is_file():
            shutil.copy2(str(source_path), str(model_lora_dir / source_path.name))

        record.path = str(model_lora_dir)

        # Update registries
        self.lora_registry[lora_id] = record.to_dict()
        self._save_lora_registry()

        # Update model registry
        if self.current_model in known_models:
            lora_ids = known_models[self.current_model].get('lora_ids', [])
            if lora_id not in lora_ids:
                lora_ids.append(lora_id)
            known_models[self.current_model]['lora_ids'] = lora_ids
            self.model_registry['known_models'] = known_models
            self._save_model_registry()

        # Mark chunks as trained
        for chunk_id in chunk_ids:
            self.mark_chunk_trained(chunk_id, lora_id)

        self._log_operation('lora_registered', {
            'lora_id': lora_id,
            'model': self.current_model,
            'chunk_ids': chunk_ids,
            'path': str(model_lora_dir)
        })

        logger.info(f"Registered LoRA {lora_id} for model {self.current_model}")
        return record

    def get_lora(self, lora_id: str) -> Optional[dict]:
        """Get LoRA info"""
        data = self.lora_registry.get(lora_id)
        if data and isinstance(data, dict) and not lora_id.startswith('_'):
            return data
        return None

    def get_all_loras(self, include_deleted: bool = False) -> List[dict]:
        """Get all LoRA records"""
        loras = []
        for lora_id, data in self.lora_registry.items():
            if lora_id.startswith('_'):
                continue
            if not isinstance(data, dict):
                continue
            if not include_deleted and data.get('status') == 'deleted':
                continue
            data['lora_id'] = lora_id
            loras.append(data)
        return sorted(loras, key=lambda l: l.get('created_at', ''), reverse=True)

    def mark_lora_unusable(self, lora_id: str, reason: str = 'Model changed'):
        """Mark a LoRA as unusable (e.g., model changed)"""
        if lora_id in self.lora_registry:
            self.lora_registry[lora_id]['status'] = 'unusable'
            self.lora_registry[lora_id]['unusable_reason'] = reason
            self.lora_registry[lora_id]['marked_unusable_at'] = datetime.now().isoformat()
            self._save_lora_registry()

            self._log_operation('lora_marked_unusable', {
                'lora_id': lora_id,
                'reason': reason
            })

    def delete_lora(self, lora_id: str, delete_files: bool = False) -> bool:
        """Delete or archive a LoRA"""
        if lora_id not in self.lora_registry:
            return False

        lora_data = self.lora_registry[lora_id]

        if delete_files and lora_data.get('path'):
            lora_path = Path(lora_data['path'])
            if lora_path.exists():
                shutil.rmtree(str(lora_path))
                logger.info(f"Deleted LoRA files: {lora_path}")

        self.lora_registry[lora_id]['status'] = 'deleted'
        self.lora_registry[lora_id]['deleted_at'] = datetime.now().isoformat()
        self._save_lora_registry()

        self._log_operation('lora_deleted', {
            'lora_id': lora_id,
            'files_deleted': delete_files
        })

        return True

    def select_lora(self, lora_id: str) -> Optional[dict]:
        """
        Select a LoRA for use. Validates compatibility with current model.
        Returns LoRA info if compatible, None if not.
        """
        lora_data = self.get_lora(lora_id)
        if not lora_data:
            return None

        if lora_data.get('status') == 'deleted':
            return None

        # Check compatibility
        lora_model = lora_data.get('model_identity_hash')
        if lora_model and lora_model != self.current_model:
            logger.warning(f"LoRA {lora_id} was trained for model {lora_model}, current model is {self.current_model}")
            return {
                'compatible': False,
                'lora': lora_data,
                'reason': f'LoRA trained for different model (trained: {lora_data.get("model_name", "unknown")}, current: {self.current_model})'
            }

        return {
            'compatible': True,
            'lora': lora_data,
            'path': lora_data.get('path')
        }

    # ==================== MODEL SWITCH HANDLING ====================

    def handle_model_switch(self, new_model_path: str, friendly_name: str = None) -> Dict[str, Any]:
        """
        Handle a model switch. This is the main entry point when the user changes models.

        Returns information about:
        - What changed
        - Which LoRAs became unusable
        - Which chunks can be restored to RAG
        - Which LoRAs are available for the new model
        """
        previous_model = self.current_model

        # Register the new model
        reg_result = self.register_model(new_model_path, friendly_name)

        if not reg_result['model_changed']:
            return {
                'changed': False,
                'message': 'Same model detected, no changes needed',
                'model': reg_result
            }

        new_model_hash = reg_result['identity_hash']

        # Find LoRAs that became unusable
        unusable_loras = []
        if previous_model:
            for lora_id, lora_data in self.lora_registry.items():
                if lora_id.startswith('_') or not isinstance(lora_data, dict):
                    continue
                if lora_data.get('model_identity_hash') == previous_model and lora_data.get('status') == 'active':
                    self.mark_lora_unusable(lora_id, f'Model changed from {previous_model} to {new_model_hash}')
                    unusable_loras.append(lora_data)

        # Find chunks that can be restored to RAG
        restorable_chunks = self.get_restorable_chunks(new_model_hash)

        # Find compatible LoRAs for the new model
        compatible_loras = self.get_compatible_loras(new_model_hash)

        result = {
            'changed': True,
            'previous_model': previous_model,
            'new_model': new_model_hash,
            'model_info': reg_result,
            'unusable_loras': [l.get('lora_id', '') for l in unusable_loras],
            'unusable_lora_count': len(unusable_loras),
            'restorable_chunks': [c.get('chunk_id', '') for c in restorable_chunks],
            'restorable_chunk_count': len(restorable_chunks),
            'restorable_document_count': sum(c.get('document_count', 0) for c in restorable_chunks),
            'compatible_loras': [l.get('lora_id', '') for l in compatible_loras],
            'compatible_lora_count': len(compatible_loras),
            'action_needed': len(restorable_chunks) > 0
        }

        self._log_operation('model_switch', result)
        logger.info(f"Model switched: {previous_model} -> {new_model_hash}. "
                     f"{len(unusable_loras)} LoRAs unusable, "
                     f"{len(restorable_chunks)} chunks restorable")

        return result

    def restore_chunks_to_rag(self, chunk_ids: List[str] = None) -> Dict[str, Any]:
        """
        Restore chunk data back to RAG system.
        If no chunk_ids specified, restores all restorable chunks.

        Returns the documents to be re-ingested into the RAG system.
        The caller (API layer) is responsible for actually inserting them.
        """
        if chunk_ids is None:
            restorable = self.get_restorable_chunks()
            chunk_ids = [c['chunk_id'] for c in restorable]

        all_documents = []
        restored_chunks = []
        failed_chunks = []

        for chunk_id in chunk_ids:
            documents = self.get_chunk_documents(chunk_id)
            if documents:
                all_documents.extend(documents)
                self.mark_chunk_restored(chunk_id)
                restored_chunks.append(chunk_id)
            else:
                failed_chunks.append(chunk_id)
                logger.warning(f"Could not restore chunk {chunk_id}: documents not found")

        result = {
            'restored_chunks': restored_chunks,
            'failed_chunks': failed_chunks,
            'total_documents': len(all_documents),
            'documents': all_documents
        }

        self._log_operation('chunks_restored', {
            'restored': restored_chunks,
            'failed': failed_chunks,
            'document_count': len(all_documents)
        })

        return result

    # ==================== STATUS & OVERVIEW ====================

    def get_overview(self) -> Dict[str, Any]:
        """Get complete system overview"""
        chunks_by_status = {}
        total_chunks = 0
        for chunk_id, data in self.chunk_registry.items():
            if chunk_id.startswith('_') or not isinstance(data, dict):
                continue
            total_chunks += 1
            status = data.get('status', 'unknown')
            chunks_by_status[status] = chunks_by_status.get(status, 0) + 1

        loras_by_status = {}
        total_loras = 0
        for lora_id, data in self.lora_registry.items():
            if lora_id.startswith('_') or not isinstance(data, dict):
                continue
            total_loras += 1
            status = data.get('status', 'unknown')
            loras_by_status[status] = loras_by_status.get(status, 0) + 1

        return {
            'current_model': self.current_model,
            'current_model_name': self._get_model_name(self.current_model),
            'known_models': len(self.model_registry.get('known_models', {})),
            'total_chunks': total_chunks,
            'chunks_by_status': chunks_by_status,
            'total_loras': total_loras,
            'loras_by_status': loras_by_status,
            'compatible_loras': len(self.get_compatible_loras()),
            'restorable_chunks': len(self.get_restorable_chunks()),
            'untrained_chunks': len(self.get_untrained_chunks())
        }

    def _get_model_name(self, model_hash: str) -> str:
        if not model_hash:
            return 'None'
        known = self.model_registry.get('known_models', {})
        if model_hash in known:
            return known[model_hash].get('friendly_name', model_hash[:8])
        return model_hash[:8]

    def get_operation_history(self, limit: int = 100) -> List[dict]:
        """Get recent operation history"""
        log_file = self.history_dir / 'operations.jsonl'
        if not log_file.exists():
            return []

        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except:
                        pass

        return entries[-limit:]