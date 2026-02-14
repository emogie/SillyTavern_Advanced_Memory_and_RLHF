"""
LoRA Trainer - Fine-tunes LoRA adapters from collected memory and RLHF data.
Handles backup, training, saving, archiving, and auto-loading of LoRA models.
"""
import os
import json
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional
from modules.training.device_manager import DeviceManager
from modules.training.progress_tracker import ProgressTracker

logger = logging.getLogger("LoRATrainer")


class LoRATrainingCallback:
    """Custom callback for progress tracking during training."""
    def __init__(self, progress_tracker: ProgressTracker, total_steps: int):
        self.tracker = progress_tracker
        self.total_steps = total_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get('loss', 0.0)
            step = state.global_step
            epoch = int(state.epoch) if state.epoch else 0
            self.tracker.update(epoch, step, loss)
        if self.tracker.should_cancel:
            control.should_training_stop = True
            return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.tracker.should_cancel:
            control.should_training_stop = True
        return control


class LoRATrainer:
    """
    Manages the full LoRA training lifecycle:
    1. Backup existing LoRA (if any)
    2. Prepare training data from vector DB + RLHF feedback
    3. Fine-tune LoRA adapter
    4. Save new LoRA
    5. Archive training data (feedback + chunks)
    6. Auto-load LoRA for inference
    """

    def __init__(self, config: dict, device_manager: DeviceManager,
                 progress_tracker: ProgressTracker):
        self.config = config
        self.device_manager = device_manager
        self.progress_tracker = progress_tracker

        self.models_path = Path(config.get('lora_models_path', 'data/lora_models'))
        self.backups_path = Path(config.get('lora_backups_path', 'data/lora_backups'))
        self.archive_path = Path(config.get('trained_archive_path', 'data/trained_archive'))
        self.feedback_path = Path(config.get('feedback_path', 'data/feedback'))

        self.models_path.mkdir(parents=True, exist_ok=True)
        self.backups_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.current_lora_path = self.models_path / "current_lora"
        self.auto_loaded = False

        # Track what was used for training
        self._last_training_stats = {}

    def get_lora_status(self) -> Dict:
        """Check if a LoRA adapter is available."""
        adapter_config = self.current_lora_path / "adapter_config.json"
        available = adapter_config.exists()
        model_name = None
        if available:
            try:
                with open(adapter_config) as f:
                    cfg = json.load(f)
                    model_name = cfg.get('base_model_name_or_path', 'unknown')
            except Exception:
                model_name = 'unknown'

        return {
            "available": available,
            "model_name": model_name,
            "path": str(self.current_lora_path) if available else None,
            "auto_loaded": self.auto_loaded,
            "last_training": self._last_training_stats
        }

    def auto_load_lora(self):
        """Auto-load LoRA adapter if available."""
        status = self.get_lora_status()
        if status['available']:
            self.auto_loaded = True
            logger.info(f"LoRA adapter auto-loaded from {self.current_lora_path}")
        else:
            logger.info("No LoRA adapter found for auto-loading")

    def _backup_existing_lora(self):
        """Backup existing LoRA before training."""
        if self.current_lora_path.exists():
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backups_path / f"lora_backup_{timestamp}"
            shutil.copytree(self.current_lora_path, backup_dir)
            logger.info(f"LoRA backed up to {backup_dir}")
            return str(backup_dir)
        return None

    def _archive_training_data(self, training_example_count: int,
                                feedback_example_count: int,
                                training_info: Dict):
        """
        Archive feedback data after successful training.
        - Moves feedback files to trained_archive/
        - Saves a training manifest
        - Does NOT delete memory data (it stays in vector DB for RAG)
        """
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        archive_dir = self.archive_path / f"training_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        archived_files = []
        archived_feedback_count = 0

        # Archive feedback files
        if self.feedback_path.exists():
            feedback_archive = archive_dir / "feedback"
            feedback_archive.mkdir(exist_ok=True)

            for feedback_file in sorted(self.feedback_path.glob("*.json")):
                try:
                    # Copy to archive
                    dest = feedback_archive / feedback_file.name
                    shutil.copy2(feedback_file, dest)
                    archived_files.append(feedback_file.name)

                    # Count entries
                    try:
                        with open(feedback_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                archived_feedback_count += len(data)
                            elif isinstance(data, dict):
                                # Some formats store feedback as dict entries
                                archived_feedback_count += len(data.get('entries', [data]))
                    except Exception:
                        pass

                    # Remove original after successful copy
                    feedback_file.unlink()
                    logger.info(f"Archived and removed: {feedback_file.name}")

                except Exception as e:
                    logger.warning(f"Could not archive {feedback_file.name}: {e}")

        # Save training manifest
        manifest = {
            "training_timestamp": timestamp,
            "training_info": training_info,
            "archived_feedback_files": archived_files,
            "archived_feedback_entries": archived_feedback_count,
            "training_examples_used": training_example_count,
            "feedback_examples_used": feedback_example_count,
            "archive_dir": str(archive_dir)
        }

        manifest_path = archive_dir / "training_manifest.json"
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"Training manifest saved: {manifest_path}")
        except Exception as e:
            logger.warning(f"Could not save manifest: {e}")

        # Update last training stats
        self._last_training_stats = {
            "timestamp": timestamp,
            "training_examples": training_example_count,
            "feedback_examples": feedback_example_count,
            "archived_files": len(archived_files),
            "archived_feedback_entries": archived_feedback_count,
            "archive_path": str(archive_dir)
        }

        logger.info(
            f"Training data archived: {len(archived_files)} feedback files "
            f"({archived_feedback_count} entries) → {archive_dir}"
        )

        return manifest

    def _prepare_training_dataset(self, training_data: List[Dict],
                                   feedback_data: List[Dict]) -> object:
        """Convert raw data into a training dataset."""
        from datasets import Dataset

        training_examples = []

        # Process vector DB data (conversation format)
        for doc in training_data:
            text = doc.get('text', '')
            if not text.strip():
                continue
            training_examples.append({
                "text": text,
                "source": "memory"
            })

        # Process RLHF positive feedback
        for fb in feedback_data:
            if fb.get('score', 0) > 0:
                prompt = fb.get('prompt_text', '')
                response = fb.get('response_text', '')
                if prompt and response:
                    training_examples.append({
                        "text": f"### Human: {prompt}\n### Assistant: {response}",
                        "source": "rlhf_positive"
                    })

        if not training_examples:
            raise ValueError("No training data available")

        logger.info(f"Prepared {len(training_examples)} training examples "
                    f"({sum(1 for e in training_examples if e['source'] == 'memory')} memory, "
                    f"{sum(1 for e in training_examples if e['source'] == 'rlhf_positive')} RLHF)")

        return Dataset.from_list(training_examples)

    def train(self, training_data: List[Dict], feedback_data: List[Dict],
              user_config: Dict):
        """
        Main training method. Runs in a background thread.
        """
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
                TrainerCallback
            )
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
                TaskType
            )

            logger.info("Starting LoRA training pipeline...")

            # 1. Backup existing LoRA
            self._backup_existing_lora()

            # 2. Prepare dataset
            dataset = self._prepare_training_dataset(training_data, feedback_data)

            # Count sources for archiving later
            memory_count = sum(1 for e in dataset if e.get('source') == 'memory')
            rlhf_count = sum(1 for e in dataset if e.get('source') == 'rlhf_positive')

            # 3. Configuration
            base_model = user_config.get('base_model',
                                          self.config.get('default_base_model'))
            epochs = user_config.get('epochs', self.config.get('default_epochs', 3))
            lr = user_config.get('learning_rate',
                                 self.config.get('default_learning_rate', 2e-4))
            lora_rank = user_config.get('lora_rank',
                                        self.config.get('default_lora_rank', 16))
            batch_size = user_config.get('batch_size',
                                         self.config.get('default_batch_size', 4))
            max_seq_len = self.config.get('max_seq_length', 2048)

            # 4. Load tokenizer
            logger.info(f"Loading tokenizer: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 5. Tokenize dataset
            def tokenize_fn(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=max_seq_len,
                    padding='max_length'
                )

            tokenized_dataset = dataset.map(tokenize_fn, batched=True,
                                            remove_columns=dataset.column_names)

            # 6. Load model
            logger.info(f"Loading base model: {base_model}")
            device = self.device_manager.get_torch_device()
            device_args = self.device_manager.get_training_args()

            model_kwargs = {"trust_remote_code": True}

            # Quantization for memory efficiency
            if device.type == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            elif device.type == "mps":
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

            if device.type == "cuda":
                try:
                    model = prepare_model_for_kbit_training(model)
                except Exception:
                    pass

            # 7. LoRA Configuration
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=self.config.get('default_lora_alpha', 32),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            model = get_peft_model(model, lora_config)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.2f}%)")

            # 8. Calculate total steps
            total_steps = (len(tokenized_dataset) // batch_size) * epochs
            self.progress_tracker.start(epochs, total_steps)

            # 9. Training Arguments
            output_dir = str(self.models_path / "training_output")
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=self.config.get(
                    'gradient_accumulation_steps', 4
                ),
                learning_rate=lr,
                warmup_ratio=self.config.get('warmup_ratio', 0.03),
                logging_steps=5,
                save_steps=self.config.get('save_steps', 100),
                save_total_limit=2,
                optim="adamw_torch",
                report_to="none",
                remove_unused_columns=False,
                **device_args
            )

            # 10. Create custom callback for progress
            class ProgressCallback(TrainerCallback):
                def __init__(self, tracker):
                    self.tracker = tracker

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        loss = logs.get('loss', logs.get('train_loss', 0.0))
                        epoch = int(state.epoch) if state.epoch else 0
                        self.tracker.update(epoch, state.global_step, loss)
                    if self.tracker.should_cancel:
                        control.should_training_stop = True
                    return control

                def on_step_end(self, args, state, control, **kwargs):
                    if self.tracker.should_cancel:
                        control.should_training_stop = True
                    return control

            # 11. Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )

            # 12. Train
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                callbacks=[ProgressCallback(self.progress_tracker)]
            )

            logger.info("Training started...")
            train_result = trainer.train()

            if self.progress_tracker.should_cancel:
                logger.info("Training was cancelled by user")
                self.progress_tracker.fail("Cancelled by user")
                return

            # 13. Save LoRA adapter
            logger.info("Saving LoRA adapter...")
            if self.current_lora_path.exists():
                shutil.rmtree(self.current_lora_path)

            model.save_pretrained(str(self.current_lora_path))
            tokenizer.save_pretrained(str(self.current_lora_path))

            # Save training info
            training_info = {
                "base_model": base_model,
                "epochs": epochs,
                "learning_rate": lr,
                "lora_rank": lora_rank,
                "batch_size": batch_size,
                "trained_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "training_examples": len(dataset),
                "memory_examples": memory_count,
                "rlhf_examples": rlhf_count,
                "final_loss": train_result.training_loss,
                "device": self.device_manager.device_name
            }

            with open(self.current_lora_path / "training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)

            # 14. Archive training data (NEW!)
            logger.info("Archiving training data...")
            try:
                self._archive_training_data(
                    training_example_count=len(dataset),
                    feedback_example_count=rlhf_count,
                    training_info=training_info
                )
            except Exception as archive_err:
                logger.warning(f"Archiving failed (non-fatal): {archive_err}")
                # Training still succeeded, archiving is best-effort

            # 15. Mark chunks as trained (if chunk system available)
            try:
                from chunk_manager import ChunkManager
                chunk_data_dir = Path(self.config.get('data_dir', 'data'))
                # Try to find and update chunk manager
                chunks_file = chunk_data_dir / "chunks" / "chunks.json"
                if chunks_file.exists():
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)

                    updated = 0
                    for chunk_id, chunk in chunks.items():
                        if chunk.get('status') in ('pending', 'untrained'):
                            chunk['status'] = 'trained'
                            chunk['trained_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                            chunk['lora_id'] = f"lora_{time.strftime('%Y%m%d_%H%M%S')}"
                            updated += 1

                    if updated > 0:
                        with open(chunks_file, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, indent=2, ensure_ascii=False)
                        logger.info(f"Marked {updated} chunks as trained")
            except Exception as chunk_err:
                logger.debug(f"Chunk marking skipped: {chunk_err}")

            # 16. Cleanup training output
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            self.auto_loaded = True
            self.progress_tracker.complete()
            logger.info("LoRA training completed successfully!")

            logger.info(
                f"Summary: {len(dataset)} examples trained, "
                f"feedback archived to {self.archive_path}, "
                f"LoRA saved to {self.current_lora_path}"
            )

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.progress_tracker.fail(str(e))