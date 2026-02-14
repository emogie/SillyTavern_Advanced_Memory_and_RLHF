"""
Training Progress Tracker - Real-time training progress with ETA.
"""
import time
import threading
import logging
logger = logging.getLogger("ProgressTracker")

class ProgressTracker:
    """Tracks training progress and provides ETA estimates."""
    def __init__(self):
        self._lock = threading.Lock()
        self._progress = {
            "status": "idle",
            "percentage": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_step": 0,
            "total_steps": 0,
            "loss": 0.0,
            "eta": "",
            "error": "",
            "start_time": None,
        }
        self.is_running = False
        self._cancel_requested = False
    def start(self, total_epochs: int, total_steps: int):
        with self._lock:
            self._progress = {
                "status": "running",
                "percentage": 0,
                "current_epoch": 0,
                "total_epochs": total_epochs,
                "current_step": 0,
                "total_steps": total_steps,
                "loss": 0.0,
                "eta": "calculating...",
                "error": "",
                "start_time": time.time(),
            }
            self.is_running = True
            self._cancel_requested = False
    def update(self, epoch: int, step: int, loss: float):
        with self._lock:
            self._progress["current_epoch"] = epoch
            self._progress["current_step"] = step
            self._progress["loss"] = loss
            total = self._progress["total_steps"]
            if total > 0:
                self._progress["percentage"] = min(100, round((step / total) * 100, 1))
            # Calculate ETA
            start = self._progress.get("start_time")
            if start and step > 0:
                elapsed = time.time() - start
                rate = step / elapsed
                remaining_steps = total - step
                if rate > 0:
                    eta_seconds = remaining_steps / rate
                    if eta_seconds < 60:
                        self._progress["eta"] = f"{int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        self._progress["eta"] = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:
                        hours = int(eta_seconds // 3600)
                        mins = int((eta_seconds % 3600) // 60)
                        self._progress["eta"] = f"{hours}h {mins}m"
    def complete(self):
        with self._lock:
            self._progress["status"] = "completed"
            self._progress["percentage"] = 100
            self._progress["eta"] = "Done!"
            self.is_running = False
    def fail(self, error: str):
        with self._lock:
            self._progress["status"] = "failed"
            self._progress["error"] = error
            self.is_running = False
    def cancel(self):
        self._cancel_requested = True
    @property
    def should_cancel(self) -> bool:
        return self._cancel_requested
    def get_progress(self) -> dict:
        with self._lock:
            return self._progress.copy()
