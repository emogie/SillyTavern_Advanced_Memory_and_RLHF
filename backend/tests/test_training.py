"""Tests for training components."""
import pytest
def test_device_manager():
    """Test device detection."""
    from modules.training.device_manager import DeviceManager
    dm = DeviceManager()
    assert dm.device_type in ['cuda', 'mps', 'cpu']
    assert dm.device_name != ''
    assert 'cpu' in dm.supported_backends
    info = dm.get_device_info()
    assert 'device_type' in info
    assert 'device_name' in info
def test_progress_tracker():
    """Test progress tracking."""
    from modules.training.progress_tracker import ProgressTracker
    tracker = ProgressTracker()
    assert not tracker.is_running
    tracker.start(3, 100)
    assert tracker.is_running
    tracker.update(1, 50, 0.5)
    progress = tracker.get_progress()
    assert progress['percentage'] == 50.0
    assert progress['current_epoch'] == 1
    assert progress['loss'] == 0.5
    tracker.complete()
    assert not tracker.is_running
    assert tracker.get_progress()['status'] == 'completed'