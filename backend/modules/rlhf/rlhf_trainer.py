"""
RLHF Trainer - Orchestrates Reinforcement Learning from Human Feedback.
Uses DPO (Direct Preference Optimization) via TRL library.
"""
import logging
from typing import List, Dict, Optional
logger = logging.getLogger("RLHFTrainer")

class RLHFTrainer:
    """
    Implements RLHF training using Direct Preference Optimization.
    Integrates with the LoRA trainer for efficient fine-tuning.
    """
    def __init__(self, config: dict, device_manager, reward_model):
        self.config = config
        self.device_manager = device_manager
        self.reward_model = reward_model
    def prepare_rlhf_data(self, feedback_data: List[Dict]) -> Optional[object]:
        """Prepare feedback data for DPO training."""
        dpo_pairs = self.reward_model.prepare_dpo_dataset(feedback_data)
        if not dpo_pairs:
            return None
        from datasets import Dataset
        # Filter out entries with empty rejected (SFT-only entries handled separately)
        dpo_entries = [d for d in dpo_pairs if d.get('rejected')]
        if not dpo_entries:
            return None
        return Dataset.from_list(dpo_entries)
