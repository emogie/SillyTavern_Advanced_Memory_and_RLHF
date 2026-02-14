"""
Reward Model - Simple reward model trained on human feedback.
Used to score responses during training.
"""
import logging
from typing import List, Dict
logger = logging.getLogger("RewardModel")

class RewardModel:
    """
    Simple reward model based on human feedback scores.
    For full RLHF, this would be a trained neural network.
    Here we use a simplified scoring approach compatible with TRL's DPO trainer.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model = None
    def prepare_dpo_dataset(self, feedback_data: List[Dict]) -> List[Dict]:
        """
        Convert feedback into DPO (Direct Preference Optimization) format.
        Each entry has: prompt, chosen, rejected
        """
        dpo_data = []
        # Group by prompt_text
        from collections import defaultdict
        groups = defaultdict(list)
        for fb in feedback_data:
            prompt = fb.get('prompt_text', '').strip()
            if prompt:
                groups[prompt].append(fb)
        for prompt, responses in groups.items():
            positive = [r for r in responses if r.get('score', 0) > 0]
            negative = [r for r in responses if r.get('score', 0) < 0]
            for pos in positive:
                for neg in negative:
                    dpo_data.append({
                        "prompt": prompt,
                        "chosen": pos.get('response_text', ''),
                        "rejected": neg.get('response_text', '')
                    })
        # If not enough pairs, also include single positive samples for SFT
        if len(dpo_data) < 10:
            for fb in feedback_data:
                if fb.get('score', 0) > 0:
                    dpo_data.append({
                        "prompt": fb.get('prompt_text', ''),
                        "chosen": fb.get('response_text', ''),
                        "rejected": ""  # Empty = SFT only
                    })
        logger.info(f"Prepared {len(dpo_data)} DPO training pairs")
        return dpo_data
