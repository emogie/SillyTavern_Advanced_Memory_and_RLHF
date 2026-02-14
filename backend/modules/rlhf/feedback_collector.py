"""
RLHF Feedback Collector - Stores human feedback for training.
Collects thumbs up/down/star ratings on AI responses.
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
logger = logging.getLogger("FeedbackCollector")

class FeedbackCollector:
    """
    Collects and stores human feedback data for RLHF training.
    Feedback is stored as JSON files organized by session.
    """
    def __init__(self, config: dict):
        self.config = config
        self.feedback_path = Path(config.get('feedback_path', 'data/feedback'))
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        self.min_samples = config.get('min_feedback_samples', 100)
        self._feedback_cache: List[Dict] = []
        self._load_existing()
    def _load_existing(self):
        """Load existing feedback from disk."""
        self._feedback_cache = []
        for f in self.feedback_path.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        self._feedback_cache.extend(data)
                    else:
                        self._feedback_cache.append(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load feedback file {f}: {e}")
        logger.info(f"Loaded {len(self._feedback_cache)} existing feedback entries")
    def store_feedback(self, feedback: Dict[str, Any]) -> Dict:
        """Store a single feedback entry."""
        feedback['stored_at'] = time.time()
        # Convert rating to numeric score
        rating_scores = {
            'positive': 1.0,
            'negative': -1.0,
            'excellent': 2.0
        }
        feedback['score'] = rating_scores.get(feedback.get('rating'), 0.0)
        self._feedback_cache.append(feedback)
        # Save to file (append to current session file)
        session_file = self.feedback_path / f"feedback_{time.strftime('%Y%m%d')}.json"
        existing = []
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, Exception):
                existing = []
        existing.append(feedback)
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        logger.info(f"Feedback stored: {feedback.get('rating')} for message {feedback.get('message_id')}")
        return {
            "status": "ok",
            "total_feedback": len(self._feedback_cache),
            "ready_for_rlhf": len(self._feedback_cache) >= self.min_samples
        }
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        ratings = Counter(f.get('rating', 'unknown') for f in self._feedback_cache)
        return {
            "positive": ratings.get('positive', 0),
            "negative": ratings.get('negative', 0),
            "excellent": ratings.get('excellent', 0),
            "total": len(self._feedback_cache),
            "ready_for_rlhf": len(self._feedback_cache) >= self.min_samples
        }
    def get_feedback_data(self) -> List[Dict]:
        """Get all feedback data for RLHF training."""
        return self._feedback_cache.copy()
    def get_preference_pairs(self) -> List[Dict]:
        """
        Generate preference pairs for DPO/RLHF training.
        Groups feedback by prompt and creates chosen/rejected pairs.
        """
        # Group by prompt
        by_prompt = {}
        for fb in self._feedback_cache:
            prompt = fb.get('prompt_text', '').strip()
            if not prompt:
                continue
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(fb)
        pairs = []
        for prompt, feedbacks in by_prompt.items():
            # Sort by score
            sorted_fb = sorted(feedbacks, key=lambda x: x.get('score', 0), reverse=True)
            if len(sorted_fb) >= 2:
                best = sorted_fb[0]
                worst = sorted_fb[-1]
                if best.get('score', 0) > worst.get('score', 0):
                    pairs.append({
                        "prompt": prompt,
                        "chosen": best.get('response_text', ''),
                        "rejected": worst.get('response_text', ''),
                        "chosen_score": best.get('score', 0),
                        "rejected_score": worst.get('score', 0)
                    })
        return pairs
