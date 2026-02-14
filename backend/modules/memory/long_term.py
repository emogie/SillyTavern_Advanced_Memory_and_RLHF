"""
Long-Term Memory - LoRA adapter represents consolidated long-term knowledge.
Created through fine-tuning on accumulated middle-term memory data.
"""
class LongTermMemory:
    """
    Represents long-term memory through a trained LoRA adapter.
    The adapter is automatically loaded with the base model.
    """
    def __init__(self, lora_path: str):
        self.lora_path = lora_path
        self.is_loaded = False
    def is_available(self) -> bool:
        import os
        return os.path.exists(os.path.join(self.lora_path, 'adapter_config.json'))
    def get_adapter_path(self) -> str:
        return self.lora_path if self.is_available() else None
