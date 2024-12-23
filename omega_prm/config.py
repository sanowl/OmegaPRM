from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class OmegaPRMConfig:
    """Configuration class for OmegaPRM hyperparameters and settings"""
    model_name: str = "gpt2"
    search_limit: int = 100
    alpha: float = 0.5
    beta: float = 0.9
    L: int = 500
    cpuct: float = 0.125
    batch_size: int = 32
    learning_rate: float = 0.001
    hidden_size: int = 256
    max_length: int = 512
    cache_size: int = 10000
    num_workers: int = 4
    use_wandb: bool = False
    checkpoint_dir: str = "checkpoints"
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert 0 <= self.beta <= 1, "Beta must be between 0 and 1"
        assert self.search_limit > 0, "Search limit must be positive"
        assert self.L > 0, "L must be positive"
        assert self.cpuct > 0, "cpuct must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        
        # Create checkpoint directory if it doesn't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)