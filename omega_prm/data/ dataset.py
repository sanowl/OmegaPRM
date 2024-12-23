import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
from transformers import PreTrainedTokenizer
import numpy as np
from collections import Counter
import random
import string
import re

class PRMDataset(Dataset):
    """Dataset class for Process Reward Model training with advanced augmentation"""
    
    def __init__(
        self,
        solutions: List[str],
        rewards: List[float],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        augment: bool = False,
        augment_prob: float = 0.1,
        min_length: Optional[int] = None,
        cache_encodings: bool = True
    ):
        """
        Initialize PRM Dataset with enhanced features
        
        Args:
            solutions (List[str]): List of solution strings
            rewards (List[float]): List of reward values
            tokenizer (PreTrainedTokenizer): Tokenizer for text processing
            max_length (int): Maximum sequence length
            augment (bool): Whether to use data augmentation
            augment_prob (float): Probability of applying augmentation
            min_length (Optional[int]): Minimum sequence length
            cache_encodings (bool): Whether to cache tokenized encodings
        """
        assert len(solutions) == len(rewards), "Solutions and rewards must have same length"
        assert 0 <= augment_prob <= 1, "Augmentation probability must be between 0 and 1"
        
        self.solutions = solutions
        self.rewards = rewards
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length or 0
        self.augment = augment
        self.augment_prob = augment_prob
        self.cache_encodings = cache_encodings
        
        # Initialize encoding cache
        self._encoding_cache = {} if cache_encodings else None
        
        # Preprocess solutions
        self.processed_solutions = self._preprocess_solutions()
        
        # Calculate vocabulary for augmentation
        self.vocab = self._build_vocabulary()

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.solutions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a dataset item with caching and augmentation
        
        Args:
            idx (int): Index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input tensor and reward tensor
        """
        solution = self.processed_solutions[idx]
        reward = self.rewards[idx]
        
        # Apply augmentation with probability
        if self.augment and random.random() < self.augment_prob:
            solution = self._augment_text(solution)
        
        # Try to get from cache first
        if self.cache_encodings and solution in self._encoding_cache:
            inputs = self._encoding_cache[solution]
        else:
            inputs = self.tokenizer.encode(
                solution,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            if self.cache_encodings:
                self._encoding_cache[solution] = inputs
        
        return inputs.squeeze(), torch.tensor([reward], dtype=torch.float32)

    def _preprocess_solutions(self) -> List[str]:
        """Preprocess all solutions"""
        return [self._preprocess_text(sol) for sol in self.solutions]

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower()  # Lowercase
        return text

    def _build_vocabulary(self) -> Dict[str, List[str]]:
        """Build vocabulary from solutions for augmentation"""
        words = []
        for solution in self.processed_solutions:
            words.extend(solution.split())
        
        # Create word frequency dictionary
        word_freq = Counter(words)
        
        # Group words by length for substitution
        vocab_by_length = {}
        for word in word_freq:
            length = len(word)
            if length not in vocab_by_length:
                vocab_by_length[length] = []
            vocab_by_length[length].append(word)
        
        return vocab_by_length

    def _augment_text(self, text: str) -> str:
        """Apply text augmentation techniques"""
        augmentation_types = [
            self._insert_typo,
            self._swap_words,
            self._substitute_word,
            self._delete_char,
            self._insert_char
        ]
        
        augmenter = random.choice(augmentation_types)
        return augmenter(text)

    def _insert_typo(self, text: str) -> str:
        """Insert random typo"""
        if not text:
            return text
            
        words = text.split()
        if not words:
            return text
            
        # Choose random word
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        if len(word) < 2:
            return text
            
        # Choose random position
        pos = random.randint(0, len(word) - 1)
        
        # Common keyboard typos
        typos = {
            'a': ['s', 'q', 'w'],
            'b': ['v', 'n', 'h'],
            'c': ['x', 'v', 'd'],
            'd': ['s', 'f', 'c'],
            'e': ['w', 'r', 'd'],
            'f': ['d', 'g', 'v'],
            'g': ['f', 'h', 'b'],
            'h': ['g', 'j', 'n'],
            'i': ['u', 'o', 'k'],
            'j': ['h', 'k', 'm'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'p', 'o'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'm', 'h'],
            'o': ['i', 'p', 'l'],
            'p': ['o', 'l', '['],
            'q': ['w', 'a', '1'],
            'r': ['e', 't', 'f'],
            's': ['a', 'd', 'w'],
            't': ['r', 'y', 'g'],
            'u': ['y', 'i', 'h'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', 's'],
            'x': ['z', 'c', 's'],
            'y': ['t', 'u', 'h'],
            'z': ['x', 'a', 's']
        }
        
        if word[pos] in typos:
            new_char = random.choice(typos[word[pos]])
            word = word[:pos] + new_char + word[pos + 1:]
            words[word_idx] = word
            
        return ' '.join(words)

    def _swap_words(self, text: str) -> str:
        """Swap adjacent words"""
        words = text.split()
        if len(words) < 2:
            return text
            
        pos = random.randint(0, len(words) - 2)
        words[pos], words[pos + 1] = words[pos + 1], words[pos]
        return ' '.join(words)

    def _substitute_word(self, text: str) -> str:
        """Substitute word with similar length word"""
        words = text.split()
        if not words:
            return text
            
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]
        
        if len(word) in self.vocab and len(self.vocab[len(word)]) > 1:
            substitutes = [w for w in self.vocab[len(word)] if w != word]
            if substitutes:
                words[word_idx] = random.choice(substitutes)
                
        return ' '.join(words)

    def _delete_char(self, text: str) -> str:
        """Delete random character"""
        if not text:
            return text
            
        pos = random.randint(0, len(text) - 1)
        return text[:pos] + text[pos + 1:]

    def _insert_char(self, text: str) -> str:
        """Insert random character"""
        if not text:
            return text
            
        pos = random.randint(0, len(text))
        char = random.choice(string.ascii_lowercase)
        return text[:pos] + char + text[pos:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        rewards_array = np.array(self.rewards)
        solution_lengths = [len(sol) for sol in self.solutions]
        
        return {
            'num_samples': len(self),
            'avg_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'min_reward': np.min(rewards_array),
            'max_reward': np.max(rewards_array),
            'median_reward': np.median(rewards_array),
            'avg_solution_length': np.mean(solution_lengths),
            'max_solution_length': max(solution_lengths),
            'min_solution_length': min(solution_lengths),
            'vocab_size': sum(len(words) for words in self.vocab.values()),
            'cache_size': len(self._encoding_cache) if self.cache_encodings else 0
        }

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random batch of samples"""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        batch_inputs = []
        batch_rewards = []
        
        for idx in indices:
            inputs, reward = self[idx]
            batch_inputs.append(inputs)
            batch_rewards.append(reward)
            
        return torch.stack(batch_inputs), torch.stack(batch_rewards)

    def clear_cache(self):
        """Clear the encoding cache"""
        if self._encoding_cache is not None:
            self._encoding_cache.clear()

    def to_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
        """Convert dataset to DataLoader"""
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def save_to_disk(self, path: str):
        """Save dataset to disk"""
        data = {
            'solutions': self.solutions,
            'rewards': self.rewards,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'augment': self.augment,
            'augment_prob': self.augment_prob,
            'cache_encodings': self.cache_encodings
        }
        torch.save(data, path)

    @classmethod
    def load_from_disk(cls, path: str, tokenizer: PreTrainedTokenizer) -> 'PRMDataset':
        """Load dataset from disk"""
        data = torch.load(path)
        return cls(
            solutions=data['solutions'],
            rewards=data['rewards'],
            tokenizer=tokenizer,
            max_length=data['max_length'],
            min_length=data['min_length'],
            augment=data['augment'],
            augment_prob=data['augment_prob'],
            cache_encodings=data['cache_encodings']
        )

    def __str__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return (
            f"PRMDataset(samples={stats['num_samples']}, "
            f"avg_reward={stats['avg_reward']:.2f}, "
            f"vocab_size={stats['vocab_size']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"PRMDataset(solutions={len(self.solutions)}, "
            f"max_length={self.max_length}, "
            f"augment={self.augment}, "
            f"cache_encodings={self.cache_encodings})"
        )