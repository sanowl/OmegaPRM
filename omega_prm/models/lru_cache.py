from collections import deque
from typing import Optional, Dict, Any
import torch

class LRUCache:
    """Least Recently Used Cache implementation"""
    def __init__(self, capacity: int):
        """
        Initialize LRU Cache
        
        Args:
            capacity (int): Maximum number of items to store in cache
        """
        self.cache: Dict[str, Any] = {}
        self.capacity = capacity
        self.usage = deque(maxlen=capacity)
        self.hits = 0
        self.total_requests = 0
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve item from cache
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[torch.Tensor]: Cached value if exists, None otherwise
        """
        self.total_requests += 1
        if key in self.cache:
            self.hits += 1
            self.usage.remove(key)
            self.usage.append(key)
            return self.cache[key]
        return None
        
    def put(self, key: str, value: torch.Tensor):
        """
        Store item in cache
        
        Args:
            key (str): Cache key
            value (torch.Tensor): Value to cache
        """
        if key in self.cache:
            self.usage.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.usage.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.usage.append(key)

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate
        
        Returns:
            float: Cache hit rate between 0 and 1
        """
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.usage.clear()
        self.hits = 0
        self.total_requests = 0