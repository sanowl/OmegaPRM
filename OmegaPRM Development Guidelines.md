# OmegaPRM Development Guidelines

## Table of Contents
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Testing](#testing)
- [Development Workflow](#development-workflow)
- [Architecture Guidelines](#architecture-guidelines)
- [Performance Optimization](#performance-optimization)
- [Security Guidelines](#security-guidelines)

## Code Style

### Python Standards
- Follow PEP 8 style guide for Python code
- Maximum line length is 88 characters (Black default)
- Use spaces for indentation (4 spaces)
- Use snake_case for functions and variables
- Use PascalCase for class names

### Type Hints
```python
# Good
def process_data(input_text: str, max_length: int = 100) -> List[str]:
    pass

# Bad
def process_data(input_text, max_length=100):
    pass
```

### Imports Organization
```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import numpy as np
import torch
from transformers import AutoTokenizer

# Local imports
from omega_prm.utils import setup_logging
from omega_prm.data import PRMDataset
```

### Variable Naming
```python
# Good
max_sequence_length = 512
num_epochs = 10
model_config = {"hidden_size": 768}

# Bad
max_len = 512
n = 10
config = {"h": 768}
```

## Documentation

### Docstring Format (Google Style)
```python
def train_model(
    input_data: torch.Tensor,
    num_epochs: int = 10,
    learning_rate: float = 0.001
) -> Dict[str, float]:
    """Train the model with given parameters.

    Args:
        input_data: Input tensor for training.
        num_epochs: Number of training epochs. Defaults to 10.
        learning_rate: Learning rate for optimization. Defaults to 0.001.

    Returns:
        Dictionary containing training metrics (loss, accuracy).

    Raises:
        ValueError: If input_data is empty or has wrong shape.
    """
    pass
```

### Module Documentation
Each module should have a top-level docstring explaining:
- Module purpose
- Main classes/functions
- Usage examples
- Any important notes

Example:
```python
"""
Process Reward Model Dataset Module.

This module provides the PRMDataset class for handling training data
in the Process Reward Model. It supports data augmentation, caching,
and efficient batch processing.

Example:
    >>> dataset = PRMDataset(solutions, rewards, tokenizer)
    >>> dataloader = dataset.to_dataloader(batch_size=32)
"""
```

### README Structure
- Project description
- Installation instructions
- Quick start guide
- Configuration options
- Advanced usage examples
- Contributing guidelines
- License information

## Testing

### Test Organization
```python
# test_dataset.py
import pytest
from omega_prm.data import PRMDataset

class TestPRMDataset:
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        return PRMDataset(...)
    
    def test_dataset_initialization(self, sample_dataset):
        """Test dataset initialization."""
        assert len(sample_dataset) > 0
        
    def test_data_augmentation(self, sample_dataset):
        """Test data augmentation functionality."""
        pass
```

### Test Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for critical paths
- Include both positive and negative test cases
- Test edge cases and error conditions

### Testing Best Practices
1. Use meaningful test names that describe the scenario
2. One assertion per test when possible
3. Use fixtures for setup and teardown
4. Mock external dependencies
5. Include performance tests for critical paths

## Development Workflow

### Branch Naming Convention
```
feature/add-data-augmentation
bugfix/fix-memory-leak
hotfix/security-patch-123
refactor/optimize-tokenization
docs/update-readme
```

### Commit Message Format
```
type(scope): Short description

Longer description if needed

Breaking changes (if any)

Closes #123
```

Types: feat, fix, docs, style, refactor, test, chore

### Pull Request Process
1. Create branch from main
2. Implement changes
3. Run quality checks:
   ```bash
   make format
   make lint
   make test
   ```
4. Update documentation
5. Create PR with description
6. Address review comments
7. Squash and merge

## Architecture Guidelines

### Component Organization
```
omega_prm/
├── core/         # Core business logic
├── models/       # ML models and components
├── data/         # Data handling and processing
├── utils/        # Utility functions
└── configs/      # Configuration handling
```

### Dependency Injection
```python
# Good
class ModelTrainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

# Bad
class ModelTrainer:
    def __init__(self):
        self.model = create_default_model()
        self.optimizer = torch.optim.Adam(self.model.parameters())
```

### Error Handling
```python
class PRMError(Exception):
    """Base exception for OmegaPRM."""
    pass

class DatasetError(PRMError):
    """Raised when dataset operations fail."""
    pass

def process_data(data: List[str]) -> List[str]:
    if not data:
        raise DatasetError("Empty data provided")
    try:
        return [preprocess(item) for item in data]
    except Exception as e:
        raise DatasetError(f"Processing failed: {str(e)}")
```

## Performance Optimization

### Memory Management
```python
# Good - Generator for large datasets
def process_large_dataset(data_path: str):
    with open(data_path) as f:
        for line in f:
            yield process_line(line)

# Bad - Loading entire dataset
def process_large_dataset(data_path: str):
    with open(data_path) as f:
        data = f.readlines()
    return [process_line(line) for line in data]
```

### Caching Strategy
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input_data: str) -> str:
    # Complex processing
    pass
```

### Batch Processing
```python
def process_batch(items: List[str], batch_size: int = 32):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield process_items(batch)
```

## Security Guidelines

### Data Handling
- Never store sensitive data in code
- Use environment variables for credentials
- Sanitize user input
- Implement proper access controls

### Configuration Security
```python
# Good - Load from environment
api_key = os.environ.get('API_KEY')
if not api_key:
    raise ValueError("API_KEY not set")

# Bad - Hardcoded credentials
api_key = "1234567890abcdef"
```

### Dependency Management
- Regular security updates
- Lock dependency versions
- Use verified packages
- Scan for vulnerabilities

### Code Review Security Checklist
- [ ] No sensitive data in code
- [ ] Input validation implemented
- [ ] Error handling doesn't expose internals
- [ ] Dependencies are up to date
- [ ] Access controls implemented