import pytest
from omega_prm.config import OmegaPRMConfig

def test_config_default_values():
    """Test default configuration values"""
    config = OmegaPRMConfig()
    assert config.model_name == "gpt2"
    assert config.search_limit == 100
    assert config.alpha == 0.5
    assert config.beta == 0.9
    assert config.batch_size == 32

def test_config_validation():
    """Test configuration validation"""
    # Test valid configuration
    config = OmegaPRMConfig()
    config.validate()  # Should not raise any exceptions
    
    # Test invalid alpha
    with pytest.raises(AssertionError):
        invalid_config = OmegaPRMConfig(alpha=1.5)
        invalid_config.validate()
    
    # Test invalid beta
    with pytest.raises(AssertionError):
        invalid_config = OmegaPRMConfig(beta=1.5)
        invalid_config.validate()
    
    # Test invalid search limit
    with pytest.raises(AssertionError):
        invalid_config = OmegaPRMConfig(search_limit=-1)
        invalid_config.validate()

def test_config_custom_values():
    """Test custom configuration values"""
    custom_config = OmegaPRMConfig(
        model_name="gpt2-medium",
        batch_size=64,
        learning_rate=0.0001,
        use_wandb=True
    )
    assert custom_config.model_name == "gpt2-medium"
    assert custom_config.batch_size == 64
    assert custom_config.learning_rate == 0.0001
    assert custom_config.use_wandb is True