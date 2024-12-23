import torch
import torch.nn as nn
from typing import Optional

class ProcessRewardModel(nn.Module):
    """Enhanced Process Reward Model with dropout and layer normalization"""
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        num_layers: Optional[int] = None
    ):
        """
        Initialize Process Reward Model
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output
            dropout (float): Dropout rate
            num_layers (Optional[int]): Number of hidden layers
        """
        super(ProcessRewardModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout
        self.num_layers = num_layers or 2
        
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Hidden layers
        hidden_layers = []
        for i in range(self.num_layers - 1):
            in_features = hidden_size if i == 0 else hidden_size // (2 ** i)
            out_features = hidden_size // (2 ** (i + 1))
            hidden_layers.extend([
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Output layer
        last_hidden_size = hidden_size // (2 ** (self.num_layers - 1))
        self.fc_out = nn.Linear(last_hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Input layer
        x = self.dropout(torch.relu(self.ln1(self.fc1(x))))
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Output layer
        x = torch.sigmoid(self.fc_out(x))
        return x
    
    def get_complexity(self) -> int:
        """
        Calculate model complexity (total parameters)
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())