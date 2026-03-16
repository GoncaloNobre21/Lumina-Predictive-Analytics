import torch
import torch.nn as nn
from typing import Dict, Any

class TimeSeriesTransformer(nn.Module):
    """
    Advanced Transformer model for Time-Series Forecasting.
    Designed for complex patterns and long-range dependencies.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, 1)  # Single step prediction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Predictive output of shape (batch_size, 1)
        """
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        # Global average pooling over the sequence dimension
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x

class PredictiveEngine:
    """
    Orchestration layer for training and inference of forecasting models.
    """
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TimeSeriesTransformer(
            input_dim=config.get("input_dim", 10),
            d_model=config.get("d_model", 64)
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get("lr", 0.001))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))
