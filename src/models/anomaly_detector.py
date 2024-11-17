# src/models/anomaly_detector.py
import torch
import torch.nn as nn

class AnomalyDetector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_anomaly_score(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            decoded = self(x)
            reconstruction_error = torch.mean((x - decoded) ** 2)
            return reconstruction_error.item()