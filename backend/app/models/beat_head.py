import torch
import torch.nn as nn


class BeatHead(nn.Module):
    """TCN beat/downbeat tracker from backbone features (high time resolution)."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.beat_classifier = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.downbeat_classifier = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, D) -> beat_logits (B,T,1), downbeat_logits (B,T,1)."""
        x = features.transpose(1, 2)
        x = self.tcn(x)
        beat = self.beat_classifier(x).transpose(1, 2)
        downbeat = self.downbeat_classifier(x).transpose(1, 2)
        return beat, downbeat
