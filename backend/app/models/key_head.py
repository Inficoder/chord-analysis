import torch
import torch.nn as nn


class KeyHead(nn.Module):
    """24-class key classifier with attention pooling."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_keys: int = 24):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_keys),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) frame or beat-level features
        Returns:
            key_logits: (B, num_keys) after attention-weighted pooling
        """
        attn_weights = torch.softmax(self.attention(features), dim=1)  # (B, T, 1)
        pooled = (features * attn_weights).sum(dim=1)  # (B, D)
        return self.classifier(pooled)
