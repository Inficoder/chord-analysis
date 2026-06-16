import torch
import torch.nn as nn

NUM_PITCH_CLASSES = 12
NUM_QUALITIES = 22  # includes N
NUM_CHORD_CLASSES = NUM_PITCH_CLASSES * NUM_QUALITIES  # 264


class ChordHead(nn.Module):
    """
    Beat-synchronous chord classifier with split heads.
    root: 13-class (12 pitch + no-root for N)
    quality: 22-class
    bass: 12-class (Phase 1.5)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.root_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_PITCH_CLASSES + 1),
        )
        self.quality_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_QUALITIES),
        )
        self.bass_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_PITCH_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, T_beat, input_dim) beat-synchronous pooled features
        Returns:
            dict with root_logits, quality_logits, bass_logits
        """
        return {
            "root_logits": self.root_classifier(features),
            "quality_logits": self.quality_classifier(features),
            "bass_logits": self.bass_classifier(features),
        }

    @staticmethod
    def root_quality_to_combined(root_logits: torch.Tensor, quality_logits: torch.Tensor) -> torch.Tensor:
        """Combine split root+quality logits into joint (264-class) logits via outer product."""
        root_probs = torch.softmax(root_logits, dim=-1)   # (B, T, 13)
        qual_probs = torch.softmax(quality_logits, dim=-1)  # (B, T, 22)
        combined = torch.einsum("bti,btj->btij", root_probs[:, :, :12], qual_probs)
        return (combined + 1e-10).log().reshape(*combined.shape[:2], -1)
