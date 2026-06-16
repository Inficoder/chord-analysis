import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from app.models.beat_head import BeatHead
from app.models.chord_head import ChordHead
from app.models.key_head import KeyHead


class AnalysisBackbone:
    """Shared MERT/MusicFM backbone with three task heads."""

    def __init__(self, model_name: str = "m-a-p/MERT-v1-330M", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model: nn.Module | None = None
        self.feature_extractor: Wav2Vec2FeatureExtractor | None = None
        self.beat_head: BeatHead | None = None
        self.chord_head: ChordHead | None = None
        self.key_head: KeyHead | None = None
        self._loaded = False

    def load(self):
        """Load backbone and heads to device."""
        if self._loaded:
            return
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        hidden = self.model.config.hidden_size
        self.beat_head = BeatHead(input_dim=hidden).to(self.device).eval()
        self.chord_head = ChordHead(input_dim=hidden).to(self.device).eval()
        self.key_head = KeyHead(input_dim=hidden).to(self.device).eval()
        self._loaded = True

    def unload(self):
        """Move all to CPU and free GPU memory."""
        for attr in ["model", "beat_head", "chord_head", "key_head"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.to("cpu")
                setattr(self, attr, None)
        self.feature_extractor = None
        self._loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_features(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract frame-level features. Returns (1, T_frame, hidden_dim)."""
        inputs = self.feature_extractor(
            audio.cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state

    def _pool_by_beats(self, features: torch.Tensor, beat_times: list[float],
                       frame_rate: float) -> torch.Tensor:
        """Beat-synchronous pooling: aggregate frame features within each beat interval."""
        T = features.shape[1]
        N_beats = len(beat_times)
        pooled = torch.zeros(1, N_beats, features.shape[2], device=features.device)
        for i in range(N_beats):
            start_time = beat_times[i]
            end_time = beat_times[i + 1] if i + 1 < N_beats else (T / frame_rate)
            start_frame = max(0, min(int(start_time * frame_rate), T - 1))
            end_frame = max(start_frame + 1, min(int(end_time * frame_rate), T))
            pooled[0, i] = features[0, start_frame:end_frame].mean(dim=0)
        return pooled

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, sr: int,
                beat_times: list[float] | None = None) -> dict:
        """Run all three heads. Chord uses beat-sync pooling if beats provided."""
        features = self.extract_features(audio, sr)
        frame_rate = sr / 320  # approximate for MERT

        # Beat head on frame-level features
        beat_logits, downbeat_logits = self.beat_head(features)

        # Chord head: beat-synchronous or fallback to frame-level
        if beat_times and len(beat_times) > 1:
            chord_features = self._pool_by_beats(features, beat_times, frame_rate)
        else:
            chord_features = features
        chord_out = self.chord_head(chord_features)

        # Key head: attention pooling over all frames
        key_logits = self.key_head(features)

        return {
            "beat_logits": beat_logits.cpu(),
            "downbeat_logits": downbeat_logits.cpu(),
            "root_logits": chord_out["root_logits"].cpu(),
            "quality_logits": chord_out["quality_logits"].cpu(),
            "bass_logits": chord_out["bass_logits"].cpu(),
            "key_logits": key_logits.cpu(),
            "frame_rate": frame_rate,
        }
