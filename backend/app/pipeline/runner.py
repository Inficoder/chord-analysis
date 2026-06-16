"""
Analysis pipeline orchestrator: runs each stage sequentially and returns
a structured AnalysisResult.
"""
import uuid
import time
from pathlib import Path
import torch
import numpy as np

from app.config import TARGET_SR, UPLOAD_DIR, RESULT_DIR
from app.schemas import (
    AnalysisResult, KeyResult, KeySegment, ChordSegment, LyricLine,
    BeatPoint, Tempo, TimeSignature,
)
from app.utils.audio import load_audio
from app.pipeline.vocal_sep import separate_vocals
from app.pipeline.beat_track import detect_beats
from app.pipeline.chord_detect import detect_chords
from app.pipeline.key_detect import ssl_detect_key, ks_detect_key, fuse_key_results, detect_key_segments
from app.pipeline.harmony_analysis import run_harmony_analysis
from app.pipeline.lyrics_transcribe import transcribe_lyrics
from app.models.backbone import AnalysisBackbone


class PipelineRunner:
    """Orchestrates the full chord analysis pipeline with GPU memory management."""

    def __init__(self, device: str = "cuda", lyrics_language: str = "zh"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lyrics_language = lyrics_language
        self.backbone = AnalysisBackbone(device=self.device)

    def run(self, audio_path: Path, analysis_id: str | None = None) -> AnalysisResult:
        """Run full analysis pipeline on an audio file."""
        aid = analysis_id or uuid.uuid4().hex[:12]
        t_start = time.perf_counter()

        # 1. Load audio
        audio, sr = load_audio(audio_path, target_sr=TARGET_SR)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float().to(self.device)

        # 2. Load backbone
        self.backbone.load()

        # 3. Run beat tracking
        backbone_out = self.backbone.forward(audio_tensor, TARGET_SR)
        frame_rate = backbone_out["frame_rate"]
        beats, tempo, time_sig = detect_beats(
            backbone_out["beat_logits"], backbone_out["downbeat_logits"], frame_rate,
        )

        # 4. Run chord detection
        chords = detect_chords(
            self.backbone, audio_tensor, TARGET_SR, beats, frame_rate,
        )

        # 5. Key detection (SSL + K-S fusion)
        ssl_key = ssl_detect_key(backbone_out["key_logits"])
        ks_key = ks_detect_key(chords)
        global_key = fuse_key_results(ssl_key, ks_key)

        # 6. Local key segments
        key_segments = detect_key_segments(chords)

        # 7. Harmony analysis
        chords = run_harmony_analysis(chords, key_segments, global_key.key)

        # 8. Unload backbone before WhisperX (GPU memory)
        self.backbone.unload()

        # 9. Vocal separation + lyrics (WhisperX)
        vocals_path = _run_vocal_sep(audio_path)
        lyrics = _run_lyrics(vocals_path, self.lyrics_language, self.device)

        # Duration
        duration = round(audio.shape[0] / TARGET_SR, 2)
        elapsed = round(time.perf_counter() - t_start, 2)
        print(f"[pipeline] Analysis {aid} complete in {elapsed}s")

        return AnalysisResult(
            id=aid,
            global_key=global_key,
            key_segments=key_segments,
            chords=chords,
            lyrics=lyrics,
            beats=beats,
            duration=duration,
            tempo=tempo,
            time_signature=time_sig,
        )


def _run_vocal_sep(audio_path: Path) -> Path | None:
    """Run Demucs vocal separation. Returns vocals path or None."""
    try:
        out_dir = RESULT_DIR / "stems"
        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_path, _ = separate_vocals(audio_path, out_dir)
        if vocals_path.exists():
            return vocals_path
    except Exception as e:
        print(f"[pipeline] Vocal separation failed: {e}")
    return None


def _run_lyrics(vocals_path: Path | None, language: str, device: str) -> list[LyricLine]:
    """Run WhisperX lyrics transcription. Returns empty list on failure."""
    if vocals_path is None or not vocals_path.exists():
        return []
    try:
        return transcribe_lyrics(vocals_path, language=language, device=device)
    except Exception as e:
        print(f"[pipeline] Lyrics transcription failed: {e}")
        return []
