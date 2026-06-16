import numpy as np
from pathlib import Path
import torch


def separate_vocals(
    audio_path: Path,
    output_dir: Path,
    device: str = "cuda",
) -> tuple[Path, Path]:
    """
    Run Demucs htdemucs to separate vocals and accompaniment.

    Returns:
        (vocals_path, accompaniment_path) - paths to separated WAV files
    """
    from demucs import separate

    results = separate.main(["--two-stems", "vocals", "-o", str(output_dir), str(audio_path)])

    stem_dir = output_dir / "htdemucs" / audio_path.stem
    vocals_path = stem_dir / "vocals.wav"
    accomp_path = stem_dir / "no_vocals.wav"

    return vocals_path, accomp_path
