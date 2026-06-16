from pathlib import Path
from app.schemas import LyricLine


def transcribe_lyrics(
    audio_path: Path,
    language: str = "zh",
    device: str = "cuda",
) -> list[LyricLine]:
    """
    Transcribe vocals to lyrics with per-sentence timestamps using WhisperX.
    Returns empty list if no speech detected.
    """
    import whisperx
    import torch

    asr_options = {"word_timestamps": False}
    model = whisperx.load_model("large-v3", device, compute_type="float16",
                                 asr_options=asr_options, language=language)

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception:
        pass

    lyrics = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            lyrics.append(LyricLine(
                start=round(seg["start"], 2),
                end=round(seg["end"], 2),
                text=text,
            ))

    del model
    torch.cuda.empty_cache()

    return lyrics
