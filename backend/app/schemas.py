from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class AnalysisStatusEnum(str, Enum):
    uploading = "uploading"
    queued = "queued"
    processing = "processing"
    done = "done"
    error = "error"


class Stages(BaseModel):
    vocal_sep: bool = False
    beat_tracking: bool = False
    chord_detection: bool = False
    key_detection: bool = False
    lyrics: bool = False
    harmony_analysis: bool = False


class AnalysisStatus(BaseModel):
    id: str
    filename: str
    status: AnalysisStatusEnum
    progress: int = Field(default=0, ge=0, le=100)
    stages: Stages = Field(default_factory=Stages)
    error: Optional[str] = None


class UploadResponse(BaseModel):
    analysis_id: str
    status_url: str


class BeatPoint(BaseModel):
    time: float
    beat_index: int
    bar_index: int
    beat_in_bar: int
    is_downbeat: bool
    confidence: float = Field(ge=0.0, le=1.0)


class KeyAlternative(BaseModel):
    key: str
    confidence: float


class KeyResult(BaseModel):
    key: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["fused", "ssl", "ks", "cadence"]
    alternatives: list[KeyAlternative] = Field(default_factory=list)


class KeySegment(BaseModel):
    start: float
    end: float
    key: str
    confidence: float = Field(ge=0.0, le=1.0)


class ChordAlternative(BaseModel):
    label: str
    confidence: float


class ChordSegment(BaseModel):
    index: int
    start: float
    end: float

    label: str
    root: str
    quality: str
    bass: Optional[str] = None

    beat_start: int
    beat_end: int
    bar: int

    roman: str = ""
    local_key: str = ""
    function: str = ""

    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[ChordAlternative] = Field(default_factory=list)


class LyricLine(BaseModel):
    start: float
    end: float
    text: str


class TimeSignature(BaseModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class Tempo(BaseModel):
    bpm: int
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    id: str
    global_key: KeyResult
    key_segments: list[KeySegment] = Field(default_factory=list)
    chords: list[ChordSegment] = Field(default_factory=list)
    lyrics: list[LyricLine] = Field(default_factory=list)
    beats: list[BeatPoint] = Field(default_factory=list)
    duration: float
    tempo: Tempo
    time_signature: TimeSignature


class ErrorResponse(BaseModel):
    error: str
    code: Literal["INVALID_FORMAT", "FILE_TOO_LARGE", "NOT_FOUND", "ANALYSIS_FAILED"]
