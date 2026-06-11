from enum import Enum
from uuid import UUID
from pydantic import BaseModel


class TaskStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ChordEvent(BaseModel):
    start: float
    end: float
    chord: str
    function: str


class AnalysisResult(BaseModel):
    key: str
    key_confidence: float
    chords: list[ChordEvent]


class TaskState(BaseModel):
    task_id: UUID
    file_id: UUID
    status: TaskStatus = TaskStatus.pending
    progress: int = 0
    stage: str = ""
    result: AnalysisResult | None = None
    error: str | None = None


class AnalyzeRequest(BaseModel):
    file_id: UUID
