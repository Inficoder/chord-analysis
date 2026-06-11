export interface ChordEvent {
  start: number;
  end: number;
  chord: string;
  function: string;
}

export interface AnalysisResult {
  key: string;
  key_confidence: number;
  chords: ChordEvent[];
}

export interface TaskState {
  task_id: string;
  file_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  stage: string;
  result: AnalysisResult | null;
  error: string | null;
}
