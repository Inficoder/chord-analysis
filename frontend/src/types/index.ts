export interface BeatPoint {
  time: number;
  beat_index: number;
  bar_index: number;
  beat_in_bar: number;
  is_downbeat: boolean;
  confidence: number;
}

export interface KeyAlternative {
  key: string;
  confidence: number;
}

export interface KeyResult {
  key: string;
  confidence: number;
  method: "fused" | "ssl" | "ks" | "cadence";
  alternatives: KeyAlternative[];
}

export interface KeySegment {
  start: number;
  end: number;
  key: string;
  confidence: number;
}

export interface ChordAlternative {
  label: string;
  confidence: number;
}

export interface ChordSegment {
  index: number;
  start: number;
  end: number;
  label: string;
  root: string;
  quality: string;
  bass: string | null;
  beat_start: number;
  beat_end: number;
  bar: number;
  roman: string;
  local_key: string;
  function: string;
  confidence: number;
  alternatives: ChordAlternative[];
}

export interface LyricLine {
  start: number;
  end: number;
  text: string;
}

export interface TimeSignature {
  value: string;
  confidence: number;
}

export interface Tempo {
  bpm: number;
  confidence: number;
}

export interface AnalysisResult {
  id: string;
  global_key: KeyResult;
  key_segments: KeySegment[];
  chords: ChordSegment[];
  lyrics: LyricLine[];
  beats: BeatPoint[];
  duration: number;
  tempo: Tempo;
  time_signature: TimeSignature;
}

export interface AnalysisStatus {
  id: string;
  filename: string;
  status: "uploading" | "queued" | "processing" | "done" | "error";
  progress: number;
  stages: Stages;
  error: string | null;
}

export interface Stages {
  vocal_sep: boolean;
  beat_tracking: boolean;
  chord_detection: boolean;
  key_detection: boolean;
  lyrics: boolean;
  harmony_analysis: boolean;
}

export interface UploadResponse {
  analysis_id: string;
  status_url: string;
}
