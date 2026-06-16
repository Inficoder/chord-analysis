import type { AnalysisStatus } from "../types";

const LABELS: Record<string, string> = {
  vocal_sep: "Vocal separation",
  beat_tracking: "Beat tracking",
  chord_detection: "Chord detection",
  key_detection: "Key detection",
  lyrics: "Lyrics transcription",
  harmony_analysis: "Harmony analysis",
};

interface Props {
  status: AnalysisStatus;
}

export function ProgressPanel({ status }: Props) {
  const pct = status.progress || 0;

  return (
    <div className="bg-surface border border-border rounded-2xl p-8 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text">Analysis in progress</h2>
        <span className="text-accent font-mono text-sm">{pct}%</span>
      </div>

      <div className="w-full bg-border rounded-full h-2 overflow-hidden">
        <div
          className="h-full bg-accent rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        {Object.entries(status.stages).map(([key, done]) => (
          <div key={key} className="flex items-center gap-2">
            <span className={done ? "text-green" : "text-muted"}>
              {done ? "✓" : "○"}
            </span>
            <span className={done ? "text-text" : "text-muted"}>{LABELS[key] || key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
