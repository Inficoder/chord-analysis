import type { ChordSegment } from "../types";

interface Props {
  chord: ChordSegment;
  isActive: boolean;
  onClick: () => void;
}

const QUALITY_COLORS: Record<string, string> = {
  maj: "border-green/60 bg-green/5",
  min: "border-blue-400/60 bg-blue-400/5",
  "7": "border-yellow/60 bg-yellow/5",
  maj7: "border-green/40 bg-green/3",
  min7: "border-blue-400/40 bg-blue-400/3",
  dim: "border-red/60 bg-red/5",
  aug: "border-orange-400/60 bg-orange-400/5",
  N: "border-muted/30 bg-transparent",
};

export function ChordLine({ chord, isActive, onClick }: Props) {
  const colorClass = QUALITY_COLORS[chord.quality] || "border-border bg-transparent";

  return (
    <div
      onClick={onClick}
      className={`
        border rounded-lg p-3 cursor-pointer transition-all duration-150
        ${colorClass}
        ${isActive ? "ring-2 ring-accent scale-[1.02] z-10" : "hover:border-muted"}
      `}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-text font-bold text-lg">{chord.label}</span>
        {chord.roman && (
          <span className="text-accent text-sm font-medium">{chord.roman}</span>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs text-muted">
        {chord.function && <span>{chord.function}</span>}
        {chord.bass && chord.bass !== chord.root && (
          <span className="text-accent/70">/{chord.bass}</span>
        )}
        <span className="ml-auto">{Math.round(chord.confidence * 100)}%</span>
      </div>
      {chord.alternatives.length > 1 && (
        <div className="flex gap-1 mt-1.5 flex-wrap">
          {chord.alternatives.slice(1, 3).map((a, i) => (
            <span key={i} className="text-[10px] text-muted bg-surface px-1.5 py-0.5 rounded">
              {a.label} {Math.round(a.confidence * 100)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
