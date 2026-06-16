import type { AnalysisResult } from "../types";

interface Props {
  result: AnalysisResult;
}

export function ResultToolbar({ result }: Props) {
  const handleExport = async () => {
    try {
      const res = await fetch(`/api/download/${result.id}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `chord-analysis-${result.id}.json`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch {}
  };

  return (
    <div className="flex items-center gap-4 flex-wrap">
      <div className="flex items-center gap-3 text-sm">
        <KeyBadge label={result.global_key.key} conf={result.global_key.confidence} />
        <span className="text-muted">
          {result.tempo.bpm} BPM &middot; {result.time_signature.value}
        </span>
        <span className="text-muted">{result.duration}s</span>
      </div>
      <div className="ml-auto">
        <button
          onClick={handleExport}
          className="px-4 py-2 bg-accent text-white rounded-lg text-sm hover:bg-accent-dim transition-colors"
        >
          Export JSON
        </button>
      </div>
    </div>
  );
}

function KeyBadge({ label, conf }: { label: string; conf: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-accent/10 border border-accent/30 rounded-full text-accent text-xs font-medium">
      {label}
      <span className="text-accent/60">{Math.round(conf * 100)}%</span>
    </span>
  );
}
