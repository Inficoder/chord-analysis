import type { AnalysisResult } from "../types";

interface Props {
  result: AnalysisResult;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, "0")}.${ms}`;
}

export default function ChordTimeline({ result }: Props) {
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "chord-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="text-lg font-semibold">Key: {result.key}</span>
          <span className="text-sm text-gray-400 ml-3">
            (confidence: {(result.key_confidence * 100).toFixed(0)}%)
          </span>
        </div>
        <button
          onClick={exportJson}
          className="px-3 py-1 text-sm bg-blue-600 hover:bg-blue-700 rounded"
        >
          Export JSON
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="text-sm text-gray-400 border-b border-gray-700">
              <th className="py-2 pr-4">Time</th>
              <th className="py-2 pr-4">Chord</th>
              <th className="py-2">Function</th>
            </tr>
          </thead>
          <tbody>
            {result.chords.map((ev, i) => (
              <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="py-2 pr-4 font-mono text-sm">
                  {formatTime(ev.start)} – {formatTime(ev.end)}
                </td>
                <td className="py-2 pr-4 font-semibold">{ev.chord}</td>
                <td className="py-2 text-blue-300">{ev.function}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
