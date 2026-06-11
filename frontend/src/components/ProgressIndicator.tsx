interface Props {
  progress: number;
  stage: string;
}

const STAGE_LABELS: Record<string, string> = {
  loading: "Loading audio...",
  key_detection: "Detecting musical key...",
  chord_recognition: "Recognizing chords...",
  function_analysis: "Analyzing harmonic function...",
  completed: "Analysis complete",
  failed: "Analysis failed",
};

export default function ProgressIndicator({ progress, stage }: Props) {
  const label = STAGE_LABELS[stage] || stage || "Waiting...";

  return (
    <div className="my-6">
      <div className="flex justify-between mb-1">
        <span className="text-sm">{label}</span>
        <span className="text-sm text-gray-400">{progress}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
