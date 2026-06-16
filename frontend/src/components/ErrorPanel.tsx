interface Props {
  message: string;
  onReset: () => void;
}

export function ErrorPanel({ message, onReset }: Props) {
  return (
    <div className="bg-surface border border-red/30 rounded-2xl p-8 space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-red text-2xl">!</span>
        <h2 className="text-lg font-semibold text-text">Analysis failed</h2>
      </div>
      <p className="text-muted text-sm">{message}</p>
      <button
        onClick={onReset}
        className="px-4 py-2 bg-surface border border-border rounded-lg text-text hover:border-muted transition-colors"
      >
        Try again
      </button>
    </div>
  );
}
