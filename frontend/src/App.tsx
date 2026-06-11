import { useState, useCallback, useRef } from "react";
import UploadArea from "./components/UploadArea";
import ProgressIndicator from "./components/ProgressIndicator";
import ChordTimeline from "./components/ChordTimeline";
import { uploadFile, analyzeFile, subscribeTask } from "./services/api";
import type { TaskState } from "./types";

type AppState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "analyzing"; taskId: string; progress: number; stage: string }
  | { phase: "done"; taskState: TaskState }
  | { phase: "error"; message: string };

export default function App() {
  const [state, setState] = useState<AppState>({ phase: "idle" });
  const eventSourceRef = useRef<EventSource | null>(null);

  const handleFile = useCallback(async (file: File) => {
    try {
      setState({ phase: "uploading" });
      const fileId = await uploadFile(file);

      const taskId = await analyzeFile(fileId);
      setState({ phase: "analyzing", taskId, progress: 0, stage: "" });

      const es = subscribeTask(
        taskId,
        (update) => {
          if (update.status === "completed" || update.status === "failed") {
            setState(
              update.status === "completed"
                ? { phase: "done", taskState: update }
                : { phase: "error", message: update.error || "Analysis failed" }
            );
          } else {
            setState({
              phase: "analyzing",
              taskId,
              progress: update.progress,
              stage: update.stage,
            });
          }
        },
        () => setState({ phase: "error", message: "Connection lost" })
      );
      eventSourceRef.current = es;
    } catch (err) {
      setState({ phase: "error", message: (err as Error).message });
    }
  }, []);

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Chord Analysis</h1>

      <UploadArea
        onFileSelected={handleFile}
        disabled={state.phase === "uploading" || state.phase === "analyzing"}
      />

      {state.phase === "analyzing" && (
        <ProgressIndicator progress={state.progress} stage={state.stage} />
      )}

      {state.phase === "done" && state.taskState.result && (
        <ChordTimeline result={state.taskState.result} />
      )}

      {state.phase === "error" && (
        <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded text-red-200">
          {state.message}
          <button
            className="ml-3 underline"
            onClick={() => setState({ phase: "idle" })}
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
