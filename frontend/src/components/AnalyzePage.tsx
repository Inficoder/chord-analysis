import { useAnalysis } from "../hooks/useAnalysis";
import { UploadZone } from "./UploadZone";
import { ProgressPanel } from "./ProgressPanel";
import { ErrorPanel } from "./ErrorPanel";
import { ResultToolbar } from "./ResultToolbar";
import { ChordChart } from "./ChordChart";
import { TimelineSidebar } from "./TimelineSidebar";
import { useState } from "react";

export function AnalyzePage() {
  const { status, result, error, uploading, upload, reset } = useAnalysis();
  const [currentTime, setCurrentTime] = useState(0);

  if (error) return <ErrorPanel message={error} onReset={reset} />;

  if (result) {
    return (
      <div className="space-y-6">
        <ResultToolbar result={result} />
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-6">
          <ChordChart result={result} />
          <TimelineSidebar
            chords={result.chords}
            beats={result.beats}
            lyrics={result.lyrics}
            duration={result.duration}
            currentTime={currentTime}
            onSeek={setCurrentTime}
          />
        </div>
      </div>
    );
  }

  if (status && status.status !== "done") {
    return <ProgressPanel status={status} />;
  }

  return <UploadZone onUpload={upload} uploading={uploading} />;
}
