import { useState, useRef } from "react";
import type { AnalysisResult } from "../types";
import { ChordLine } from "./ChordLine";
import { TimelineSidebar } from "./TimelineSidebar";

interface Props {
  result: AnalysisResult;
}

export function ChordChart({ result }: Props) {
  const [activeChord, setActiveChord] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const bars = new Map<number, typeof result.chords>();
  result.chords.forEach((ch) => {
    const b = bars.get(ch.bar) || [];
    b.push(ch);
    bars.set(ch.bar, b);
  });
  const sortedBars = [...bars.entries()].sort((a, b) => a[0] - b[0]);

  const handleSeek = (t: number) => {
    setCurrentTime(t);
    if (audioRef.current) {
      audioRef.current.currentTime = t;
    }
  };

  return (
    <div className="space-y-6">
      {/* Bar grid */}
      <div className="grid grid-cols-[auto_1fr_1fr_1fr_1fr] gap-2 items-start">
        <div className="text-xs text-muted font-medium p-2">Bar</div>
        {[1, 2, 3, 4].map((beat) => (
          <div key={beat} className="text-xs text-muted font-medium p-2 text-center">
            Beat {beat}
          </div>
        ))}

        {sortedBars.map(([barNum, barChords]) => {
          const sorted = [...barChords].sort((a, b) => a.beat_in_bar - b.beat_in_bar || a.start - b.start);
          return (
            <div key={barNum} className="contents">
              <div className="text-xs text-muted p-2 font-mono">{barNum + 1}</div>
              {sorted.map((ch) => (
                <div
                  key={ch.index}
                  className="col-span-1"
                  style={{}}
                >
                  <ChordLine
                    chord={ch}
                    isActive={activeChord === ch.index}
                    onClick={() => setActiveChord(activeChord === ch.index ? null : ch.index)}
                  />
                </div>
              ))}
            </div>
          );
        })}
      </div>

      {/* Lyrics over chords */}
      {result.lyrics.length > 0 && (
        <div className="border-t border-border pt-4">
          <h3 className="text-sm font-semibold text-muted uppercase tracking-wider mb-3">Lyrics</h3>
          <div className="flex flex-wrap gap-1">
            {result.lyrics.map((line, i) => (
              <span
                key={i}
                className="text-sm text-text hover:text-accent cursor-pointer transition-colors"
                onClick={() => handleSeek(line.start)}
                title={`${line.start.toFixed(1)}s - ${line.end.toFixed(1)}s`}
              >
                {line.text}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Active chord detail */}
      {activeChord !== null && (() => {
        const ch = result.chords.find(c => c.index === activeChord);
        if (!ch) return null;
        return (
          <div className="bg-surface border border-border rounded-xl p-4 space-y-2">
            <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Chord Detail</h3>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div><span className="text-muted">Label:</span> <span className="text-text font-medium">{ch.label}</span></div>
              <div><span className="text-muted">Roman:</span> <span className="text-accent">{ch.roman || "-"}</span></div>
              <div><span className="text-muted">Function:</span> <span className="text-text">{ch.function || "-"}</span></div>
              <div><span className="text-muted">Root:</span> <span className="text-text">{ch.root || "-"}</span></div>
              <div><span className="text-muted">Quality:</span> <span className="text-text">{ch.quality}</span></div>
              <div><span className="text-muted">Bass:</span> <span className="text-text">{ch.bass || ch.root || "-"}</span></div>
              <div><span className="text-muted">Local key:</span> <span className="text-text">{ch.local_key || "-"}</span></div>
              <div><span className="text-muted">Confidence:</span> <span className="text-text">{Math.round(ch.confidence * 100)}%</span></div>
              <div><span className="text-muted">Time:</span> <span className="text-text">{ch.start.toFixed(1)}s - {ch.end.toFixed(1)}s</span></div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
