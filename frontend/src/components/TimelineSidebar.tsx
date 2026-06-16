import { useRef, useEffect, useState } from "react";
import type { ChordSegment, BeatPoint, LyricLine } from "../types";

interface Props {
  chords: ChordSegment[];
  beats: BeatPoint[];
  lyrics: LyricLine[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}

export function TimelineSidebar({ chords, beats, lyrics, duration, currentTime, onSeek }: Props) {
  const barRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    if (!barRef.current) return;
    const rect = barRef.current.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(duration, ratio * duration)));
  };

  return (
    <div className="bg-surface border border-border rounded-xl p-4 space-y-3">
      <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Timeline</h3>

      {/* Time bar */}
      <div
        ref={barRef}
        onClick={handleClick}
        className="relative w-full h-6 bg-border rounded cursor-pointer overflow-hidden"
      >
        <div
          className="absolute h-full bg-accent/30 rounded transition-all"
          style={{ width: `${(currentTime / duration) * 100}%` }}
        />
        <div
          className="absolute h-full w-0.5 bg-white"
          style={{ left: `${(currentTime / duration) * 100}%` }}
        />
      </div>

      <div className="text-xs text-muted flex justify-between">
        <span>{fmt(currentTime)}</span>
        <span>{fmt(duration)}</span>
      </div>

      {/* Bar markers */}
      <div className="relative h-4 w-full">
        {beats.filter(b => b.is_downbeat).map((b, i) => (
          <div
            key={i}
            className="absolute top-0 w-px h-full bg-border"
            style={{ left: `${(b.time / duration) * 100}%` }}
            title={`Bar ${b.bar_index}`}
          />
        ))}
      </div>

      {/* Chords condensed view */}
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {chords.map((ch) => (
          <div
            key={ch.index}
            className="flex items-center gap-2 text-xs cursor-pointer hover:bg-surface/50 px-1 py-0.5 rounded"
            onClick={() => onSeek(ch.start)}
          >
            <span className="text-muted w-14 shrink-0">{fmt(ch.start)}</span>
            <span className="text-accent font-medium">{ch.label}</span>
            {ch.roman && <span className="text-text">{ch.roman}</span>}
            {ch.function && (
              <span className="text-muted ml-auto">{ch.function}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
