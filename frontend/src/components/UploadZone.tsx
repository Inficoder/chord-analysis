import { useRef, useState, type DragEvent, type ChangeEvent } from "react";

interface Props {
  onUpload: (file: File) => void;
  uploading: boolean;
}

export function UploadZone({ onUpload, uploading }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer
        transition-colors duration-200
        ${dragging
          ? "border-accent bg-accent/5"
          : "border-border hover:border-muted"
        }
        ${uploading ? "pointer-events-none opacity-50" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.m4a,.flac,.ogg"
        onChange={handleChange}
        className="hidden"
      />
      <div className="text-4xl mb-4">&#9835;</div>
      <p className="text-text font-medium text-lg mb-2">
        {uploading ? "Analyzing..." : "Drop audio file here or click to browse"}
      </p>
      <p className="text-muted text-sm">
        MP3, WAV, M4A, FLAC, OGG &middot; Up to 50 MB &middot; Max 10 minutes
      </p>
    </div>
  );
}
