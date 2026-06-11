import { useCallback, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled: boolean;
}

export default function UploadArea({ onFileSelected, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
        dragOver ? "border-blue-400 bg-blue-400/10" : "border-gray-600 hover:border-gray-400"
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept=".wav,.mp3,.flac,.m4a,.aac"
        onChange={handleChange}
        className="hidden"
        id="file-input"
        disabled={disabled}
      />
      <label htmlFor="file-input" className="cursor-pointer">
        <p className="text-lg mb-2">
          {dragOver ? "Drop your file here" : "Drag & drop an audio file"}
        </p>
        <p className="text-sm text-gray-400">
          or click to browse — WAV, MP3, FLAC, M4A (max 50MB)
        </p>
      </label>
    </div>
  );
}
