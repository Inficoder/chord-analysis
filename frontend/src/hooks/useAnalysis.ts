import { useState, useCallback, useRef } from "react";
import type { AnalysisResult, AnalysisStatus } from "../types";

const API = "/api";

export function useAnalysis() {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const upload = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/upload?language=zh`, { method: "POST", body: form });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Upload failed");
      }
      const { analysis_id } = await res.json();
      pollRef.current = setInterval(() => poll(analysis_id), 2000);
    } catch (e: any) {
      setError(e.message);
      setUploading(false);
    }
  }, []);

  const poll = useCallback(async (id: string) => {
    try {
      const res = await fetch(`${API}/status/${id}`);
      if (!res.ok) throw new Error("Status check failed");
      const s: AnalysisStatus = await res.json();
      setStatus(s);
      if (s.status === "done") {
        if (pollRef.current) clearInterval(pollRef.current);
        setUploading(false);
        const r = await fetch(`${API}/result/${id}`);
        if (r.ok) setResult(await r.json());
      } else if (s.status === "error") {
        if (pollRef.current) clearInterval(pollRef.current);
        setUploading(false);
        setError(s.error || "Analysis failed");
      }
    } catch (e: any) {
      if (pollRef.current) clearInterval(pollRef.current);
      setUploading(false);
      setError(e.message);
    }
  }, []);

  const reset = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    setStatus(null);
    setResult(null);
    setError(null);
    setUploading(false);
  }, []);

  return { status, result, error, uploading, upload, reset };
}
