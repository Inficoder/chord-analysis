import type { TaskState } from "../types";

const BASE = "/api";

export async function uploadFile(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  const data = await res.json();
  return data.file_id;
}

export async function analyzeFile(fileId: string): Promise<string> {
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId }),
  });
  if (!res.ok) throw new Error(`Analyze failed: ${res.statusText}`);
  const data = await res.json();
  return data.task_id;
}

export async function getTask(taskId: string): Promise<TaskState> {
  const res = await fetch(`${BASE}/task/${taskId}`);
  if (!res.ok) throw new Error(`Task fetch failed: ${res.statusText}`);
  return res.json();
}

export function subscribeTask(
  taskId: string,
  onUpdate: (state: TaskState) => void,
  onError: (err: Event) => void
): EventSource {
  const es = new EventSource(`${BASE}/task/${taskId}/stream`);
  es.onmessage = (event) => {
    const data = JSON.parse(event.data) as TaskState;
    onUpdate(data);
    if (data.status === "completed" || data.status === "failed") {
      es.close();
    }
  };
  es.onerror = onError;
  return es;
}
