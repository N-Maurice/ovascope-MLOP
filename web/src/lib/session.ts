// Classification history for this browser tab. It lives in memory only (never sent anywhere), so
// it survives navigation between pages but not a reload.
import { useSyncExternalStore } from "react";
import type { NotEggDetail, Prediction } from "../api/types";

export type HistoryEntry = {
  id: number;
  at: Date;
  fileName: string;
  thumbnail: string;   // object URL of the image that was sent
  latencyMs: number;
  servedBy: string | null;
  outcome: { kind: "prediction"; prediction: Prediction } | { kind: "rejected"; detail: NotEggDetail };
};

let entries: HistoryEntry[] = [];
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export const sessionHistory = {
  add(entry: Omit<HistoryEntry, "id">) {
    const full = { ...entry, id: Date.now() + Math.random() };
    entries = [full, ...entries].slice(0, 20);
    emit();
    return full;
  },
  clear() {
    entries.forEach((e) => URL.revokeObjectURL(e.thumbnail));
    entries = [];
    emit();
  },
};

export function useSessionHistory() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => entries,
  );
}
