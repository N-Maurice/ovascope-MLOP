import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Database, RefreshCw } from "lucide-react";
import { send } from "../../api/client";
import { useMongoStatus, type MongoStatus } from "../../api/hooks";
import { useToast } from "../../components/Toast";
import { dateTime, int } from "../../lib/format";

const COLLECTIONS = ["uploads", "retrain_jobs", "request_log", "models", "users"];

/** Status of the optional MongoDB Atlas copy of the records, with a "Sync now" button. */
export default function MongoCard() {
  const status = useMongoStatus(true);
  const queryClient = useQueryClient();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const s = status.data;
  if (!s) return null;

  async function syncNow() {
    setBusy(true);
    try {
      const result = await send<MongoStatus>("POST", "/mongo/sync");
      queryClient.setQueryData(["mongo"], result);
      toast("Records copied to MongoDB Atlas");
    } catch (error) {
      toast(error instanceof Error ? error.message : String(error), "bad");
      queryClient.invalidateQueries({ queryKey: ["mongo"] });
    } finally {
      setBusy(false);
    }
  }

  const counts = s.documents ?? s.last_result;
  return (
    <div className="card">
      <div className="spread">
        <span className="eyebrow"><Database size={14} style={{ verticalAlign: -2 }} /> MongoDB Atlas copy</span>
        {s.enabled && (
          <button className="btn sm" onClick={syncNow} disabled={busy}>
            {busy ? <span className="spinner" /> : <RefreshCw size={14} />} Sync now
          </button>
        )}
      </div>
      {!s.enabled ? (
        <p className="small muted" style={{ marginTop: 8 }}>
          Off. Add <span className="mono">MONGODB_URI</span> to <span className="mono">.env</span> and restart the API to keep a
          copy of these records in MongoDB Atlas. The local database stays the main store.
        </p>
      ) : (
        <>
          <p className="small" style={{ marginTop: 8 }}>
            <span className={`badge ${s.last_error ? "bad" : s.last_sync ? "ok" : "warn"}`}>
              {s.last_error ? "error" : s.last_sync ? "in sync" : "waiting"}
            </span>{" "}
            <span className="mono">{s.cluster}</span> · database <span className="mono">{s.database}</span>
            {" · "}last sync {s.last_sync ? dateTime(s.last_sync) : "not yet"} · every {s.sync_interval_seconds} s
            {s.store_images ? " · images included" : ""}
          </p>
          {s.last_error && <p className="small" style={{ color: "var(--red)", marginTop: 6 }}>{s.last_error}</p>}
          {counts && Object.keys(counts).length > 0 && (
            <div className="tiny muted" style={{ marginTop: 6 }}>
              {COLLECTIONS.filter((c) => c in counts).map((c) => `${c}: ${int(counts[c])}`).join(" · ")}
            </div>
          )}
        </>
      )}
    </div>
  );
}
