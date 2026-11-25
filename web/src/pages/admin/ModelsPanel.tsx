import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Undo2 } from "lucide-react";
import { ApiError, post } from "../../api/client";
import { useVersions } from "../../api/hooks";
import { useAuth } from "../../auth";
import { dateTime, num } from "../../lib/format";
import { Callout, ErrorState, Modal, Skeleton } from "../../components/ui";
import { useToast } from "../../components/Toast";

export default function ModelsPanel() {
  const versions = useVersions();
  const client = useQueryClient();
  const toast = useToast();
  const { signOut } = useAuth();
  const [confirm, setConfirm] = useState<{ action: "activate" | "rollback"; version?: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!confirm) return;
    setBusy(true);
    setError(null);
    try {
      const pointer = await post<{ version: string }>(confirm.action === "rollback" ? "/model/rollback" : `/model/activate/${confirm.version}`);
      toast(`Now serving ${pointer.version}`);
      ["versions", "health", "metrics", "retrain-status"].forEach((k) => client.invalidateQueries({ queryKey: [k] }));
      setConfirm(null);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) return signOut();
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (versions.isLoading) return <Skeleton height={300} />;
  if (versions.isError) return <ErrorState error={versions.error} what="model versions" />;
  const list = [...(versions.data ?? [])].reverse();
  return (
    <div className="stack">
      <div className="spread">
        <p className="small muted" style={{ maxWidth: 640 }}>
          Every trained or retrained model is kept as a version. Activating one makes the API serve it immediately (all API
          instances pick up the change). Rollback returns to the previously active version.
        </p>
        <button className="btn" onClick={() => setConfirm({ action: "rollback" })}><Undo2 size={16} /> Roll back</button>
      </div>
      {error && <Callout tone="bad"><p>{error}</p></Callout>}
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr><th>Version</th><th>Created</th><th>From</th><th className="num">Val F1</th><th className="num">Val accuracy</th><th className="num">Test F1</th><th className="num">Size</th><th /></tr>
          </thead>
          <tbody>
            {list.map((v) => (
              <tr key={v.version} className={v.active ? "hl" : ""}>
                <td>
                  <div className="mono small" style={{ fontWeight: 600 }}>{v.version}</div>
                  <div className="tiny muted">{v.architecture}</div>
                </td>
                <td className="small nowrap">{dateTime(v.created_at)}</td>
                <td className="mono tiny">{v.parent_version ?? "—"}</td>
                <td className="num">{num(v.validation_metrics?.macro_f1)}</td>
                <td className="num">{num(v.validation_metrics?.accuracy)}</td>
                <td className="num">{num(v.test_metrics?.macro_f1)}</td>
                <td className="num">{v.model_size_mb} MB</td>
                <td className="right">
                  {v.active
                    ? <span className="badge ok"><CheckCircle2 size={13} /> serving</span>
                    : <button className="btn sm" onClick={() => setConfirm({ action: "activate", version: v.version })}>Activate</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="tiny muted">Validation F1 of retrained versions is measured on the same validation set; test F1 exists only for initially trained models.</p>

      {confirm && (
        <Modal onClose={() => setConfirm(null)} label="Confirm model change">
          <div className="stack" style={{ maxWidth: 460 }}>
            <h3>{confirm.action === "rollback" ? "Roll back to the previous model?" : <>Activate <span className="mono">{confirm.version}</span>?</>}</h3>
            <p className="small muted">All new predictions will use this model straight away. You can switch back at any time.</p>
            <div className="row">
              <button className="btn primary" onClick={run} disabled={busy}>{busy && <span className="spinner" />} Confirm</button>
              <button className="btn" onClick={() => setConfirm(null)}>Cancel</button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
