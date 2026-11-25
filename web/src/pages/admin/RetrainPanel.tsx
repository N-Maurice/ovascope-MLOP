import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { ApiError, post } from "../../api/client";
import { useJob, useJobs, useRetrainStatus } from "../../api/hooks";
import type { Job } from "../../api/types";
import { useAuth } from "../../auth";
import { dateTime, int, num } from "../../lib/format";
import { Callout, Meter, Skeleton } from "../../components/ui";
import { useToast } from "../../components/Toast";

const TONE: Record<string, string> = { completed: "ok", running: "teal", queued: "teal", failed: "bad" };
const STEPS = ["queued", "loading data", "training", "evaluating", "done"];

function stepOf(job: Job) {
  if (job.status === "completed") return 4;
  if (job.status === "queued") return 0;
  const p = job.progress ?? "";
  if (p.startsWith("evaluating")) return 3;
  if (p.startsWith("epoch")) return 2;
  return 1;
}

function JobCard({ job }: { job: Job }) {
  const step = stepOf(job);
  const active = job.status === "running" || job.status === "queued";
  return (
    <div className="card stack">
      <div className="spread">
        <div>
          <span className="eyebrow">Job <span className="mono">{job.id}</span></span>
          <div className="tiny muted" style={{ marginTop: 4 }}>{job.trigger}{job.requested_by ? ` by ${job.requested_by}` : ""} · created {dateTime(job.created_at)}</div>
        </div>
        <span className={`badge ${TONE[job.status]}`}>{active && <span className="dot ok live" />}{job.status}</span>
      </div>
      {job.status !== "failed" && (
        <div>
          <div className="row" style={{ gap: 4 }}>
            {STEPS.map((s, i) => (
              <div key={s} style={{ flex: 1 }}>
                <div style={{ height: 6, borderRadius: 9, background: i <= step ? "var(--teal)" : "var(--surface-2)", border: "1px solid var(--line)" }} />
                <div className="tiny muted" style={{ marginTop: 4, textAlign: "center" }}>{s}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      <p className="small mono" style={{ background: "var(--surface-2)", padding: "8px 10px", borderRadius: 8, overflowWrap: "anywhere" }}>
        {job.error ?? job.progress ?? "—"}
      </p>
      <dl className="kv">
        <dt>Base model</dt><dd className="mono">{job.base_version ?? "—"}</dd>
        <dt>New images</dt><dd>{int(job.n_new_images)}</dd>
        {job.candidate_version && <><dt>Candidate</dt><dd className="mono">{job.candidate_version}</dd></>}
        {job.finished_at && <><dt>Finished</dt><dd>{dateTime(job.finished_at)}</dd></>}
      </dl>
      {job.candidate_metrics && job.active_metrics && (
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>Validation metric</th><th className="num">Active model</th><th className="num">Candidate</th></tr></thead>
            <tbody>
              {["accuracy", "macro_precision", "macro_recall", "macro_f1"].map((k) => (
                <tr key={k}>
                  <td>{k.replace("macro_", "macro ")}</td>
                  <td className="num">{num(job.active_metrics![k])}</td>
                  <td className="num" style={{ color: job.candidate_metrics![k] >= job.active_metrics![k] ? "var(--green)" : "var(--amber)" }}>{num(job.candidate_metrics![k])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {job.promotion_decision && (
        <Callout tone={job.promoted ? "info" : "warn"} title={job.promoted ? "Candidate promoted" : "Candidate kept back"}>
          <p className="mono small">{job.promotion_decision}</p>
        </Callout>
      )}
    </div>
  );
}

export default function RetrainPanel() {
  const status = useRetrainStatus();
  const jobs = useJobs();
  const client = useQueryClient();
  const toast = useToast();
  const { signOut } = useAuth();
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const s = status.data;
  const watched = useJob(jobId ?? (s?.job_running ? s.latest_job?.id ?? null : null));

  // When a watched job finishes, refresh everything that depends on it.
  const finished = watched.data && ["completed", "failed"].includes(watched.data.status);
  useEffect(() => {
    if (finished) {
      ["retrain-status", "jobs", "versions", "health", "uploads"].forEach((k) => client.invalidateQueries({ queryKey: [k] }));
    }
  }, [finished, client]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await post<{ job_id: string }>("/retrain");
      setJobId(response.job_id);
      toast(`Retraining job ${response.job_id} started`);
      client.invalidateQueries({ queryKey: ["retrain-status"] });
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) return signOut();
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (!s) return <Skeleton height={300} />;
  const canStart = !s.job_running && s.pending_new_images >= s.min_new_images;
  return (
    <div className="classify-layout">
      <div className="stack">
        <div className="card stack">
          <span className="eyebrow">Start retraining</span>
          <div className="stat">
            <div className="value">{int(s.pending_new_images)}<small>new images pending</small></div>
          </div>
          <Meter value={s.pending_new_images / s.trigger_threshold} />
          <p className="small muted">
            Minimum {s.min_new_images} · recommended from {s.trigger_threshold}. The active model <span className="mono">{s.active_model}</span> is
            fine-tuned on the new images mixed with a replay sample of the original training data, then compared with the active model
            on the validation set. It replaces the active model only if its macro-F1 is not worse (within tolerance).
          </p>
          <button className="btn teal lg" disabled={!canStart || busy} onClick={start}>
            {busy ? <span className="spinner" /> : <Play size={18} />} Start retraining
          </button>
          {!canStart && !s.job_running && <p className="small" style={{ color: "var(--amber)" }}>Upload at least {s.min_new_images - s.pending_new_images} more labelled image(s) first.</p>}
          {s.job_running && <p className="small muted">A job is already running; only one can run at a time.</p>}
          {error && <Callout tone="bad"><p>{error}</p></Callout>}
        </div>
        {watched.data && <JobCard job={watched.data} />}
      </div>
      <div className="card">
        <span className="eyebrow">Job history</span>
        <div className="grid" style={{ gap: 8, marginTop: 12 }}>
          {(jobs.data ?? []).map((j) => (
            <button key={j.id} className="history-item" style={{ gridTemplateColumns: "minmax(0,1fr) auto", textAlign: "left", background: j.id === watched.data?.id ? "var(--teal-soft)" : undefined }}
              onClick={() => setJobId(j.id)}>
              <div style={{ minWidth: 0 }}>
                <div className="mono small">{j.id}</div>
                <div className="tiny muted ellipsis">{dateTime(j.created_at)} · {j.promotion_decision?.split(":")[0] ?? j.error ?? j.progress}</div>
              </div>
              <span className={`badge ${TONE[j.status]}`}>{j.status}</span>
            </button>
          ))}
          {jobs.data?.length === 0 && <p className="muted small">No retraining jobs yet.</p>}
        </div>
      </div>
    </div>
  );
}
