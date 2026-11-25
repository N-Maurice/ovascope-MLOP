import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { RefreshCw } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useHealth, useRetrainStatus, useServiceMetrics } from "../api/hooks";
import { info } from "../lib/classes";
import { dateTime, duration, int, ms, num, pct } from "../lib/format";
import { Callout, ErrorState, Meter, PageHead, Skeleton, Stat } from "../components/ui";

function useNow(intervalMs = 1000) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs]);
  return now;
}

const STATUS_TONE: Record<string, string> = { completed: "ok", running: "teal", queued: "teal", failed: "bad" };

export default function Status() {
  const health = useHealth();
  const metrics = useServiceMetrics();
  const retrain = useRetrainStatus();
  const client = useQueryClient();
  const now = useNow();
  const m = metrics.data;
  const h = health.data;
  const updatedAgo = metrics.dataUpdatedAt ? Math.round((now - metrics.dataUpdatedAt) / 1000) : null;

  const latency = (m?.recent_requests ?? []).slice().reverse().map((r, i) => ({
    i: i + 1, latency: r.latency_ms, ok: r.status_code < 400, time: dateTime(r.timestamp),
  }));
  const successRate = m && m.prediction_requests_total ? m.prediction_requests_successful / m.prediction_requests_total : null;

  return (
    <div className="container">
      <PageHead
        eyebrow="Operations"
        title="Service status"
        actions={
          <div className="row">
            <span className="badge"><span className={`dot ${metrics.isError ? "bad" : "ok"} live`} />{metrics.isError ? "offline" : `live · updated ${updatedAgo ?? "–"}s ago`}</span>
            <button className="btn sm" onClick={() => client.invalidateQueries()}><RefreshCw size={14} /> Refresh</button>
          </div>
        }
      >
        Live health of the API, the loaded model and recent prediction traffic. Refreshes every 5 seconds.
      </PageHead>

      {health.isError && <ErrorState error={health.error} what="service health" />}
      {h && !h.model_loaded && <Callout tone="bad" title="No model loaded"><p>{h.model_error ?? "Train and activate a model."}</p></Callout>}

      {h && (
        <div className="grid g4 tight">
          <Stat label="API" value={<span className="row" style={{ gap: 10 }}><span className="dot ok live" />{h.status === "ok" ? "Healthy" : h.status}</span>} hint={`instance ${h.instance}`} />
          <Stat label="Active model" value={<span className="mono" style={{ fontSize: 18 }}>{h.model_version ?? "none"}</span>} hint={h.model_loaded ? "loaded in memory" : "not loaded"} />
          <Stat label="Uptime" value={duration(h.service_uptime_seconds + (health.dataUpdatedAt ? (now - health.dataUpdatedAt) / 1000 : 0))} hint="since this API process started" />
          <Stat label="Egg check" value={h.input_check.enabled ? "On" : "Off"}
            hint={h.input_check.enabled ? `threshold ${num(h.input_check.threshold)} · ${pct(h.input_check.test_non_egg_rejection_rate, 0)} non-eggs rejected` : "every image is classified"} />
        </div>
      )}

      <section className="section">
        <div className="section-head"><span className="eyebrow">Prediction traffic (all time, from the request log)</span></div>
        {metrics.isLoading && <Skeleton height={120} />}
        {m && (
          <div className="grid g4 tight">
            <Stat label="Requests" value={int(m.prediction_requests_total)} hint={`${int(m.prediction_requests_since_start)} since this process started`} />
            <Stat label="Successful" value={successRate === null ? "—" : (successRate * 100).toFixed(1)} unit={successRate === null ? undefined : "%"}
              hint={`${int(m.prediction_requests_failed)} failed or refused`} meter={successRate ?? undefined} />
            <Stat label="Latency p50" value={ms(m.recent_latency_ms.p50)} hint={`p95 ${ms(m.recent_latency_ms.p95)} · last ${m.recent_latency_ms.count}`} />
            <Stat label="Mean latency" value={ms(m.average_latency_ms)} hint={`max recent ${ms(m.recent_latency_ms.max)}`} />
          </div>
        )}
      </section>

      {m && (
        <div className="grid g2 section">
          <div className="card">
            <span className="eyebrow">Server latency · last {latency.length} requests</span>
            {latency.length ? (
              <div className="chart-box sm" style={{ marginTop: 12 }}>
                <ResponsiveContainer>
                  <LineChart data={latency} margin={{ left: -10, right: 8 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="i" tick={false} />
                    <YAxis unit=" ms" width={70} />
                    <Tooltip formatter={(v: number) => [ms(v), "latency"]} labelFormatter={(_, p) => p?.[0]?.payload.time ?? ""} />
                    <Line dataKey="latency" stroke="#0f766e" strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : <div className="empty">No predictions yet.</div>}
          </div>
          <div className="card">
            <span className="eyebrow">Recent requests</span>
            <div className="table-wrap" style={{ marginTop: 12, maxHeight: 250, overflowY: "auto" }}>
              <table className="data">
                <thead><tr><th>Time</th><th>Status</th><th>Prediction</th><th className="num">Score</th><th className="num">Latency</th></tr></thead>
                <tbody>
                  {m.recent_requests.slice(0, 12).map((r, i) => (
                    <tr key={i}>
                      <td className="small nowrap">{dateTime(r.timestamp)}</td>
                      <td><span className={`badge mono ${r.status_code < 400 ? "ok" : r.status_code === 422 ? "warn" : "bad"}`}>{r.status_code}</span></td>
                      <td className="small">{r.predicted_class_id !== null ? info(r.predicted_class_id).short : r.status_code === 422 ? "not an egg" : "—"}</td>
                      <td className="num">{r.confidence !== null ? pct(r.confidence, 0) : "—"}</td>
                      <td className="num">{ms(r.latency_ms)}</td>
                    </tr>
                  ))}
                  {!m.recent_requests.length && <tr><td colSpan={5} className="muted">No requests logged yet.</td></tr>}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {retrain.data && (
        <section className="section card">
          <div className="spread">
            <div>
              <span className="eyebrow">Retraining pipeline</span>
              <h3 style={{ marginTop: 6 }}>{retrain.data.job_running ? "A retraining job is running" : retrain.data.retraining_recommended ? "Retraining recommended" : "Collecting new images"}</h3>
            </div>
            <span className={`badge ${retrain.data.retraining_recommended ? "warn" : "teal"}`}>
              {int(retrain.data.pending_new_images)} / {retrain.data.trigger_threshold} new images
            </span>
          </div>
          <Meter value={retrain.data.pending_new_images / retrain.data.trigger_threshold} />
          {retrain.data.latest_job && (
            <p className="small muted" style={{ marginTop: 12 }}>
              Latest job <span className="mono">{retrain.data.latest_job.id}</span>:{" "}
              <span className={`badge ${STATUS_TONE[retrain.data.latest_job.status] ?? ""}`}>{retrain.data.latest_job.status}</span>{" "}
              {retrain.data.latest_job.promotion_decision ?? retrain.data.latest_job.error ?? retrain.data.latest_job.progress} · {dateTime(retrain.data.latest_job.created_at)}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
