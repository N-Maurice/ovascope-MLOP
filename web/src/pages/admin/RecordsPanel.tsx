import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { API_BASE, authHeaders } from "../../api/client";
import { useUploads } from "../../api/hooks";
import { bytes, dateTime, int } from "../../lib/format";
import { ErrorState, Segmented, Skeleton, Species, Stat } from "../../components/ui";
import MongoCard from "./MongoCard";

/** Upload thumbnails need the admin key header, so they are fetched as blobs. */
function AuthImage({ id }: { id: number }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let revoked: string | null = null;
    const controller = new AbortController();
    fetch(`${API_BASE}/uploads/${id}/image`, { headers: authHeaders(), signal: controller.signal })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => { if (b) { revoked = URL.createObjectURL(b); setUrl(revoked); } })
      .catch(() => {});
    return () => { controller.abort(); if (revoked) URL.revokeObjectURL(revoked); };
  }, [id]);
  return url ? <img className="thumb-sm" src={url} alt="" loading="lazy" /> : <div className="thumb-sm" />;
}

export default function RecordsPanel() {
  const uploads = useUploads(true);
  const [filter, setFilter] = useState<"all" | "pending" | "used">("all");

  const rows = useMemo(
    () => (uploads.data?.records ?? []).filter((r) => filter === "all" || r.training_status === filter),
    [uploads.data, filter],
  );
  const perClass = Object.entries(uploads.data?.counts.pending_per_class ?? {}).map(([name, n]) => ({ name, n }));

  if (uploads.isLoading) return <Skeleton height={400} />;
  if (uploads.isError) return <ErrorState error={uploads.error} what="uploads" />;
  const c = uploads.data!.counts;
  return (
    <div className="stack">
      <div className="grid g4 tight">
        <Stat label="Total uploads" value={int(c.total)} hint="validated images in the database" />
        <Stat label="Pending" value={int(c.pending)} hint="waiting for the next retraining" />
        <Stat label="Used" value={int(c.used)} hint="already used by a retraining job" />
        <Stat label="Classes pending" value={int(perClass.length)} hint="of 11" />
      </div>
      <MongoCard />
      {perClass.length > 0 && (
        <div className="card">
          <span className="eyebrow">Pending images per class</span>
          <div className="chart-box sm" style={{ marginTop: 10 }}>
            <ResponsiveContainer>
              <BarChart data={perClass} margin={{ left: -20, right: 8, bottom: 30 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} height={56} tick={{ fontSize: 10.5 }} />
                <YAxis allowDecimals={false} />
                <Tooltip cursor={{ fill: "var(--surface-2)" }} />
                <Bar dataKey="n" name="pending" fill="#0f766e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      <div className="spread">
        <Segmented label="Filter" value={filter} onChange={setFilter} options={[
          { value: "all", label: `All (${c.total})` }, { value: "pending", label: `Pending (${c.pending})` }, { value: "used", label: `Used (${c.used})` },
        ]} />
        <span className="tiny muted">Newest first · showing up to 200 records · refreshes every 15 s</span>
      </div>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr><th>Image</th><th>File</th><th>Class</th><th className="num">Size</th><th>Uploaded</th><th>By</th><th>Status</th><th>Job</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td><AuthImage id={r.id} /></td>
                <td style={{ maxWidth: 220 }}>
                  <div className="small ellipsis" style={{ fontWeight: 600 }}>{r.filename}</div>
                  <div className="tiny muted mono">#{r.id} · {r.file_hash.slice(0, 10)}</div>
                </td>
                <td className="small nowrap"><Species name={r.class_label} /></td>
                <td className="num">{r.width}×{r.height}<div className="tiny muted">{bytes(r.bytes)}</div></td>
                <td className="small nowrap">{dateTime(r.upload_timestamp)}</td>
                <td className="small">{r.uploaded_by ?? "—"}</td>
                <td><span className={`badge ${r.training_status === "pending" ? "warn" : "ok"}`}>{r.training_status}</span></td>
                <td className="mono tiny">{r.used_in_job ?? "—"}</td>
              </tr>
            ))}
            {!rows.length && <tr><td colSpan={8} className="muted">No records.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
