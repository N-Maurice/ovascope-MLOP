import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, FileArchive, FolderUp, Trash2, Upload, XCircle } from "lucide-react";
import { ApiError, post } from "../../api/client";
import { useClasses, useRetrainStatus } from "../../api/hooks";
import type { UploadResult } from "../../api/types";
import { useAuth } from "../../auth";
import { bytes, int } from "../../lib/format";
import { Callout, Meter, Species } from "../../components/ui";
import { useToast } from "../../components/Toast";

type Item = { file: File; preview: string | null; label: string };

const isZip = (f: File) => f.name.toLowerCase().endsWith(".zip");

export default function UploadPanel({ onDone }: { onDone: () => void }) {
  const classes = useClasses();
  const status = useRetrainStatus();
  const client = useQueryClient();
  const toast = useToast();
  const { signOut } = useAuth();
  const input = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [defaultLabel, setDefaultLabel] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => items.forEach((i) => i.preview && URL.revokeObjectURL(i.preview)), [items]);

  const add = (files: FileList | File[]) => {
    const list = Array.from(files).map((file) => ({
      file, preview: isZip(file) ? null : URL.createObjectURL(file), label: isZip(file) ? "" : defaultLabel,
    }));
    setItems((old) => [...old, ...list]);
    setResult(null);
  };

  const setAll = (label: string) => {
    setDefaultLabel(label);
    setItems((old) => old.map((i) => (isZip(i.file) ? i : { ...i, label })));
  };

  const images = items.filter((i) => !isZip(i.file));
  const zips = items.filter((i) => isZip(i.file));
  const missing = images.filter((i) => i.label === "").length;

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      // The API takes one label per file (comma-separated, same order). ZIP archives carry their
      // labels in their folder names, so they are sent in a separate request.
      let combined: UploadResult | null = null;
      const merge = (r: UploadResult) => {
        combined = combined
          ? { ...r, accepted: [...combined.accepted, ...r.accepted], rejected: [...combined.rejected, ...r.rejected], saved_to_database: combined.saved_to_database + r.saved_to_database }
          : r;
      };
      if (images.length) {
        const body = new FormData();
        images.forEach((i) => body.append("files", i.file, i.file.name));
        body.append("labels", images.map((i) => i.label).join(","));
        merge(await post<UploadResult>("/upload", body));
      }
      for (const zip of zips) {
        const body = new FormData();
        body.append("files", zip.file, zip.file.name);
        merge(await post<UploadResult>("/upload", body));
      }
      const final = combined as UploadResult | null;
      setResult(final);
      setItems([]);
      client.invalidateQueries({ queryKey: ["uploads"] });
      client.invalidateQueries({ queryKey: ["retrain-status"] });
      if (final) toast(`${final.saved_to_database} image(s) saved, ${final.rejected.length} rejected`, final.rejected.length ? "bad" : "ok");
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        signOut();
        return;
      }
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const s = status.data;
  return (
    <div className="classify-layout">
      <div className="stack">
        <div
          className={`dropzone ${over ? "over" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}
        >
          <div className="icon"><FolderUp size={26} /></div>
          <h3>Add labelled egg crops</h3>
          <p className="muted small">Images (JPG, PNG, BMP, TIFF, ≤ 10 MB each) or a ZIP with one folder per class, e.g. <span className="mono">00_ascaris_lumbricoides/img1.jpg</span></p>
          <input ref={input} type="file" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.zip,image/*" hidden onChange={(e) => e.target.files && add(e.target.files)} />
        </div>

        {items.length > 0 && (
          <div className="card stack">
            <div className="spread">
              <span className="eyebrow">{items.length} file(s) ready</span>
              <div className="row">
                <select className="input sm" style={{ width: "auto", maxWidth: 230 }} value={defaultLabel} onChange={(e) => setAll(e.target.value)} aria-label="Label for all images">
                  <option value="">Set the class for all images…</option>
                  {classes.data?.map((c) => <option key={c.class_id} value={String(c.class_id)}>{c.display_name}</option>)}
                </select>
                <button className="btn sm ghost" onClick={() => setItems([])}><Trash2 size={14} /> Clear</button>
              </div>
            </div>
            <div className="file-list">
              {items.map((item, index) => (
                <div className="file-row" key={index}>
                  {item.preview ? <img className="thumb-sm" src={item.preview} alt="" /> : <div className="thumb-sm" style={{ display: "grid", placeItems: "center" }}><FileArchive size={20} /></div>}
                  <div style={{ minWidth: 0 }}>
                    <div className="small ellipsis" style={{ fontWeight: 600 }}>{item.file.name}</div>
                    <div className="tiny muted">{bytes(item.file.size)}{isZip(item.file) ? " · labels from folder names" : ""}</div>
                  </div>
                  {isZip(item.file) ? <span className="badge mono">ZIP</span> : (
                    <select className="input sm" value={item.label} aria-label={`Class of ${item.file.name}`}
                      onChange={(e) => setItems((old) => old.map((x, i) => (i === index ? { ...x, label: e.target.value } : x)))}
                      style={{ borderColor: item.label === "" ? "var(--amber)" : undefined }}>
                      <option value="">Choose class…</option>
                      {classes.data?.map((c) => <option key={c.class_id} value={String(c.class_id)}>{c.display_name}</option>)}
                    </select>
                  )}
                  <button className="icon-btn" style={{ width: 32, height: 32 }} aria-label={`Remove ${item.file.name}`}
                    onClick={() => setItems((old) => old.filter((_, i) => i !== index))}>
                    <XCircle size={16} />
                  </button>
                </div>
              ))}
            </div>
            {missing > 0 && <p className="small" style={{ color: "var(--amber)" }}>{missing} image(s) still need a class.</p>}
            <button className="btn teal lg" disabled={busy || missing > 0} onClick={submit}>
              {busy ? <span className="spinner" /> : <Upload size={18} />} {busy ? "Uploading and checking…" : `Upload ${items.length} file(s)`}
            </button>
          </div>
        )}
        {error && <Callout tone="bad" title="Upload failed"><p>{error}</p></Callout>}

        {result && (
          <div className="card stack">
            <div className="spread">
              <span className="eyebrow">Upload result</span>
              <button className="btn sm" onClick={onDone}>View database</button>
            </div>
            <p><strong>{result.saved_to_database}</strong> saved to the database · <strong>{result.rejected.length}</strong> rejected</p>
            <div className="file-list" style={{ maxHeight: 300 }}>
              {result.accepted.map((a) => (
                <div key={`a${a.id}`} className="row small"><CheckCircle2 size={16} color="var(--green)" /> <span className="ellipsis">{a.filename}</span> → <Species name={a.class_label} /></div>
              ))}
              {result.rejected.map((r, i) => (
                <div key={`r${i}`} className="row small" style={{ alignItems: "flex-start" }}>
                  <XCircle size={16} color="var(--red)" style={{ flex: "none", marginTop: 3 }} />
                  <span><strong>{r.filename}</strong>: {r.reason}</span>
                </div>
              ))}
            </div>
            {result.retraining.auto_started_job && <Callout tone="info"><p>Automatic retraining started (job {result.retraining.auto_started_job}).</p></Callout>}
          </div>
        )}
      </div>

      <div className="stack">
        {s && (
          <div className="card">
            <span className="eyebrow">Retraining readiness</span>
            <div className="stat" style={{ marginTop: 10 }}>
              <div className="value">{int(s.pending_new_images)}<small>/ {s.trigger_threshold} new images</small></div>
            </div>
            <Meter value={s.pending_new_images / s.trigger_threshold} />
            <p className="small muted" style={{ marginTop: 10 }}>
              Retraining needs at least {s.min_new_images} new images and is recommended from {s.trigger_threshold}.
            </p>
          </div>
        )}
        <Callout tone="info" title="What happens to uploads">
          <ul>
            <li>Every file is checked: format, size, readable image, known class.</li>
            <li>Exact duplicates and protected test images are refused.</li>
            <li>The egg check refuses images that do not look like a parasite egg.</li>
            <li>Accepted images are stored in the SQLite database as <em>pending</em> until a retraining job uses them.</li>
          </ul>
        </Callout>
      </div>
    </div>
  );
}
