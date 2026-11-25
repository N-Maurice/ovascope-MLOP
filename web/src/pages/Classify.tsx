import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Cropper, { type Area } from "react-easy-crop";
import { Link } from "react-router-dom";
import {
  Camera, CheckCircle2, ClipboardPaste, Crop, History, ImagePlus, RotateCcw, ScanSearch, ShieldAlert, Trash2, Upload, ZoomIn,
} from "lucide-react";
import { ApiError, exampleUrl, request } from "../api/client";
import { useClasses, useEvaluation, useHealth } from "../api/hooks";
import type { NotEggDetail, Prediction } from "../api/types";
import { info } from "../lib/classes";
import { dateTime, ms, num, pct } from "../lib/format";
import { sessionHistory, useSessionHistory, type HistoryEntry } from "../lib/session";
import { Callout, PageHead, Species } from "../components/ui";

const ACCEPT = ".jpg,.jpeg,.png,.bmp,.tif,.tiff,image/*";
const MAX_MB = 10;

type Source = { file: File; url: string; width: number; height: number };

async function loadSource(file: File): Promise<Source> {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  await image.decode().catch(() => {
    throw new Error("This file could not be opened as an image. Use JPG, PNG, BMP or TIFF.");
  });
  return { file, url, width: image.naturalWidth, height: image.naturalHeight };
}

/** Cut the selected square out of the image (max 800 px side) and encode it as JPEG. */
async function cropToBlob(source: Source, area: Area): Promise<Blob> {
  const image = new Image();
  image.src = source.url;
  await image.decode();
  const side = Math.min(800, Math.round(area.width));
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = Math.round((area.height / area.width) * side);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, area.x, area.y, area.width, area.height, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Crop failed"))), "image/jpeg", 0.95));
}

function Dropzone({ onFile }: { onFile: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) onFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [onFile]);

  return (
    <div className="stack">
      <div
        className={`dropzone ${over ? "over" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const file = e.dataTransfer.files[0];
          if (file) onFile(file);
        }}
      >
        <div className="icon"><Upload size={26} /></div>
        <h3>Drop a microscope image here</h3>
        <p className="muted small">or click to choose a file · JPG, PNG, BMP, TIFF · up to {MAX_MB} MB</p>
        <p className="muted tiny row" style={{ justifyContent: "center", gap: 6 }}>
          <ClipboardPaste size={13} /> You can also paste an image from the clipboard
        </p>
      </div>
      <div className="row show-sm">
        <button className="btn block" onClick={() => camera.current?.click()}><Camera size={18} /> Take a photo</button>
      </div>
      <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
      <input ref={camera} type="file" accept="image/*" capture="environment" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
    </div>
  );
}

function Examples({ onPick }: { onPick: (classId: number) => void }) {
  return (
    <div className="card flat">
      <div className="card-title">
        <span className="eyebrow">No image at hand?</span>
      </div>
      <p className="small muted" style={{ marginBottom: 12 }}>Try a validation crop from the dataset (already cropped around one egg).</p>
      <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(64px, 1fr))" }}>
        {[0, 2, 4, 6, 9, 10].map((id) => (
          <button key={id} onClick={() => onPick(id)} style={{ padding: 0, border: 0, background: "none", cursor: "pointer" }} title={`Example: ${info(id).short}`}>
            <img src={exampleUrl(id, 5)} alt={`Example ${info(id).short}`} loading="lazy" />
          </button>
        ))}
      </div>
    </div>
  );
}

function confusedWith(names: string[], matrix: number[][] | undefined, classId: number) {
  if (!matrix) return null;
  let best = -1, count = 0;
  names.forEach((_, j) => {
    if (j === classId) return;
    const n = (matrix[classId]?.[j] ?? 0) + (matrix[j]?.[classId] ?? 0);
    if (n > count) { count = n; best = j; }
  });
  return best >= 0 ? { id: best, name: names[best], count } : null;
}

function PredictionView({ entry }: { entry: HistoryEntry & { outcome: { kind: "prediction"; prediction: Prediction } } }) {
  const p = entry.outcome.prediction;
  const evaluation = useEvaluation();
  const names = evaluation.data?.class_names ?? [];
  const lookAlike = confusedWith(names, evaluation.data?.test.confusion_matrix, p.predicted_class_id);
  const testF1 = evaluation.data?.test.per_class[p.predicted_class]?.f1;
  const low = p.confidence < 0.6;
  const colour = info(p.predicted_class_id).colour;
  return (
    <div className="stack">
      <div className="card pad-lg">
        <div className="spread" style={{ marginBottom: 14 }}>
          <span className="eyebrow">Prediction</span>
          <span className={`badge ${low ? "warn" : "ok"}`}>{low ? "Low confidence" : "Egg check passed"}</span>
        </div>
        <div className="result-head">
          <img src={entry.thumbnail} alt="Image that was classified" />
          <div style={{ minWidth: 0 }}>
            <div className="mono tiny muted">#{String(p.predicted_class_id).padStart(2, "0")} · {info(p.predicted_class_id).common}</div>
            <h2 style={{ margin: "4px 0 6px" }}><Species name={p.predicted_class} /></h2>
            <div className="row" style={{ gap: 6 }}>
              <span className="big-conf" style={{ color: colour }}>{(p.confidence * 100).toFixed(1)}</span>
              <span className="muted">% model score</span>
            </div>
          </div>
        </div>

        <div className="topk" style={{ marginTop: 22 }}>
          {p.top_k.map((k) => (
            <div className="line" key={k.class_id}>
              <span className="ellipsis"><Species name={k.class_name} /></span>
              <span className="mono small">{pct(k.confidence)}</span>
              <div className="track"><span style={{ width: `${k.confidence * 100}%`, background: info(k.class_id).colour }} /></div>
            </div>
          ))}
        </div>

        {low && (
          <div style={{ marginTop: 16 }}>
            <Callout tone="warn" title="The model is unsure">
              <p>Several classes look similar to the model. Try a tighter crop around a sharp, well-lit egg, or confirm manually.</p>
            </Callout>
          </div>
        )}

        <dl className="kv" style={{ marginTop: 20 }}>
          <dt>Egg check</dt>
          <dd>
            {p.input_check.enabled
              ? <>score {num(p.input_check.egg_score)} ≥ threshold {num(p.input_check.threshold)}</>
              : "not enabled on this server"}
          </dd>
          <dt>Model</dt>
          <dd className="mono">{p.model_version}</dd>
          <dt>Response time</dt>
          <dd>{ms(entry.latencyMs)}{entry.servedBy ? <span className="muted"> · served by {entry.servedBy}</span> : null}</dd>
        </dl>
      </div>

      <div className="card">
        <div className="card-title">
          <span className="eyebrow">Compare with reference crops</span>
          <Link to="/classes" className="small">Class atlas</Link>
        </div>
        <p className="small muted" style={{ marginBottom: 10 }}>
          Validation images of <Species name={p.predicted_class} />. {info(p.predicted_class_id).features}
        </p>
        <div className="ref-strip">
          {[0, 1, 2, 3].map((i) => <img key={i} src={exampleUrl(p.predicted_class_id, i)} alt={`${p.predicted_class} reference ${i + 1}`} loading="lazy" />)}
        </div>
        {lookAlike && (
          <>
            <p className="small" style={{ margin: "14px 0 10px" }}>
              <strong>Known look-alike:</strong> on the test set this class was mixed up with <Species name={lookAlike.name} /> {lookAlike.count} times
              {testF1 !== undefined && <> (class test F1 {testF1.toFixed(2)})</>}.
            </p>
            <div className="ref-strip">
              {[0, 1, 2, 3].map((i) => <img key={i} src={exampleUrl(lookAlike.id, i)} alt={`${lookAlike.name} reference ${i + 1}`} loading="lazy" />)}
            </div>
          </>
        )}
      </div>
      <p className="tiny muted">{p.note}</p>
    </div>
  );
}

function RejectedView({ entry, detail }: { entry: HistoryEntry; detail: NotEggDetail }) {
  return (
    <div className="card pad-lg stack">
      <div className="spread">
        <span className="eyebrow">Image refused</span>
        <span className="badge warn"><ShieldAlert size={13} /> Not a parasite egg</span>
      </div>
      <div className="result-head">
        <img src={entry.thumbnail} alt="Image that was refused" />
        <div>
          <h3>This doesn't look like a parasite egg</h3>
          <p className="small muted" style={{ marginTop: 4 }}>
            Egg score <span className="mono">{num(detail.egg_score)}</span> is below the threshold <span className="mono">{num(detail.threshold)}</span>,
            so the image was not classified.
          </p>
        </div>
      </div>
      <Callout tone="warn" title="To get a prediction">
        <ul>
          <li>Upload a <strong>microscope</strong> photo (stool smear), not a photo of a person, object or scene.</li>
          <li>Crop tightly around <strong>one</strong> egg, so it fills most of the square.</li>
          <li>Empty background, debris or a very blurred egg can also be refused. Refocus and try again.</li>
        </ul>
      </Callout>
      <p className="tiny muted">{detail.message}</p>
    </div>
  );
}

export default function Classify() {
  const health = useHealth();
  const classes = useClasses();
  const history = useSessionHistory();
  const [source, setSource] = useState<Source | null>(null);
  const [wholeImage, setWholeImage] = useState(false);
  const [crop, setCrop] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [area, setArea] = useState<Area | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const current = useMemo(() => history.find((h) => h.id === selected) ?? null, [history, selected]);

  const pick = useCallback(async (file: File, alreadyCropped = false) => {
    setError(null);
    if (file.size > MAX_MB * 1_000_000) return setError(`The file is larger than ${MAX_MB} MB.`);
    try {
      const loaded = await loadSource(file);
      setSource((old) => { if (old) URL.revokeObjectURL(old.url); return loaded; });
      // Small, roughly square images are probably crops already; large photos need cropping.
      const squareish = Math.abs(loaded.width / loaded.height - 1) < 0.2 && Math.max(loaded.width, loaded.height) <= 1000;
      setWholeImage(alreadyCropped || squareish);
      setZoom(1);
      setCrop({ x: 0, y: 0 });
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const pickExample = async (classId: number) => {
    const response = await fetch(exampleUrl(classId, 5));
    if (!response.ok) return setError("Example images are not available on this server.");
    const blob = await response.blob();
    pick(new File([blob], `example_${info(classId).short.replace(/\W+/g, "_").toLowerCase()}.jpg`, { type: "image/jpeg" }), true);
  };

  const classify = async () => {
    if (!source) return;
    setBusy(true);
    setError(null);
    try {
      const blob = wholeImage || !area ? source.file : await cropToBlob(source, area);
      const body = new FormData();
      body.append("file", blob, wholeImage ? source.file.name : `crop_${source.file.name.replace(/\.\w+$/, "")}.jpg`);
      const thumbnail = URL.createObjectURL(blob);
      const start = performance.now();
      try {
        const { data, headers } = await request<Prediction>("/predict", { method: "POST", body });
        const entry = sessionHistory.add({
          at: new Date(), fileName: source.file.name, thumbnail, latencyMs: performance.now() - start,
          servedBy: headers.get("X-Served-By"), outcome: { kind: "prediction", prediction: data },
        });
        setSelected(entry.id);
      } catch (e) {
        if (e instanceof ApiError && e.status === 422 && (e.detail as NotEggDetail)?.error === "not_a_parasite_egg") {
          const entry = sessionHistory.add({
            at: new Date(), fileName: source.file.name, thumbnail, latencyMs: performance.now() - start, servedBy: null,
            outcome: { kind: "rejected", detail: e.detail as NotEggDetail },
          });
          setSelected(entry.id);
        } else {
          URL.revokeObjectURL(thumbnail);
          throw e;
        }
      }
      window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    if (source) URL.revokeObjectURL(source.url);
    setSource(null);
    setError(null);
  };

  const unavailable = health.isError || (health.data && !health.data.model_loaded);

  return (
    <div className="container">
      <PageHead eyebrow="Classifier" title="Classify a parasite egg">
        Upload a microscope image, crop it around one egg and get the model's top-3 classes. Images are processed by the
        OvaScope API and are not stored.
      </PageHead>

      {unavailable && (
        <div style={{ marginBottom: 16 }}>
          <Callout tone="bad" title="Classification is unavailable">
            <p>{health.isError ? "The API cannot be reached." : `No model is loaded: ${health.data?.model_error ?? "unknown error"}`}</p>
          </Callout>
        </div>
      )}

      <div className="classify-layout">
        <div className="stack">
          {!source ? (
            <>
              <Dropzone onFile={pick} />
              <Examples onPick={pickExample} />
            </>
          ) : (
            <div className="card stack">
              <div className="spread">
                <span className="eyebrow"><Crop size={13} /> {wholeImage ? "Whole image" : "Crop one egg"}</span>
                <span className="mono tiny muted ellipsis" style={{ maxWidth: "60%" }}>{source.file.name} · {source.width}×{source.height}</span>
              </div>
              {wholeImage ? (
                <div className="crop-area" style={{ display: "grid", placeItems: "center" }}>
                  <img src={source.url} alt="Selected image" style={{ maxHeight: "100%", objectFit: "contain" }} />
                </div>
              ) : (
                <>
                  <div className="crop-area">
                    <Cropper
                      image={source.url}
                      crop={crop}
                      zoom={zoom}
                      maxZoom={8}
                      aspect={1}
                      onCropChange={setCrop}
                      onZoomChange={setZoom}
                      onCropComplete={(_, pixels) => setArea(pixels)}
                      showGrid
                    />
                  </div>
                  <label className="row small" style={{ gap: 10 }}>
                    <ZoomIn size={16} className="muted" />
                    <input type="range" min={1} max={8} step={0.05} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} aria-label="Zoom" />
                  </label>
                  <p className="tiny muted">Drag to move, pinch or scroll to zoom. Centre one egg so it fills most of the square.</p>
                </>
              )}
              <label className="row small" style={{ gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={wholeImage} onChange={(e) => setWholeImage(e.target.checked)} />
                The image is already cropped around one egg (send it as it is)
              </label>
              <div className="row">
                <button className="btn teal lg" onClick={classify} disabled={busy || Boolean(unavailable)}>
                  {busy ? <span className="spinner" /> : <ScanSearch size={18} />}
                  {busy ? "Classifying…" : "Classify"}
                </button>
                <button className="btn" onClick={reset} disabled={busy}><ImagePlus size={17} /> Choose another</button>
              </div>
            </div>
          )}
          {error && <Callout tone="bad" title="Something went wrong"><p>{error}</p></Callout>}
        </div>

        <div className="stack" ref={resultRef} style={{ scrollMarginTop: 80 }}>
          {current ? (
            current.outcome.kind === "prediction"
              ? <PredictionView entry={current as HistoryEntry & { outcome: { kind: "prediction"; prediction: Prediction } }} />
              : <RejectedView entry={current} detail={current.outcome.detail} />
          ) : (
            <div className="card tinted empty" style={{ minHeight: 260, alignContent: "center" }}>
              <ScanSearch size={34} />
              <h3>Results appear here</h3>
              <p className="small">You will see the predicted class, the top-3 ranking, the egg check and reference images.</p>
              {classes.data && <p className="tiny">{classes.data.length} classes · model {health.data?.model_version ?? "…"}</p>}
            </div>
          )}

          {history.length > 0 && (
            <div className="card">
              <div className="card-title">
                <span className="eyebrow"><History size={13} /> This session</span>
                <button className="btn sm ghost" onClick={() => { sessionHistory.clear(); setSelected(null); }}>
                  <Trash2 size={14} /> Clear
                </button>
              </div>
              <div className="grid" style={{ gap: 4 }}>
                {history.map((h) => (
                  <div key={h.id} className={`history-item ${h.id === selected ? "on" : ""}`} onClick={() => setSelected(h.id)} role="button" tabIndex={0}
                    onKeyDown={(e) => e.key === "Enter" && setSelected(h.id)}>
                    <img src={h.thumbnail} alt="" />
                    <div style={{ minWidth: 0 }}>
                      <div className="ellipsis small" style={{ fontWeight: 600 }}>
                        {h.outcome.kind === "prediction" ? <Species name={h.outcome.prediction.predicted_class} /> : "Refused: not an egg"}
                      </div>
                      <div className="tiny muted ellipsis">{h.fileName} · {dateTime(h.at.toISOString())}</div>
                    </div>
                    {h.outcome.kind === "prediction"
                      ? <span className="mono small">{pct(h.outcome.prediction.confidence, 0)}</span>
                      : <ShieldAlert size={16} color="var(--amber)" />}
                  </div>
                ))}
              </div>
              <p className="tiny muted" style={{ marginTop: 8 }}><RotateCcw size={11} /> Kept in this tab only; cleared on reload.</p>
            </div>
          )}
          {!current && history.length === 0 && (
            <p className="tiny muted row" style={{ gap: 6 }}><CheckCircle2 size={13} /> Non-egg images are refused before classification.</p>
          )}
        </div>
      </div>
    </div>
  );
}
