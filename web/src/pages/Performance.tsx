import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { ArrowDown, ArrowUp, Download } from "lucide-react";
import { API_BASE } from "../api/client";
import { useEvaluation, useGate, useHistory, useVersions } from "../api/hooks";
import type { Evaluation, GateReport, Metrics, PerClass } from "../api/types";
import { info } from "../lib/classes";
import { int, num, pct } from "../lib/format";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { BarCell, Callout, ErrorState, PageHead, Segmented, Skeleton, Species } from "../components/ui";

type Tab = "overview" | "matrix" | "classes" | "training" | "gate";

const HEADLINE: { key: keyof Metrics; label: string; hint: string; lowerBetter?: boolean; percent?: boolean }[] = [
  { key: "accuracy", label: "Accuracy", hint: "share of crops classified correctly", percent: true },
  { key: "macro_precision", label: "Precision", hint: "macro average over classes", percent: true },
  { key: "macro_recall", label: "Recall", hint: "sensitivity, macro average", percent: true },
  { key: "macro_f1", label: "F1-score", hint: "primary metric, macro average", percent: true },
  { key: "roc_auc_macro", label: "ROC-AUC", hint: "one-vs-rest, macro" },
  { key: "pr_auc_macro", label: "PR-AUC", hint: "precision–recall area, macro" },
  { key: "log_loss", label: "Log loss", hint: "cross-entropy, lower is better", lowerBetter: true },
];

function Overview({ data }: { data: Evaluation }) {
  const { test, validation } = data;
  const compare = HEADLINE.filter((m) => m.key !== "log_loss").map((m) => ({
    name: m.label, validation: validation?.[m.key] as number, test: test[m.key] as number,
  }));
  return (
    <div className="stack">
      <div className="grid g4 tight" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 142px), 1fr))" }}>
        {HEADLINE.map((m) => {
          const t = test[m.key] as number;
          const v = validation?.[m.key] as number | undefined;
          const delta = v !== undefined ? t - v : undefined;
          return (
            <div className="card stat" key={m.key}>
              <div className="label">{m.label}</div>
              <div className="value">{m.percent ? (t * 100).toFixed(1) : t.toFixed(3)}{m.percent && <small>%</small>}</div>
              <div className="hint">{m.hint}</div>
              {delta !== undefined && (
                <div className="tiny mono" style={{ marginTop: 6, color: (m.lowerBetter ? delta <= 0 : delta >= 0) ? "var(--green)" : "var(--amber)" }}>
                  {delta >= 0 ? <ArrowUp size={11} /> : <ArrowDown size={11} />} {m.percent ? `${(Math.abs(delta) * 100).toFixed(1)} pts` : Math.abs(delta).toFixed(3)} vs validation
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="grid g2">
        <div className="card">
          <span className="eyebrow">Generalisation gap</span>
          <h3 style={{ margin: "6px 0 10px" }}>Validation vs held-out test</h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <BarChart data={compare} margin={{ left: -16, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
                <YAxis domain={[0.6, 1]} tickFormatter={(v) => v.toFixed(1)} />
                <Tooltip formatter={(v: number) => v.toFixed(3)} cursor={{ fill: "var(--surface-2)" }} />
                <Legend verticalAlign="top" height={28} />
                <Bar dataKey="validation" name="Validation" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                <Bar dataKey="test" name="Test" fill="#0f766e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card stack">
          <span className="eyebrow">Why test is lower</span>
          <h3>Different cameras, not overfitting</h3>
          <p className="small">
            Validation photos come from the same collection as the training photos. The test photos were released separately and
            were taken with a different mix of cameras: they are darker and sharper on average, and half of them come from
            three high-resolution phone formats.
          </p>
          <p className="small">
            Validation-to-test drop in F1: <strong className="mono">{validation ? ((validation.macro_f1 - test.macro_f1) * 100).toFixed(1) : "—"} pts</strong>.
            ROC-AUC barely moves, so the model still ranks the right class highly; it is the top-1 choice between look-alike
            eggs that fails more often. Expect accuracy close to the <strong>test</strong> figure on new microscopes.
          </p>
          <Callout tone="info"><p>Scores are model scores, not calibrated probabilities: log loss rises from {num(validation?.log_loss)} to {num(test.log_loss)} on test.</p></Callout>
        </div>
      </div>

      {data.by_resolution.length > 0 && (
        <div className="card">
          <span className="eyebrow">Accuracy by source camera format</span>
          <h3 style={{ margin: "6px 0 10px" }}>Where the accuracy is lost</h3>
          <div className="chart-box">
            <ResponsiveContainer>
              <BarChart data={data.by_resolution} margin={{ left: -16, right: 8, bottom: 30 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="resolution" angle={-30} textAnchor="end" interval={0} height={50} tick={{ fontSize: 10.5 }} />
                <YAxis domain={[0.6, 1]} />
                <Tooltip formatter={(v: number) => (v === null ? "not in this split" : pct(v))} cursor={{ fill: "var(--surface-2)" }} />
                <Legend verticalAlign="top" height={28} />
                <Bar dataKey="val_accuracy" name="Validation" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                <Bar dataKey="test_accuracy" name="Test" fill="#0f766e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="small muted">Formats without a test bar do not occur in the test photos.</p>
        </div>
      )}

      {data.comparison.length > 0 && (
        <div>
          <div className="section-head"><span className="eyebrow">Model comparison (validation set)</span></div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Model</th><th className="num">Accuracy</th><th className="num">F1</th><th className="num">ROC-AUC</th><th className="num">PR-AUC</th>
                  <th className="num">Log loss</th><th className="num">Params</th><th className="num">Size</th><th className="num">CPU latency</th><th className="num">Train time</th>
                </tr>
              </thead>
              <tbody>
                {data.comparison.map((row) => (
                  <tr key={String(row.version)} className={row.version === data.model_version ? "hl" : ""}>
                    <td>
                      <div className="mono small">{row.version}</div>
                      {row.version === data.model_version && <span className="badge ok" style={{ marginTop: 4 }}>selected</span>}
                    </td>
                    <td className="num">{num(row.Accuracy as number)}</td>
                    <td className="num">{num(row["F1-score"] as number)}</td>
                    <td className="num">{num(row["ROC-AUC"] as number)}</td>
                    <td className="num">{num(row["PR-AUC"] as number)}</td>
                    <td className="num">{num(row["Log loss"] as number)}</td>
                    <td className="num">{((row.parameters as number) / 1e6).toFixed(2)} M</td>
                    <td className="num">{row.size_MB} MB</td>
                    <td className="num">{row.CPU_latency_ms} ms</td>
                    <td className="num">{row.train_minutes} min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="tiny muted" style={{ marginTop: 8 }}>The model with the higher validation F1 is selected; the test set is used once, after selection.</p>
        </div>
      )}
    </div>
  );
}

function MatrixTab({ data }: { data: Evaluation }) {
  const [split, setSplit] = useState<"test" | "validation">("test");
  const [normalise, setNormalise] = useState(false);
  const metrics = split === "test" ? data.test : data.validation;
  if (!metrics) return <ErrorState error={new Error("No validation results")} what="validation results" />;
  return (
    <div className="classify-layout">
      <div className="card" style={{ minWidth: 0 }}>
        <div className="spread" style={{ marginBottom: 10 }}>
          <Segmented label="Split" value={split} onChange={setSplit}
            options={[{ value: "test", label: `Test (${int(data.test.samples)})` }, { value: "validation", label: `Validation (${int(data.validation?.samples)})` }]} />
          <Segmented label="Values" value={normalise ? "pct" : "n"} onChange={(v) => setNormalise(v === "pct")}
            options={[{ value: "n", label: "Counts" }, { value: "pct", label: "Row %" }]} />
        </div>
        <ConfusionMatrix matrix={metrics.confusion_matrix} names={data.class_names} normalise={normalise} />
      </div>
      <div className="card">
        <span className="eyebrow">Top confusions · {split}</span>
        <div className="grid" style={{ marginTop: 12, gap: 10 }}>
          {metrics.top_confusions.slice(0, 6).map((c, i) => {
            const support = metrics.per_class[c.true]?.support ?? 0;
            return (
              <div key={i} className="card tinted" style={{ padding: 12 }}>
                <div className="spread">
                  <span className="mono tiny muted">#{i + 1}</span>
                  <span className="mono small" style={{ color: "var(--amber)" }}>{c.count} / {support}</span>
                </div>
                <div className="small" style={{ marginTop: 4 }}>
                  <strong><Species name={c.true} /></strong> <span className="muted">predicted as</span> <strong><Species name={c.predicted} /></strong>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

type SortKey = keyof PerClass | "name";

function ClassesTab({ data }: { data: Evaluation }) {
  const [split, setSplit] = useState<"test" | "validation">("test");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "f1", desc: true });
  const metrics = split === "test" ? data.test : data.validation;
  const rows = useMemo(() => {
    if (!metrics) return [];
    const list = data.class_names.map((name, id) => ({ id, name, ...metrics.per_class[name] }));
    return list.sort((a, b) => {
      const x = sort.key === "name" ? a.name : (a[sort.key] as number) ?? 0;
      const y = sort.key === "name" ? b.name : (b[sort.key] as number) ?? 0;
      const order = x < y ? -1 : x > y ? 1 : 0;
      return sort.desc ? -order : order;
    });
  }, [metrics, data.class_names, sort]);
  const header = (key: SortKey, label: string, numeric = true) => (
    <th className={`sortable ${numeric ? "num" : ""}`} onClick={() => setSort({ key, desc: sort.key === key ? !sort.desc : key !== "name" && key !== "log_loss" })}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      {label}{sort.key === key ? (sort.desc ? " ↓" : " ↑") : ""}
    </th>
  );
  return (
    <div className="stack">
      <Segmented label="Split" value={split} onChange={setSplit}
        options={[{ value: "test", label: "Test set" }, { value: "validation", label: "Validation set" }]} />
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              {header("name", "Class", false)}{header("precision", "Precision")}{header("recall", "Recall")}{header("f1", "F1-score")}
              {header("roc_auc", "ROC-AUC")}{header("pr_auc", "PR-AUC")}{header("log_loss", "Log loss")}{header("support", "Crops")}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td className="nowrap"><span className="swatch" style={{ background: info(r.id).colour, marginRight: 8 }} /><Species name={r.name} /></td>
                <td className="num">{num(r.precision)}</td>
                <td className="num">{num(r.recall)}</td>
                <td><BarCell value={r.f1} colour={info(r.id).colour} /></td>
                <td className="num">{num(r.roc_auc)}</td>
                <td className="num">{num(r.pr_auc)}</td>
                <td className="num">{num(r.log_loss)}</td>
                <td className="num">{int(r.support)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="tiny muted">Per-class accuracy equals recall. Click a column to sort.</p>
    </div>
  );
}

const CURVES = [
  { key: "f1", label: "F1-score" }, { key: "accuracy", label: "Accuracy" }, { key: "log_loss", label: "Log loss" },
  { key: "roc_auc", label: "ROC-AUC" }, { key: "pr_auc", label: "PR-AUC" }, { key: "precision", label: "Precision" }, { key: "recall", label: "Recall" },
];

function TrainingTab({ data }: { data: Evaluation }) {
  const versions = useVersions();
  const [version, setVersion] = useState(data.model_version);
  const [metric, setMetric] = useState("f1");
  const history = useHistory(version);
  const rows = history.data ?? [];
  const fineTuneStart = rows.find((r) => r.phase === "fine_tune")?.epoch;
  const hasEpochMetrics = rows.length > 0 && `val_${metric}` in rows[0];
  const meta = versions.data?.find((v) => v.version === version);
  return (
    <div className="card stack">
      <div className="spread">
        <div className="row">
          <label className="small muted" htmlFor="version">Model version</label>
          <select id="version" className="input sm" style={{ width: "auto" }} value={version} onChange={(e) => setVersion(e.target.value)}>
            {(versions.data ?? []).map((v) => <option key={v.version} value={v.version}>{v.version}{v.active ? " (active)" : ""}</option>)}
          </select>
        </div>
        <Segmented label="Metric" value={metric} onChange={setMetric} options={CURVES.map((c) => ({ value: c.key, label: c.label }))} />
      </div>
      {history.isLoading && <Skeleton height={320} />}
      {history.isError && <ErrorState error={history.error} what="training history" />}
      {rows.length > 0 && !hasEpochMetrics && (
        <Callout tone="info"><p>This version was trained without per-epoch metric tracking; only loss and accuracy are recorded.</p></Callout>
      )}
      {rows.length > 0 && (
        <div className="chart-box lg">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ left: -10, right: 12, top: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="epoch" />
              <YAxis domain={["auto", "auto"]} tickFormatter={(v) => Number(v).toFixed(2)} />
              <Tooltip formatter={(v: number) => num(v)} labelFormatter={(e) => `Epoch ${e}`} />
              <Legend verticalAlign="top" height={30} />
              {fineTuneStart && <ReferenceLine x={fineTuneStart} stroke="var(--amber)" strokeDasharray="4 4" label={{ value: "fine-tuning", position: "insideTopRight", fill: "var(--amber)", fontSize: 11 }} />}
              <Line type="monotone" dataKey={hasEpochMetrics ? `train_${metric}` : metric === "log_loss" ? "loss" : "accuracy"} name="Training" stroke="#0f766e" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey={hasEpochMetrics ? `val_${metric}` : metric === "log_loss" ? "val_loss" : "val_accuracy"} name="Validation" stroke="#b45309" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {meta && (
        <dl className="kv">
          <dt>Architecture</dt><dd>{meta.architecture}</dd>
          <dt>Created</dt><dd>{new Date(meta.created_at).toLocaleString()}</dd>
          <dt>Validation F1</dt><dd className="mono">{num(meta.validation_metrics?.macro_f1)}</dd>
          <dt>Test F1</dt><dd className="mono">{num(meta.test_metrics?.macro_f1)}</dd>
          {meta.parent_version && <><dt>Retrained from</dt><dd className="mono">{meta.parent_version}</dd></>}
        </dl>
      )}
      <p className="tiny muted">Training metrics are measured on the un-augmented training set after every epoch, so both curves are comparable.</p>
    </div>
  );
}

function GateTab({ gate }: { gate: GateReport | undefined }) {
  if (!gate) return <Skeleton height={300} />;
  if (!gate.enabled) return <Callout tone="warn" title="No egg check is installed"><p>Train it with <code>python -m src.gate --prepare --train</code>.</p></Callout>;
  const sources = Object.entries(gate.test.per_source).map(([source, s]) => ({
    source, test: s.rate, validation: gate.validation.per_source[source]?.rate, meaning: s.rate_meaning, images: s.images,
  }));
  const hist = gate.history as Record<string, number[]>;
  const curve = (hist.loss ?? []).map((loss, i) => ({ epoch: i + 1, loss, val_loss: hist.val_loss?.[i] }));
  const label: Record<string, string> = { egg: "Egg crops (accepted)", background: "Empty microscope background", cifar100: "CIFAR-100 photos", imagenette: "Imagenette photos", synthetic: "Blank / noise images" };
  return (
    <div className="stack">
      <div className="grid g4 tight">
        <div className="card stat"><div className="label">Eggs accepted</div><div className="value">{(gate.test.egg_acceptance_rate * 100).toFixed(1)}<small>%</small></div><div className="hint">{int(gate.test.egg_images)} test egg crops</div></div>
        <div className="card stat"><div className="label">Non-eggs rejected</div><div className="value">{(gate.test.non_egg_rejection_rate * 100).toFixed(1)}<small>%</small></div><div className="hint">{int(gate.test.non_egg_images)} test images</div></div>
        <div className="card stat"><div className="label">ROC-AUC</div><div className="value">{gate.test.roc_auc.toFixed(3)}</div><div className="hint">egg vs not egg</div></div>
        <div className="card stat"><div className="label">Threshold</div><div className="value">{gate.threshold.toFixed(3)}</div><div className="hint">keeps ≥ {pct(gate.target_egg_recall)} of validation eggs</div></div>
      </div>
      <div className="grid g2">
        <div className="card">
          <span className="eyebrow">Per image source (test)</span>
          <div className="grid" style={{ gap: 12, marginTop: 14 }}>
            {sources.map((s) => (
              <div key={s.source}>
                <div className="spread small"><span>{label[s.source] ?? s.source} <span className="muted tiny">· {int(s.images)}</span></span><span className="mono">{pct(s.test)} {s.meaning}</span></div>
                <div className="meter"><span style={{ width: `${s.test * 100}%`, background: s.source === "egg" ? "var(--teal)" : s.test < 0.9 ? "var(--amber)" : "#0f766e" }} /></div>
              </div>
            ))}
          </div>
          <p className="small muted" style={{ marginTop: 14 }}>
            Empty microscope background is the hardest case: it shares the colours and debris of real egg images.
          </p>
        </div>
        <div className="card">
          <span className="eyebrow">Training</span>
          <div className="chart-box sm" style={{ marginTop: 10 }}>
            <ResponsiveContainer>
              <LineChart data={curve} margin={{ left: -14, right: 8 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="epoch" />
                <YAxis tickFormatter={(v) => Number(v).toFixed(2)} />
                <Tooltip formatter={(v: number) => num(v)} labelFormatter={(e) => `Epoch ${e}`} />
                <Legend verticalAlign="top" height={28} />
                <Line dataKey="loss" name="Training loss" stroke="#0f766e" strokeWidth={2} dot={false} />
                <Line dataKey="val_loss" name="Validation loss" stroke="#b45309" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="small muted">{gate.architecture}. Trained on {int(gate.training_images.egg)} egg crops and {int(gate.training_images.non_egg)} non-egg images.</p>
        </div>
      </div>
      {gate.classifier_on_non_eggs.length > 0 && (
        <div>
          <div className="section-head">
            <div>
              <span className="eyebrow">Why the check is needed</span>
              <p className="small muted" style={{ marginTop: 6 }}>Without it, the 11-class model gives confident answers even for images that are not eggs.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Images</th><th className="num">Count</th><th className="num">Median top score</th><th className="num">Score ≥ 0.9</th><th>Most common answer</th></tr></thead>
              <tbody>
                {gate.classifier_on_non_eggs.map((r) => (
                  <tr key={r.source} className={r.source === "egg" ? "hl" : ""}>
                    <td>{label[r.source]?.replace(" (accepted)", "") ?? r.source}</td>
                    <td className="num">{int(r.images)}</td>
                    <td className="num">{num(r.median_confidence)}</td>
                    <td className="num">{pct(r.share_confidence_over_0_9, 0)}</td>
                    <td><Species name={r.most_common_prediction} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Performance() {
  const evaluation = useEvaluation();
  const gate = useGate();
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as Tab) || "overview";
  const setTab = (t: Tab) => setParams(t === "overview" ? {} : { tab: t }, { replace: true });
  const data = evaluation.data;

  return (
    <div className="container">
      <PageHead
        eyebrow={`Model evaluation · held-out test set${data ? ` (${int(data.test.samples)} crops)` : ""}`}
        title="Model performance"
        actions={<a className="btn sm" href={`${API_BASE}/results/evaluation`} target="_blank" rel="noreferrer"><Download size={15} /> Metrics JSON</a>}
      >
        {data ? <>Results for <span className="mono">{data.model_version}</span>, a fine-tuned MobileNetV2, on official test photos it never saw during training or model selection.</> : "Loading results…"}
      </PageHead>
      <div style={{ marginBottom: 18, overflowX: "auto" }}>
        <Segmented label="Section" value={tab} onChange={setTab} options={[
          { value: "overview", label: "Overview" }, { value: "matrix", label: "Confusion matrix" }, { value: "classes", label: "Per class" },
          { value: "training", label: "Training curves" }, { value: "gate", label: "Egg check" },
        ]} />
      </div>
      {evaluation.isLoading && <Skeleton height={420} />}
      {evaluation.isError && tab !== "gate" && <ErrorState error={evaluation.error} what="evaluation results" />}
      {data && tab === "overview" && <Overview data={data} />}
      {data && tab === "matrix" && <MatrixTab data={data} />}
      {data && tab === "classes" && <ClassesTab data={data} />}
      {data && tab === "training" && <TrainingTab data={data} />}
      {tab === "gate" && <GateTab gate={gate.data} />}
    </div>
  );
}
