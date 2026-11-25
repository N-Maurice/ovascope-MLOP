import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { exampleUrl } from "../api/client";
import { useClasses, useDataset, useEvaluation } from "../api/hooks";
import type { Evaluation, PerClass } from "../api/types";
import { info } from "../lib/classes";
import { int, num } from "../lib/format";
import { ErrorState, Modal, PageHead, Segmented, Skeleton, Species } from "../components/ui";

type Group = "All" | "Nematode" | "Cestode" | "Trematode";
type Sort = "id" | "best" | "worst";

function confusions(evaluation: Evaluation, classId: number) {
  const m = evaluation.test.confusion_matrix;
  const names = evaluation.class_names;
  const asOther = names.map((name, j) => ({ id: j, name, n: j === classId ? 0 : m[classId]?.[j] ?? 0 })).filter((x) => x.n > 0).sort((a, b) => b.n - a.n);
  const fromOther = names.map((name, j) => ({ id: j, name, n: j === classId ? 0 : m[j]?.[classId] ?? 0 })).filter((x) => x.n > 0).sort((a, b) => b.n - a.n);
  return { asOther: asOther.slice(0, 3), fromOther: fromOther.slice(0, 3) };
}

const ROWS: [keyof PerClass, string][] = [
  ["accuracy", "Accuracy (recall)"], ["precision", "Precision"], ["f1", "F1-score"], ["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["log_loss", "Log loss"],
];

function ClassDetail({ classId, name, onClose }: { classId: number; name: string; onClose: () => void }) {
  const evaluation = useEvaluation();
  const dataset = useDataset();
  const c = info(classId);
  const val = evaluation.data?.validation?.per_class[name];
  const test = evaluation.data?.test.per_class[name];
  const mix = evaluation.data ? confusions(evaluation.data, classId) : null;
  const counts = dataset.data?.crops_per_split_class;
  return (
    <Modal onClose={onClose} label={name}>
      <div className="stack">
        <div>
          <span className="mono tiny muted">#{String(classId).padStart(2, "0")} · {c.group}</span>
          <h2 style={{ margin: "4px 0" }}><Species name={name} /></h2>
          <p className="muted">{c.common}</p>
        </div>
        <div className="gallery">
          {Array.from({ length: 8 }, (_, i) => <img key={i} src={exampleUrl(classId, i)} alt={`${name} example ${i + 1}`} loading="lazy" />)}
        </div>
        <p className="tiny muted">Validation crops from the dataset. Test crops are never displayed.</p>
        <div className="card tinted">
          <span className="eyebrow plain">Typical features (textbook)</span>
          <p style={{ marginTop: 6 }}>{c.features}</p>
          <p className="tiny muted" style={{ marginTop: 6 }}>For orientation only; confirm identifications with a reference atlas.</p>
        </div>
        <div className="grid g2">
          <div>
            <h4 style={{ marginBottom: 8 }}>Model performance</h4>
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Metric</th><th className="num">Validation</th><th className="num">Test</th></tr></thead>
                <tbody>
                  {ROWS.map(([key, label]) => (
                    <tr key={key}>
                      <td>{label}</td>
                      <td className="num">{num(val?.[key] as number)}</td>
                      <td className="num">{num(test?.[key] as number)}</td>
                    </tr>
                  ))}
                  <tr><td>Crops evaluated</td><td className="num">{int(val?.support)}</td><td className="num">{int(test?.support)}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div className="stack" style={{ gap: 12 }}>
            <div>
              <h4 style={{ marginBottom: 8 }}>Dataset crops</h4>
              <dl className="kv">
                <dt>Training</dt><dd className="mono">{int(counts?.train?.[classId])}</dd>
                <dt>Validation</dt><dd className="mono">{int(counts?.val?.[classId])}</dd>
                <dt>Test (protected)</dt><dd className="mono">{int(counts?.test?.[classId])}</dd>
              </dl>
            </div>
            {mix && (
              <div>
                <h4 style={{ marginBottom: 8 }}>Test-set mix-ups</h4>
                <p className="small"><strong>Predicted as another class:</strong></p>
                <ul className="small" style={{ margin: "4px 0 8px", paddingLeft: 18 }}>
                  {mix.asOther.length ? mix.asOther.map((x) => <li key={x.id}><Species name={x.name} /> — {x.n}</li>) : <li>none</li>}
                </ul>
                <p className="small"><strong>Other classes predicted as this:</strong></p>
                <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                  {mix.fromOther.length ? mix.fromOther.map((x) => <li key={x.id}><Species name={x.name} /> — {x.n}</li>) : <li>none</li>}
                </ul>
              </div>
            )}
          </div>
        </div>
        <Link to="/classify" className="btn teal" style={{ justifySelf: "start" }}>Classify an image <ArrowRight size={16} /></Link>
      </div>
    </Modal>
  );
}

export default function Classes() {
  const classes = useClasses();
  const evaluation = useEvaluation();
  const [group, setGroup] = useState<Group>("All");
  const [sort, setSort] = useState<Sort>("id");
  const [open, setOpen] = useState<number | null>(null);

  const list = useMemo(() => {
    const rows = (classes.data ?? []).map((c) => ({
      ...c,
      info: info(c.class_id),
      f1: evaluation.data?.test.per_class[c.display_name]?.f1,
      valF1: evaluation.data?.validation?.per_class[c.display_name]?.f1,
    }));
    const filtered = group === "All" ? rows : rows.filter((r) => r.info.group === group);
    if (sort === "best") filtered.sort((a, b) => (b.f1 ?? 0) - (a.f1 ?? 0));
    if (sort === "worst") filtered.sort((a, b) => (a.f1 ?? 0) - (b.f1 ?? 0));
    return filtered;
  }, [classes.data, evaluation.data, group, sort]);

  return (
    <div className="container">
      <PageHead eyebrow="Taxonomic repertoire" title="The 11 egg classes">
        Every class the model knows, with real dataset images, typical features and how reliably the model recognises it.
        Select a class for details.
      </PageHead>
      <div className="spread" style={{ marginBottom: 16 }}>
        <Segmented label="Group" value={group} onChange={setGroup}
          options={(["All", "Nematode", "Cestode", "Trematode"] as Group[]).map((g) => ({ value: g, label: g === "All" ? "All 11" : `${g}s` }))} />
        <Segmented label="Sort" value={sort} onChange={setSort}
          options={[{ value: "id", label: "By id" }, { value: "best", label: "Best F1" }, { value: "worst", label: "Weakest" }]} />
      </div>
      {classes.isLoading && <div className="grid g3">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} height={300} />)}</div>}
      {classes.isError && <ErrorState error={classes.error} what="classes" />}
      <div className="grid g3">
        {list.map((c) => {
          const f1 = c.f1;
          const tone = f1 === undefined ? "" : f1 >= 0.85 ? "ok" : f1 >= 0.75 ? "teal" : "warn";
          return (
            <button key={c.class_id} className="card class-card" onClick={() => setOpen(c.class_id)}>
              <div className="thumb">
                <img src={exampleUrl(c.class_id, 1)} alt={`${c.display_name} example`} loading="lazy" />
                <span className="badge mono">#{String(c.class_id).padStart(2, "0")}</span>
              </div>
              <div className="body">
                <div className="spread">
                  <span className="row" style={{ gap: 8 }}>
                    <span className="swatch" style={{ background: c.info.colour }} />
                    <span className="tiny muted mono">{c.info.group.toUpperCase()}</span>
                  </span>
                  {f1 !== undefined && <span className={`badge ${tone} mono`}>test F1 {f1.toFixed(2)}</span>}
                </div>
                <h3><Species name={c.display_name} /></h3>
                <p className="small muted">{c.info.common}</p>
                <p className="small" style={{ color: "var(--ink-2)" }}>{c.info.features}</p>
                {f1 !== undefined && (
                  <div className="meter" title={`validation F1 ${c.valF1?.toFixed(3)} · test F1 ${f1.toFixed(3)}`}>
                    <span style={{ width: `${f1 * 100}%`, background: c.info.colour }} />
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
      {open !== null && classes.data && (
        <ClassDetail classId={open} name={classes.data[open].display_name} onClose={() => setOpen(null)} />
      )}
    </div>
  );
}
