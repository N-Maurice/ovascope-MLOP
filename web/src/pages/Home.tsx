import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, BarChart3, Crop, ScanSearch, ShieldCheck, Sparkles } from "lucide-react";
import { exampleUrl } from "../api/client";
import { useClasses, useDataset, useEvaluation, useGate } from "../api/hooks";
import { CLASS_INFO, info } from "../lib/classes";
import { int, pct } from "../lib/format";
import { Callout, Skeleton, Species, Stat } from "../components/ui";

function Scope() {
  const [index, setIndex] = useState(0);
  const classes = useClasses();
  useEffect(() => {
    const timer = window.setInterval(() => setIndex((i) => (i + 1) % CLASS_INFO.length), 4500);
    return () => window.clearInterval(timer);
  }, []);
  const name = classes.data?.[index]?.display_name ?? info(index).short;
  return (
    <div>
      <div className="scope">
        <img key={index} src={exampleUrl(index, 2)} alt={`Validation crop of ${name}`} />
        <div className="scope-label">
          <span className="mono tiny muted">#{String(index).padStart(2, "0")}</span>
          <strong style={{ fontSize: 14 }}><Species name={name} /></strong>
          <span className="badge teal mono">val crop</span>
        </div>
      </div>
      <div className="scope-dots" role="tablist" aria-label="Example classes">
        {CLASS_INFO.map((c, i) => (
          <button key={c.short} className={i === index ? "on" : ""} onClick={() => setIndex(i)} aria-label={`Show ${c.short}`} />
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const evaluation = useEvaluation();
  const dataset = useDataset();
  const gate = useGate();
  const test = evaluation.data?.test;
  const comparison = evaluation.data?.comparison.find((row) => row.version === evaluation.data?.model_version);
  const names = evaluation.data?.class_names ?? [];
  const perClass = names.map((name, id) => ({ id, name, f1: test?.per_class[name]?.f1 ?? 0 })).sort((a, b) => b.f1 - a.f1);

  return (
    <div className="container">
      <section className="hero">
        <div>
          <span className="eyebrow">AI-assisted microscopy</span>
          <h1>
            Identify parasite eggs from <em>microscope images</em>
          </h1>
          <p className="lead">
            Crop one egg, and OvaScope's MobileNetV2 model ranks 11 egg classes in a fraction of a second. It first checks
            that the image really shows a parasite egg, and it reports its accuracy honestly, including where it struggles.
          </p>
          <div className="row">
            <Link to="/classify" className="btn primary lg">Classify an image <ArrowRight size={18} /></Link>
            <Link to="/performance" className="btn lg"><BarChart3 size={18} /> View performance</Link>
          </div>
          <p className="small muted" style={{ marginTop: 16 }}>
            Research prototype for education and research. Not a diagnostic device.
          </p>
        </div>
        <Scope />
      </section>

      <section className="section" style={{ marginTop: 44 }}>
        <div className="section-head">
          <span className="eyebrow">Benchmark at a glance</span>
          <span className="badge mono">{evaluation.data?.model_version ?? "…"}</span>
        </div>
        <div className="grid g4 tight">
          <Stat label="Classes" value="11" hint="helminth egg types" />
          <Stat label="Dataset" value={dataset.data ? int(dataset.data.crops_total) : "…"} unit="crops" hint="Chula-ParasiteEgg-11, balanced" />
          <Stat label="Test accuracy" value={test ? (test.accuracy * 100).toFixed(1) : "…"} unit="%" hint={`${test ? int(test.samples) : "…"} unseen test crops`} meter={test?.accuracy} />
          <Stat label="CPU latency" value={comparison?.CPU_latency_ms ?? "…"} unit="ms" hint="model forward pass, one image" />
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <span className="eyebrow">How it works</span>
            <h2 style={{ marginTop: 8 }}>Three steps from photo to prediction</h2>
          </div>
        </div>
        <div className="steps">
          <div className="card step">
            <span className="n">01</span>
            <h3 className="row"><Crop size={18} color="var(--teal)" /> Crop one egg</h3>
            <p>Upload a microscope photo and drag the square around a single egg. The crop is resized to 224 × 224 px.</p>
          </div>
          <div className="card step">
            <span className="n">02</span>
            <h3 className="row"><ShieldCheck size={18} color="var(--teal)" /> Egg check</h3>
            <p>
              A second model confirms the image shows a parasite egg. Photos of people, objects or empty background are refused
              {gate.data?.enabled ? ` (${pct(gate.data.test.non_egg_rejection_rate, 0)} of non-egg test images rejected).` : "."}
            </p>
          </div>
          <div className="card step">
            <span className="n">03</span>
            <h3 className="row"><ScanSearch size={18} color="var(--teal)" /> Review the ranking</h3>
            <p>See the top three classes with scores, reference images of the predicted class and its known look-alikes.</p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <div>
            <span className="eyebrow">Per class</span>
            <h2 style={{ marginTop: 8 }}>How well each class is recognised</h2>
            <p className="muted small" style={{ marginTop: 6 }}>F1-score on the held-out test set (higher is better).</p>
          </div>
          <Link to="/classes" className="btn sm">All classes <ArrowRight size={15} /></Link>
        </div>
        <div className="card">
          {evaluation.isLoading && <Skeleton height={300} />}
          {test && (
            <div className="grid" style={{ gap: 12 }}>
              {perClass.map(({ id, name, f1 }) => (
                <Link key={id} to="/classes" style={{ color: "inherit", textDecoration: "none" }}>
                  <div className="spread" style={{ gap: 6 }}>
                    <span className="row" style={{ gap: 8 }}>
                      <span className="swatch" style={{ background: info(id).colour }} />
                      <strong style={{ fontSize: 14.5 }}><Species name={name} /></strong>
                      <span className="muted small hide-sm">{info(id).common}</span>
                    </span>
                    <span className="mono small">{f1.toFixed(3)}</span>
                  </div>
                  <div className="meter" style={{ marginTop: 6 }}>
                    <span style={{ width: `${f1 * 100}%`, background: info(id).colour }} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="section">
        <Callout tone="warn" title="Know the limits before you rely on a result">
          <ul>
            <li>The input must be a crop around <strong>one</strong> egg; whole slides and overlapping eggs are not supported.</li>
            <li>
              Accuracy drops on images from microscopes and cameras the model has not seen: validation accuracy{" "}
              {pct(evaluation.data?.validation?.accuracy)} vs {pct(test?.accuracy)} on the separately collected test set.
            </li>
            <li>Scores are model scores, not calibrated probabilities. Always confirm with a trained microscopist.</li>
          </ul>
          <Link to="/about" className="small" style={{ display: "inline-flex", gap: 6, marginTop: 8, alignItems: "center" }}>
            <Sparkles size={14} /> Read the full limitations
          </Link>
        </Callout>
      </section>
    </div>
  );
}
