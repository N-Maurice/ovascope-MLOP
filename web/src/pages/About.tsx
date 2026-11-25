import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { useEvaluation, useHealth } from "../api/hooks";
import { pct } from "../lib/format";
import { Callout, PageHead } from "../components/ui";

export default function About() {
  const evaluation = useEvaluation();
  const health = useHealth();
  const e = evaluation.data;
  return (
    <div className="container" style={{ maxWidth: 900 }}>
      <PageHead eyebrow="About" title="About OvaScope">
        An end-to-end machine-learning project: data preparation, model training and evaluation, an API with an input check,
        a database of uploaded images, retraining with automatic promotion, monitoring and this interface.
      </PageHead>

      <div className="stack">
        <div className="card pad-lg stack">
          <h3>What it does</h3>
          <p>
            OvaScope classifies a microscope image cropped around <strong>one parasite egg</strong> into one of 11 classes from the
            Chula-ParasiteEgg-11 dataset. Before classifying, a separate egg check refuses images that do not show a parasite egg,
            such as photos of people or objects, or empty background.
          </p>
          <h3>How it was built</h3>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            <li>13,259 egg crops cut from the official annotated photos, with an audit for leakage between splits.</li>
            <li>A baseline CNN and a fine-tuned MobileNetV2 (ImageNet) compared on validation data; the better one is deployed.</li>
            <li>Seven metrics tracked every epoch: accuracy, precision, recall, F1, ROC-AUC, PR-AUC and log loss.</li>
            <li>A one-time evaluation on the protected official test set.</li>
            <li>Retraining from uploaded, labelled images, mixed with old data; a new model is promoted only if it is not worse.</li>
          </ul>
        </div>

        <Callout tone="warn" title="Limitations">
          <ul>
            <li><strong>Not a diagnostic device.</strong> Results must be confirmed by a trained microscopist.</li>
            <li>Only single-egg crops are supported; there is no egg detection on whole slides.</li>
            <li>
              Accuracy depends on the capture setup: {pct(e?.validation?.accuracy)} on validation photos but {pct(e?.test.accuracy)} on the
              separately collected test photos, mainly because of different cameras.
            </li>
            <li>Look-alike pairs are the main errors (for example the two <em>Hymenolepis</em> species, or <em>Capillaria</em> and <em>Opisthorchis</em>).</li>
            <li>Scores are model scores, not calibrated probabilities, and can be high even when the answer is wrong.</li>
            <li>The egg check lets some empty-background images through and occasionally refuses a real egg; retry with a tighter, sharper crop.</li>
            <li>Only these 11 classes are known: eggs of any other parasite will be assigned to one of them.</li>
          </ul>
        </Callout>

        <div className="card pad-lg stack">
          <h3>Using the API directly</h3>
          <p className="small">Every number on this site comes from the OvaScope REST API. The interactive documentation lists all endpoints.</p>
          <pre className="mono small" style={{ background: "var(--surface-2)", padding: 14, borderRadius: 10, overflowX: "auto", margin: 0 }}>
{`curl -F "file=@egg.jpg" ${window.location.origin}/api/predict`}
          </pre>
          <div className="row">
            <a className="btn sm" href="/api/docs" target="_blank" rel="noreferrer">API docs <ExternalLink size={14} /></a>
            <a className="btn sm" href="https://zenodo.org/records/15866298" target="_blank" rel="noreferrer">Dataset <ExternalLink size={14} /></a>
            <Link className="btn sm" to="/status">Service status</Link>
          </div>
          <p className="tiny muted">Active model: <span className="mono">{health.data?.model_version ?? "—"}</span></p>
        </div>

        <div className="card pad-lg">
          <h3 style={{ marginBottom: 8 }}>Data source</h3>
          <p className="small">
            Chula-ParasiteEgg-11 (Zenodo record 15866298), released for the ICIP 2022 Challenge on Parasitic Egg Detection and
            Classification in Microscopic Images. Please cite the dataset authors when using it.
          </p>
        </div>
      </div>
    </div>
  );
}
