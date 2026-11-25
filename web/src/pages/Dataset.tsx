import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, ExternalLink, XCircle } from "lucide-react";
import { exampleUrl } from "../api/client";
import { useDataset } from "../api/hooks";
import { info } from "../lib/classes";
import { int, pct } from "../lib/format";
import { ErrorState, PageHead, Skeleton, Species, Stat } from "../components/ui";

const SPLIT_COLOURS = { train: "#0f766e", val: "#2dd4bf", test: "#b45309" };

function Check({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <li>
      {ok ? <CheckCircle2 size={18} color="var(--green)" /> : <XCircle size={18} color="var(--red)" />}
      <span>{children}</span>
    </li>
  );
}

export default function Dataset() {
  const dataset = useDataset();
  const d = dataset.data;

  const perClass = d?.classes.map((c) => ({
    name: info(c.class_id).short,
    full: c.display_name,
    train: d.crops_per_split_class.train?.[c.class_id] ?? 0,
    val: d.crops_per_split_class.val?.[c.class_id] ?? 0,
    test: d.crops_per_split_class.test?.[c.class_id] ?? 0,
  }));
  const sizes = d ? Object.entries(d.source_image_sizes).map(([size, n]) => ({ size, n })).sort((a, b) => b.n - a.n) : [];
  const nearDup = d ? Object.values(d.near_duplicate_pairs_across_splits).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="container">
      <PageHead
        eyebrow="Dataset"
        title="Chula-ParasiteEgg-11"
        actions={<a className="btn sm" href="https://zenodo.org/records/15866298" target="_blank" rel="noreferrer">Source on Zenodo <ExternalLink size={14} /></a>}
      >
        11,000 annotated training photos and 2,200 test photos of stool samples, released for the ICIP 2022 parasite-egg
        challenge. Every annotated egg was cut out as a square crop with 10% margin; these crops are what the model sees.
      </PageHead>

      {dataset.isLoading && <Skeleton height={400} />}
      {dataset.isError && <ErrorState error={dataset.error} what="dataset summary" />}
      {d && perClass && (
        <>
          <div className="grid g4 tight">
            <Stat label="Egg crops" value={int(d.crops_total)} hint={`from ${int(d.source_photos.train_and_val + d.source_photos.test)} photos`} />
            <Stat label="Training" value={int(d.crops_per_split.train)} hint={pct(d.crops_per_split.train / d.crops_total, 0) + " of crops"} />
            <Stat label="Validation" value={int(d.crops_per_split.val)} hint="model selection only" />
            <Stat label="Test" value={int(d.crops_per_split.test)} hint="official test photos, protected" />
          </div>

          <section className="section card">
            <div className="card-title">
              <div>
                <span className="eyebrow">Class balance</span>
                <h3 style={{ marginTop: 6 }}>Crops per class and split</h3>
              </div>
            </div>
            <div className="chart-box lg">
              <ResponsiveContainer>
                <BarChart data={perClass} margin={{ top: 8, right: 8, left: -12, bottom: 40 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} height={60} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} labelFormatter={(_, p) => p?.[0]?.payload.full ?? ""} />
                  <Legend verticalAlign="top" height={30} />
                  <Bar dataKey="train" stackId="a" fill={SPLIT_COLOURS.train} name="Training" />
                  <Bar dataKey="val" stackId="a" fill={SPLIT_COLOURS.val} name="Validation" />
                  <Bar dataKey="test" stackId="a" fill={SPLIT_COLOURS.test} name="Test" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="small muted">
              The classes are balanced (850–860 training crops each), so accuracy is meaningful and no class weighting is needed.
            </p>
          </section>

          <div className="grid g2 section">
            <div className="card">
              <span className="eyebrow">Integrity checks</span>
              <h3 style={{ margin: "6px 0 14px" }}>Data audit</h3>
              <ul className="checklist">
                <Check ok={d.source_images_in_multiple_splits.length === 0}>No source photo appears in more than one split.</Check>
                <Check ok={d.exact_duplicates_across_splits.length === 0 && d.exact_duplicate_crops_across_splits === 0}>
                  No exact duplicate photos or crops across splits.
                </Check>
                <Check ok={nearDup === 0}>
                  No near-duplicates across splits (perceptual hash, ≤ {d.near_duplicate_threshold_bits} bits). Closest test photo to any
                  training photo: {d.test_to_trainval_closest_distance.min} bits.
                </Check>
                <Check ok={d.categories_identical_train_test && d.class_map_matches_annotations}>
                  Identical class list in the training and test annotations.
                </Check>
                <Check ok={d.non_square_crops === 0 && d.empty_or_flat_crops === 0}>All crops are square and non-empty.</Check>
              </ul>
            </div>
            <div className="card">
              <span className="eyebrow">Image properties</span>
              <h3 style={{ margin: "6px 0 14px" }}>Crop sizes</h3>
              <dl className="kv">
                <dt>Smallest crop</dt><dd className="mono">{d.crop_side_px.min} px</dd>
                <dt>Median crop</dt><dd className="mono">{d.crop_side_px.median} px</dd>
                <dt>Largest crop</dt><dd className="mono">{d.crop_side_px.max} px</dd>
                <dt>Upsampled to 224 px</dt><dd className="mono">{pct(d.crop_side_px.share_below_224)}</dd>
              </dl>
              <p className="small muted" style={{ marginTop: 14 }}>
                Egg size differs strongly between species, but every crop is resized to 224 × 224 px, so the model judges
                shape and texture, not absolute size.
              </p>
            </div>
          </div>

          <section className="section card">
            <span className="eyebrow">Capture devices</span>
            <h3 style={{ margin: "6px 0 4px" }}>Source photo resolutions</h3>
            <p className="small muted" style={{ marginBottom: 10 }}>
              Twelve camera formats appear in the data. The test photos use a different mix of cameras, which is the main reason
              test accuracy is lower than validation accuracy (see Performance).
            </p>
            <div className="chart-box">
              <ResponsiveContainer>
                <BarChart data={sizes} layout="vertical" margin={{ left: 20, right: 16 }}>
                  <CartesianGrid horizontal={false} />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="size" width={80} tick={{ fontSize: 11, fontFamily: "var(--mono)" }} />
                  <Tooltip cursor={{ fill: "var(--surface-2)" }} formatter={(v: number) => [int(v), "photos"]} />
                  <Bar dataKey="n" fill="#0f766e" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="section">
            <div className="section-head"><span className="eyebrow">One validation crop per class</span></div>
            <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))" }}>
              {d.classes.map((c) => (
                <figure key={c.class_id} style={{ margin: 0 }}>
                  <img src={exampleUrl(c.class_id, 3)} alt={c.display_name} loading="lazy" />
                  <figcaption className="tiny" style={{ marginTop: 6 }}><Species name={c.display_name} /></figcaption>
                </figure>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
