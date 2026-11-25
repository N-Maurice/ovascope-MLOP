import { useState } from "react";
import { info } from "../lib/classes";
import { Species } from "./ui";

/** Interactive 11×11 confusion matrix. Rows = true class, columns = predicted class. */
export function ConfusionMatrix({ matrix, names, normalise }: { matrix: number[][]; names: string[]; normalise: boolean }) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));
  const offMax = Math.max(1, ...matrix.flatMap((row, i) => row.filter((_, j) => j !== i)));

  const cellStyle = (i: number, j: number) => {
    const n = matrix[i][j];
    const share = rowTotals[i] ? n / rowTotals[i] : 0;
    if (i === j) {
      const a = 0.25 + 0.75 * share;
      return { background: `rgba(15, 118, 110, ${a})`, color: a > 0.55 ? "#fff" : "var(--ink)" };
    }
    if (n === 0) return { background: "var(--surface-2)", color: "var(--muted)" };
    const a = 0.12 + 0.8 * (n / offMax);
    return { background: `rgba(180, 83, 9, ${a})`, color: a > 0.55 ? "#fff" : "var(--ink)" };
  };

  const h = hover;
  return (
    <div>
      <div className="cm-tip small" aria-live="polite">
        {h ? (
          <p>
            <strong>{matrix[h.i][h.j]}</strong> of {rowTotals[h.i]} <Species name={names[h.i]} /> crops ({((matrix[h.i][h.j] / rowTotals[h.i]) * 100).toFixed(1)}%)
            {h.i === h.j ? " were classified correctly." : <> were predicted as <Species name={names[h.j]} />.</>}
          </p>
        ) : (
          <p className="muted">Hover or tap a cell. Rows are the true class, columns the predicted class; the diagonal is correct.</p>
        )}
      </div>
      <div className="cm-wrap">
        <table className="cm" onMouseLeave={() => setHover(null)}>
          <thead>
            <tr>
              <th />
              {names.map((n, j) => (
                <th key={j} className="col-h" title={n}>
                  <div style={{ fontWeight: hover?.j === j ? 700 : 400 }}>{info(j).short}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <th className="row-h" title={names[i]} style={{ fontWeight: hover?.i === i ? 700 : 400 }}>{info(i).short}</th>
                {row.map((n, j) => (
                  <td
                    key={j}
                    className="cell"
                    style={cellStyle(i, j)}
                    onMouseEnter={() => setHover({ i, j })}
                    onClick={() => setHover({ i, j })}
                  >
                    {normalise ? (rowTotals[i] ? Math.round((n / rowTotals[i]) * 100) : 0) : n}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend-row" style={{ marginTop: 10, justifyContent: "center" }}>
        <span><span className="swatch" style={{ background: "rgba(15,118,110,0.9)" }} /> correct</span>
        <span><span className="swatch" style={{ background: "rgba(180,83,9,0.8)" }} /> confusion (darker = more)</span>
        <span>{normalise ? "values: % of the true class" : "values: crop counts"}</span>
      </div>
    </div>
  );
}
