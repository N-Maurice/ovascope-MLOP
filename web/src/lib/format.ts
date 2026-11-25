export const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);

export const ms = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;

export const int = (value: number | null | undefined) => (value === null || value === undefined ? "—" : value.toLocaleString("en-US"));

export function duration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.floor(seconds);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

export function dateTime(iso: string | null | undefined) {
  if (!iso) return "—";
  const date = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function bytes(value: number | null | undefined) {
  if (!value && value !== 0) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}
