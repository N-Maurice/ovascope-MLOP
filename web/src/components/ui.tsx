import { useEffect, type ReactNode } from "react";
import { AlertTriangle, Info, X, XCircle } from "lucide-react";
import { ApiError } from "../api/client";

export function Stat({ label, value, unit, hint, meter }: { label: string; value: ReactNode; unit?: string; hint?: ReactNode; meter?: number }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {hint && <div className="hint">{hint}</div>}
      {meter !== undefined && <Meter value={meter} />}
    </div>
  );
}

export function Meter({ value, colour }: { value: number; colour?: string }) {
  return (
    <div className="meter" role="presentation">
      <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: colour }} />
    </div>
  );
}

export function PageHead({ eyebrow, title, children, actions }: { eyebrow: string; title: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <span className="eyebrow">{eyebrow}</span>
      <div className="spread" style={{ alignItems: "end" }}>
        <h1 style={{ fontSize: "clamp(26px, 4vw, 38px)" }}>{title}</h1>
        {actions}
      </div>
      {children && <p>{children}</p>}
    </header>
  );
}

export function Segmented<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (value: T) => void; options: { value: T; label: ReactNode }[]; label: string;
}) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} className={value === o.value ? "on" : ""} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Skeleton({ height = 120, style }: { height?: number; style?: React.CSSProperties }) {
  return <div className="skeleton" style={{ height, ...style }} aria-busy="true" />;
}

export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn" | "bad"; title?: ReactNode; children?: ReactNode }) {
  const Icon = tone === "bad" ? XCircle : tone === "warn" ? AlertTriangle : Info;
  return (
    <div className={`callout ${tone}`} role={tone === "info" ? "note" : "alert"}>
      <Icon size={20} />
      <div>
        {title && <h4>{title}</h4>}
        {children}
      </div>
    </div>
  );
}

export function ErrorState({ error, what }: { error: unknown; what: string }) {
  const status = error instanceof ApiError ? error.status : undefined;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <Callout tone={status === 404 ? "warn" : "bad"} title={status === 404 ? `No ${what} available yet` : `Could not load ${what}`}>
      <p>{message}</p>
    </Callout>
  );
}

export function Modal({ onClose, children, label }: { onClose: () => void; children: ReactNode; label: string }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [onClose]);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={label} onClick={(e) => e.stopPropagation()}>
        <button className="icon-btn close" onClick={onClose} aria-label="Close">
          <X size={18} />
        </button>
        {children}
      </div>
    </div>
  );
}

export function BarCell({ value, colour = "var(--teal)", digits = 3 }: { value: number | null | undefined; colour?: string; digits?: number }) {
  return (
    <div className="bar-cell">
      <div className="track">
        <span style={{ width: `${Math.max(0, Math.min(1, value ?? 0)) * 100}%`, background: colour }} />
      </div>
      <span className="mono small">{value === null || value === undefined ? "—" : value.toFixed(digits)}</span>
    </div>
  );
}

export function Species({ name }: { name: string }) {
  // Genus/species names are italic; generic words such as "egg" or "spp." stay upright.
  const parts = name.split(" ");
  return (
    <span>
      {parts.map((part, i) => (
        <span key={i} style={{ fontStyle: /^(egg|spp\.?)$/i.test(part) ? "normal" : "italic" }}>
          {part}
          {i < parts.length - 1 ? " " : ""}
        </span>
      ))}
    </span>
  );
}
