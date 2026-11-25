import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import {
  Activity, BarChart3, Database, Home, Info, LayoutGrid, LogIn, Menu, Microscope, Moon, ShieldCheck, Sun, UserCircle2, X,
} from "lucide-react";
import { useHealth } from "../api/hooks";
import { useAuth } from "../auth";

const LINKS = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/classify", label: "Classify", icon: Microscope },
  { to: "/classes", label: "Classes", icon: LayoutGrid },
  { to: "/dataset", label: "Dataset", icon: Database },
  { to: "/performance", label: "Performance", icon: BarChart3 },
  { to: "/status", label: "Status", icon: Activity },
  { to: "/about", label: "About", icon: Info },
];
const TABS = ["/", "/classify", "/classes", "/performance", "/status"];

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="brand" aria-label="OvaScope home">
      <img src="/ovascope-mark.svg" alt="" />
      {!compact && (
        <div>
          <div className="brand-name">
            Ova<span>Scope</span>
          </div>
          <div className="brand-sub">Parasite egg classifier</div>
        </div>
      )}
    </Link>
  );
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const saved = localStorage.getItem("ovascope.theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch { /* storage unavailable */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("ovascope.theme", theme); } catch { /* ignore */ }
  }, [theme]);
  return [theme, () => setTheme(theme === "dark" ? "light" : "dark")] as const;
}

function ModelPill() {
  const health = useHealth();
  if (health.isError) return <span className="badge bad"><span className="dot bad" />API offline</span>;
  if (!health.data) return <span className="badge mono">connecting…</span>;
  return (
    <Link to="/status" className="badge mono" title="Active model (live)" style={{ textDecoration: "none" }}>
      <span className={`dot ${health.data.model_loaded ? "ok" : "bad"}`} />
      {health.data.model_version ?? "no model"}
    </Link>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="cols">
          <div>
            <div className="row" style={{ marginBottom: 10 }}>
              <img src="/ovascope-mark.svg" alt="" width={30} height={30} />
              <strong style={{ color: "#fff", fontSize: 18 }}>OvaScope</strong>
            </div>
            <p style={{ fontSize: 14 }}>
              Classifies microscope crops of single parasite eggs into 11 classes, and refuses images that are not parasite eggs.
            </p>
            <span className="badge mono" style={{ marginTop: 12, background: "#17232a", color: "#f59e0b", borderColor: "#2e434c" }}>
              RESEARCH PROTOTYPE · NOT A DIAGNOSTIC DEVICE
            </span>
          </div>
          <div>
            <h4>Use</h4>
            <ul>
              <li><Link to="/classify">Classify an image</Link></li>
              <li><Link to="/classes">Egg classes</Link></li>
              <li><Link to="/performance">Model performance</Link></li>
            </ul>
          </div>
          <div>
            <h4>Project</h4>
            <ul>
              <li><Link to="/dataset">Dataset</Link></li>
              <li><Link to="/status">Service status</Link></li>
              <li><Link to="/about">About & limitations</Link></li>
            </ul>
          </div>
          <div>
            <h4>Resources</h4>
            <ul>
              <li><a href="/api/docs" target="_blank" rel="noreferrer">API documentation</a></li>
              <li><a href="https://zenodo.org/records/15866298" target="_blank" rel="noreferrer">Chula-ParasiteEgg-11 data</a></li>
              <li><Link to="/admin">Admin console</Link></li>
            </ul>
          </div>
        </div>
        <div className="legal">
          <span>© {new Date().getFullYear()} OvaScope · student MLOps project</span>
          <span className="mono">TensorFlow · Keras · FastAPI · React</span>
        </div>
      </div>
    </footer>
  );
}

export function Layout({ children }: { children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [theme, toggleTheme] = useTheme();
  const { user, viaKey, isAdmin } = useAuth();
  const signedIn = Boolean(user) || viaKey;
  const accountLabel = user ? user.username : "Admin";
  const health = useHealth();
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
    window.scrollTo({ top: 0 });
  }, [location.pathname]);

  return (
    <>
      <a href="#main" className="sr-only">Skip to content</a>
      <header className="topbar">
        <div className="container">
          <Logo />
          <nav className="nav-links" aria-label="Main">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={l.to === "/about" ? "hide-mid" : undefined}>
                {l.label}
              </NavLink>
            ))}
          </nav>
          <div className="nav-right">
            <span className="hide-sm"><ModelPill /></span>
            <button className="icon-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            {signedIn ? (
              <>
                {isAdmin && <Link to="/admin" className="btn sm hide-sm"><ShieldCheck size={16} /> Admin</Link>}
                <Link to="/account" className="btn sm ghost hide-sm" title="My account"><UserCircle2 size={17} /> {accountLabel}</Link>
              </>
            ) : (
              <Link to="/sign-in" className="btn sm hide-sm"><LogIn size={16} /> Sign in</Link>
            )}
            <button className="icon-btn menu-btn" onClick={() => setOpen(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
          </div>
        </div>
      </header>
      {health.isError && <div className="offline-banner">The OvaScope API is not reachable. Live data and classification are unavailable.</div>}

      {open && (
        <>
          <div className="drawer-backdrop" onClick={() => setOpen(false)} />
          <nav className="drawer" aria-label="Menu">
            <div className="spread" style={{ marginBottom: 12 }}>
              <Logo />
              <button className="icon-btn" onClick={() => setOpen(false)} aria-label="Close menu"><X size={18} /></button>
            </div>
            <div style={{ margin: "0 0 10px" }}><ModelPill /></div>
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end}><l.icon size={18} />{l.label}</NavLink>
            ))}
            {signedIn ? (
              <>
                {isAdmin && <NavLink to="/admin"><ShieldCheck size={18} />Admin console</NavLink>}
                <NavLink to="/account"><UserCircle2 size={18} />{accountLabel}</NavLink>
              </>
            ) : (
              <NavLink to="/sign-in"><LogIn size={18} />Sign in / create account</NavLink>
            )}
          </nav>
        </>
      )}

      <main id="main">{children ?? <Outlet />}</main>
      <Footer />

      <nav className="tabbar" aria-label="Quick navigation">
        {LINKS.filter((l) => TABS.includes(l.to)).map((l) => (
          <NavLink key={l.to} to={l.to} end={l.end}>
            <l.icon size={21} />
            {l.label === "Performance" ? "Metrics" : l.label}
          </NavLink>
        ))}
      </nav>
    </>
  );
}
