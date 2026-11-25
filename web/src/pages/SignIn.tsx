import { useState } from "react";
import { Link, Navigate, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { AtSign, Eye, EyeOff, KeyRound, Lock, LogIn, User as UserIcon, UserPlus } from "lucide-react";
import { ApiError } from "../api/client";
import { useAuthStatus } from "../api/hooks";
import { useAuth } from "../auth";
import { Callout, Segmented } from "../components/ui";

type Mode = "signin" | "register" | "key";

function PasswordInput({ id, value, onChange, autoComplete }: { id: string; value: string; onChange: (v: string) => void; autoComplete: string }) {
  const [show, setShow] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <input id={id} className="input" type={show ? "text" : "password"} value={value} onChange={(e) => onChange(e.target.value)}
        autoComplete={autoComplete} required style={{ paddingLeft: 38, paddingRight: 42 }} />
      <Lock size={16} className="muted" style={{ position: "absolute", left: 12, top: 13 }} />
      <button type="button" className="btn ghost sm" onClick={() => setShow(!show)} aria-label={show ? "Hide password" : "Show password"}
        style={{ position: "absolute", right: 4, top: 4 }}>
        {show ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  );
}

function IconInput({ id, icon, ...props }: React.InputHTMLAttributes<HTMLInputElement> & { icon: React.ReactNode }) {
  return (
    <div style={{ position: "relative" }}>
      <input id={id} className="input" {...props} style={{ paddingLeft: 38 }} />
      <span className="muted" style={{ position: "absolute", left: 12, top: 11 }}>{icon}</span>
    </div>
  );
}

export default function SignIn() {
  const { user, viaKey, isAdmin, signIn, register, signInWithKey } = useAuth();
  const status = useAuthStatus();
  const client = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const [mode, setMode] = useState<Mode>(params.get("mode") === "register" ? "register" : "signin");
  const [login, setLogin] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [remember, setRemember] = useState(true);
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const target = (location.state as { from?: string } | null)?.from;

  if (user || viaKey) return <Navigate to={target ?? (isAdmin ? "/admin" : "/account")} replace />;

  const s = status.data;
  const minLength = s?.min_password_length ?? 8;

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      const signedIn = await action();
      client.invalidateQueries({ queryKey: ["auth-status"] });
      const admin = !signedIn || (signedIn as { role?: string }).role === "admin";   // key sign-in returns nothing
      navigate(admin ? target ?? "/admin" : "/account", { replace: true });
    } catch (e) {
      setError(e instanceof ApiError && e.status === 0 ? e.message : (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "signin") return run(() => signIn(login.trim(), password, remember));
    if (mode === "key") return run(() => signInWithKey(key.trim()));
    if (password !== confirm) return setError("The two passwords do not match.");
    return run(() => register(username.trim(), email.trim(), password));
  };

  return (
    <div className="container" style={{ maxWidth: 470, paddingTop: 40 }}>
      <div className="card pad-lg stack">
        <div style={{ textAlign: "center" }}>
          <img src="/ovascope-mark.svg" alt="" width={56} height={56} style={{ margin: "0 auto 14px" }} />
          <h2>{mode === "register" ? "Create your account" : "Welcome back"}</h2>
          <p className="muted small" style={{ marginTop: 6 }}>
            Administrators can upload training images, retrain the model and manage model versions and users.
          </p>
        </div>

        {mode !== "key" && (
          <div style={{ display: "flex", justifyContent: "center" }}>
            <Segmented label="Sign in or create account" value={mode} onChange={(m) => { setMode(m); setError(null); }}
              options={[{ value: "signin", label: <><LogIn size={15} /> Sign in</> }, { value: "register", label: <><UserPlus size={15} /> Create account</> }]} />
          </div>
        )}

        {mode === "register" && s?.first_account_becomes_admin && (
          <Callout tone="info" title="First account">
            <p>No administrator exists yet, so this account will become the <strong>administrator</strong>.</p>
          </Callout>
        )}
        {mode === "register" && s && !s.first_account_becomes_admin && (
          <Callout tone="info">
            <p>New accounts start without admin rights. An administrator can grant them in the admin console.</p>
          </Callout>
        )}
        {mode === "register" && s && !s.signup_open && (
          <Callout tone="warn"><p>Sign-up is disabled on this server. Ask an administrator for an account.</p></Callout>
        )}

        <form className="stack" onSubmit={submit} noValidate={false}>
          {mode === "signin" && (
            <>
              <div className="field">
                <label htmlFor="login">Username or email</label>
                <IconInput id="login" icon={<UserIcon size={16} />} value={login} onChange={(e) => setLogin(e.target.value)}
                  autoComplete="username" autoCapitalize="none" required autoFocus />
              </div>
              <div className="field">
                <label htmlFor="password">Password</label>
                <PasswordInput id="password" value={password} onChange={setPassword} autoComplete="current-password" />
              </div>
              <label className="row small" style={{ gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} /> Keep me signed in on this device
              </label>
            </>
          )}
          {mode === "register" && (
            <>
              <div className="field">
                <label htmlFor="username">Username</label>
                <IconInput id="username" icon={<UserIcon size={16} />} value={username} onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username" autoCapitalize="none" required minLength={3} maxLength={32} pattern="[A-Za-z0-9_.\-]+" autoFocus />
                <span className="tiny muted">3–32 characters: letters, digits, “.”, “_” or “-”.</span>
              </div>
              <div className="field">
                <label htmlFor="email">Email</label>
                <IconInput id="email" icon={<AtSign size={16} />} type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required />
              </div>
              <div className="field">
                <label htmlFor="new-password">Password</label>
                <PasswordInput id="new-password" value={password} onChange={setPassword} autoComplete="new-password" />
                <span className="tiny muted">At least {minLength} characters, mixing letters with digits or symbols.</span>
              </div>
              <div className="field">
                <label htmlFor="confirm">Repeat password</label>
                <PasswordInput id="confirm" value={confirm} onChange={setConfirm} autoComplete="new-password" />
              </div>
            </>
          )}
          {mode === "key" && (
            <div className="field">
              <label htmlFor="key">Operator admin key</label>
              <IconInput id="key" icon={<KeyRound size={16} />} className="input mono" type="password" value={key}
                onChange={(e) => setKey(e.target.value)} placeholder="OVASCOPE_ADMIN_KEY" autoFocus />
              <span className="tiny muted">The value of OVASCOPE_ADMIN_KEY on the server, for operators without an account.</span>
            </div>
          )}

          {error && <Callout tone="bad"><p>{error}</p></Callout>}
          <button className="btn primary lg block" disabled={busy || (mode === "register" && s !== undefined && !s.signup_open)}>
            {busy ? <span className="spinner" /> : mode === "register" ? <UserPlus size={18} /> : <LogIn size={18} />}
            {mode === "register" ? "Create account" : "Sign in"}
          </button>
        </form>

        <p className="tiny muted" style={{ textAlign: "center" }}>
          {mode === "key"
            ? <button className="btn ghost sm" onClick={() => setMode("signin")}>Back to account sign-in</button>
            : <button className="btn ghost sm" onClick={() => setMode("key")}><KeyRound size={13} /> Use the operator admin key instead</button>}
        </p>
        <p className="tiny muted" style={{ textAlign: "center" }}>
          Forgot your password? An administrator can reset it with <span className="mono">python -m src.users reset-password</span>. <Link to="/">Back home</Link>
        </p>
      </div>
    </div>
  );
}
