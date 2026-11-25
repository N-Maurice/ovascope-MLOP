import { useState } from "react";
import { Link, Navigate } from "react-router-dom";
import { KeyRound, LogOut, ShieldCheck, ShieldOff } from "lucide-react";
import { send } from "../api/client";
import { useAuth } from "../auth";
import { dateTime } from "../lib/format";
import { Callout, PageHead } from "../components/ui";
import { useToast } from "../components/Toast";

export default function Account() {
  const { user, viaKey, isAdmin, signOut } = useAuth();
  const toast = useToast();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!user && !viaKey) return <Navigate to="/sign-in" replace />;

  const change = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await send("POST", "/auth/password", { current_password: current, new_password: next });
      toast("Password changed. Please sign in again.");
      signOut();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: 760 }}>
      <PageHead eyebrow="Account" title={user ? user.username : "Operator key"}
        actions={<button className="btn sm" onClick={signOut}><LogOut size={15} /> Sign out</button>} />
      <div className="stack">
        {user && (
          <div className="card pad-lg">
            <dl className="kv">
              <dt>Username</dt><dd>{user.username}</dd>
              <dt>Email</dt><dd>{user.email}</dd>
              <dt>Role</dt><dd><span className={`badge ${user.role === "admin" ? "ok" : ""}`}>{user.role === "admin" ? "Administrator" : "User"}</span></dd>
              <dt>Member since</dt><dd>{dateTime(user.created_at)}</dd>
              <dt>Last sign-in</dt><dd>{dateTime(user.last_login)}</dd>
            </dl>
          </div>
        )}
        {isAdmin ? (
          <Callout tone="info" title="You have administrator rights">
            <p>Upload training images, start retraining, switch model versions and manage users.</p>
            <Link className="btn sm teal" to="/admin" style={{ marginTop: 10 }}><ShieldCheck size={15} /> Open the admin console</Link>
          </Callout>
        ) : (
          <Callout tone="warn" title="No administrator rights yet">
            <p>Your account can use every public page. An administrator can give it admin rights in the admin console (Users tab).</p>
          </Callout>
        )}
        {user && (
          <form className="card pad-lg stack" onSubmit={change}>
            <h3 className="row"><KeyRound size={18} /> Change password</h3>
            <div className="grid g2">
              <div className="field">
                <label htmlFor="current">Current password</label>
                <input id="current" className="input" type="password" value={current} onChange={(e) => setCurrent(e.target.value)} autoComplete="current-password" required />
              </div>
              <div className="field">
                <label htmlFor="next">New password</label>
                <input id="next" className="input" type="password" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password" required />
              </div>
            </div>
            {error && <Callout tone="bad"><p>{error}</p></Callout>}
            <div><button className="btn primary" disabled={busy || !current || !next}>{busy && <span className="spinner" />} Change password</button></div>
            <p className="tiny muted">Changing the password signs you out on every device.</p>
          </form>
        )}
      </div>
    </div>
  );
}

export function NoAdminAccess() {
  const { user } = useAuth();
  return (
    <div className="container" style={{ maxWidth: 560, paddingTop: 48 }}>
      <div className="card pad-lg stack" style={{ textAlign: "center", justifyItems: "center" }}>
        <ShieldOff size={40} color="var(--amber)" />
        <h2>Administrator rights needed</h2>
        <p className="muted">
          You are signed in as <strong>{user?.username}</strong>, but this account is not an administrator yet.
          Ask an administrator to grant admin rights in the admin console (Users tab).
        </p>
        <div className="row" style={{ justifyContent: "center" }}>
          <Link className="btn" to="/account">My account</Link>
          <Link className="btn primary" to="/">Home</Link>
        </div>
      </div>
    </div>
  );
}
