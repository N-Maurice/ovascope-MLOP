import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ShieldCheck, ShieldOff, Trash2, UserCheck, UserX } from "lucide-react";
import { send } from "../../api/client";
import { useUsers } from "../../api/hooks";
import { useAuth, type User } from "../../auth";
import { dateTime } from "../../lib/format";
import { Callout, ErrorState, Modal, Skeleton } from "../../components/ui";
import { useToast } from "../../components/Toast";

type Action = { user: User; kind: "promote" | "demote" | "disable" | "enable" | "delete" };

const TEXT: Record<Action["kind"], (u: string) => string> = {
  promote: (u) => `Give ${u} administrator rights?`,
  demote: (u) => `Remove administrator rights from ${u}?`,
  disable: (u) => `Deactivate ${u}? They will be signed out and cannot sign in until reactivated.`,
  enable: (u) => `Reactivate ${u}?`,
  delete: (u) => `Delete the account ${u} permanently? Their uploads stay in the database.`,
};

export default function UsersPanel() {
  const { user: me, refresh } = useAuth();
  const users = useUsers(true);
  const client = useQueryClient();
  const toast = useToast();
  const [action, setAction] = useState<Action | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!action) return;
    setBusy(true);
    setError(null);
    const { user, kind } = action;
    try {
      if (kind === "delete") await send("DELETE", `/users/${user.id}`);
      else await send("PATCH", `/users/${user.id}`, kind === "promote" || kind === "demote" ? { role: kind === "promote" ? "admin" : "user" } : { active: kind === "enable" });
      toast(`Updated ${user.username}`);
      client.invalidateQueries({ queryKey: ["users"] });
      client.invalidateQueries({ queryKey: ["auth-status"] });
      if (user.id === me?.id) await refresh();   // e.g. an admin removed their own rights
      setAction(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (users.isLoading) return <Skeleton height={300} />;
  if (users.isError) return <ErrorState error={users.error} what="accounts" />;
  const list = users.data ?? [];
  const admins = list.filter((u) => u.role === "admin" && u.active).length;
  return (
    <div className="stack">
      <p className="small muted" style={{ maxWidth: 680 }}>
        {list.length} account(s), {admins} active administrator(s). New sign-ups start as <strong>user</strong>; promote trusted
        people to <strong>admin</strong> to let them upload data, retrain and switch models. The last active administrator cannot
        be demoted, deactivated or deleted.
      </p>
      {error && <Callout tone="bad"><p>{error}</p></Callout>}
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>User</th><th>Role</th><th>Status</th><th>Created</th><th>Last sign-in</th><th className="right">Actions</th></tr></thead>
          <tbody>
            {list.map((u) => (
              <tr key={u.id} className={u.id === me?.id ? "hl" : ""}>
                <td>
                  <div className="small" style={{ fontWeight: 650 }}>{u.username}{u.id === me?.id && <span className="muted"> (you)</span>}</div>
                  <div className="tiny muted">{u.email}</div>
                </td>
                <td><span className={`badge ${u.role === "admin" ? "ok" : ""}`}>{u.role}</span></td>
                <td><span className={`badge ${u.active ? "teal" : "bad"}`}>{u.active ? "active" : "deactivated"}</span></td>
                <td className="small nowrap">{dateTime(u.created_at)}</td>
                <td className="small nowrap">{dateTime(u.last_login)}</td>
                <td className="right">
                  <div className="row" style={{ justifyContent: "flex-end", gap: 6, flexWrap: "nowrap" }}>
                    {u.role === "admin"
                      ? <button className="btn sm" onClick={() => setAction({ user: u, kind: "demote" })} title="Remove admin rights"><ShieldOff size={14} /> Make user</button>
                      : <button className="btn sm teal" onClick={() => setAction({ user: u, kind: "promote" })} title="Grant admin rights"><ShieldCheck size={14} /> Make admin</button>}
                    {u.active
                      ? <button className="icon-btn" style={{ width: 32, height: 32 }} onClick={() => setAction({ user: u, kind: "disable" })} aria-label={`Deactivate ${u.username}`} title="Deactivate"><UserX size={15} /></button>
                      : <button className="icon-btn" style={{ width: 32, height: 32 }} onClick={() => setAction({ user: u, kind: "enable" })} aria-label={`Reactivate ${u.username}`} title="Reactivate"><UserCheck size={15} /></button>}
                    <button className="icon-btn" style={{ width: 32, height: 32, color: "var(--red)" }} onClick={() => setAction({ user: u, kind: "delete" })} aria-label={`Delete ${u.username}`} title="Delete"><Trash2 size={15} /></button>
                  </div>
                </td>
              </tr>
            ))}
            {!list.length && <tr><td colSpan={6} className="muted">No accounts yet (you are signed in with the operator key).</td></tr>}
          </tbody>
        </table>
      </div>
      {action && (
        <Modal onClose={() => setAction(null)} label="Confirm account change">
          <div className="stack" style={{ maxWidth: 460 }}>
            <h3>{TEXT[action.kind](action.user.username)}</h3>
            {error && <Callout tone="bad"><p>{error}</p></Callout>}
            <div className="row">
              <button className={`btn ${action.kind === "delete" ? "danger" : "primary"}`} onClick={run} disabled={busy}>{busy && <span className="spinner" />} Confirm</button>
              <button className="btn" onClick={() => { setAction(null); setError(null); }}>Cancel</button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
