// Accounts: sign in with username or email + password, or create an account. The API returns a
// session token; admin pages need an account with the "admin" role (or the operator admin key).
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getAdminKey, getToken, request, send, setAdminKey, setToken, setUnauthorisedHandler } from "./api/client";

export type User = { id: number; username: string; email: string; role: "admin" | "user"; active: boolean; created_at: string; last_login: string | null };
type Session = { token: string; expires_at: string; user: User };

type Auth = {
  user: User | null;
  viaKey: boolean;          // signed in with the operator admin key instead of an account
  isAdmin: boolean;
  checking: boolean;        // validating a stored session on page load
  signIn: (login: string, password: string, remember: boolean) => Promise<User>;
  register: (username: string, email: string, password: string) => Promise<User>;
  signInWithKey: (key: string) => Promise<void>;
  signOut: () => void;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<Auth | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const client = useQueryClient();
  const [user, setUser] = useState<User | null>(null);
  const [viaKey, setViaKey] = useState(() => getAdminKey() !== null);
  const [checking, setChecking] = useState(() => getToken() !== null);

  const clear = useCallback(() => {
    setToken(null);
    setAdminKey(null);
    setUser(null);
    setViaKey(false);
    client.removeQueries({ queryKey: ["uploads"] });
    client.removeQueries({ queryKey: ["users"] });
  }, [client]);

  const refresh = useCallback(async () => {
    if (!getToken()) return;
    try {
      setUser((await request<User>("/auth/me")).data);
    } catch {
      clear();
    } finally {
      setChecking(false);
    }
  }, [clear]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { setUnauthorisedHandler(clear); return () => setUnauthorisedHandler(null); }, [clear]);

  const start = useCallback((session: Session, remember: boolean) => {
    setAdminKey(null);
    setViaKey(false);
    setToken(session.token, remember);
    setUser(session.user);
    return session.user;
  }, []);

  const signIn = useCallback(async (login: string, password: string, remember: boolean) =>
    start(await send<Session>("POST", "/auth/login", { login, password, remember }), remember), [start]);

  const register = useCallback(async (username: string, email: string, password: string) =>
    start(await send<Session>("POST", "/auth/register", { username, email, password }), true), [start]);

  const signInWithKey = useCallback(async (key: string) => {
    await request("/auth/verify", { method: "POST", headers: { "X-API-Key": key } });
    setToken(null);
    setUser(null);
    setAdminKey(key);
    setViaKey(true);
  }, []);

  const signOut = useCallback(() => {
    if (getToken()) send("POST", "/auth/logout").catch(() => {});   // end the session on the server too
    clear();
  }, [clear]);

  const value = useMemo<Auth>(() => ({
    user, viaKey, checking, isAdmin: viaKey || user?.role === "admin",
    signIn, register, signInWithKey, signOut, refresh,
  }), [user, viaKey, checking, signIn, register, signInWithKey, signOut, refresh]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const auth = useContext(AuthContext);
  if (!auth) throw new Error("useAuth outside AuthProvider");
  return auth;
}
