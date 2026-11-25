// Thin wrapper around the OvaScope FastAPI service. Every request goes to `${API_BASE}/<path>`;
// in development and in the Docker web container `/api` is proxied to the API (same origin).

export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

// Credentials: a session token from /auth/login (normal accounts) or, for operators, the raw admin key.
// "Remember me" keeps the token in localStorage; otherwise it lives in sessionStorage (this tab only).
const TOKEN = "ovascope.token";
const KEY = "ovascope.adminKey";

function read(name: string): string | null {
  try {
    return sessionStorage.getItem(name) ?? localStorage.getItem(name);
  } catch {
    return null;
  }
}

function write(name: string, value: string | null, persistent = false) {
  try {
    sessionStorage.removeItem(name);
    localStorage.removeItem(name);
    if (value !== null) (persistent ? localStorage : sessionStorage).setItem(name, value);
  } catch {
    /* storage unavailable (private mode): credentials then only last until reload */
  }
}

export const getToken = () => read(TOKEN);
export const setToken = (token: string | null, remember = true) => write(TOKEN, token, remember);
export const getAdminKey = () => read(KEY);
export const setAdminKey = (key: string | null) => write(KEY, key);

/** Headers that identify the signed-in person (used for every request and for image fetches). */
export function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = {};
  const token = getToken();
  const key = getAdminKey();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (key) headers["X-API-Key"] = key;
  return headers;
}

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : (detail as { message?: string })?.message ?? `Request failed (${status})`);
    this.status = status;
    this.detail = detail;
  }
}

type Options = { method?: string; body?: BodyInit; json?: unknown; headers?: Record<string, string>; signal?: AbortSignal };

/** Called when a signed-in request is refused with 401 (expired or revoked session). */
let onUnauthorised: (() => void) | null = null;
export const setUnauthorisedHandler = (handler: (() => void) | null) => { onUnauthorised = handler; };

export async function request<T>(path: string, options: Options = {}): Promise<{ data: T; headers: Headers }> {
  const headers: Record<string, string> = { ...authHeaders(), ...options.headers };
  let body = options.body;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { method: options.method ?? "GET", body, headers, signal: options.signal });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "The OvaScope API cannot be reached. Is the server running?");
  }
  const type = response.headers.get("content-type") ?? "";
  const payload = response.status === 204 ? null : type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    let detail = typeof payload === "object" && payload && "detail" in payload ? (payload as { detail: unknown }).detail : payload;
    if (Array.isArray(detail)) detail = detail.map((d: { msg?: string }) => d.msg).join("; ");   // FastAPI validation errors
    if (response.status === 401 && (headers.Authorization || headers["X-API-Key"]) && !path.startsWith("/auth/")) onUnauthorised?.();
    throw new ApiError(response.status, detail || response.statusText);
  }
  return { data: payload as T, headers: response.headers };
}

export const get = async <T>(path: string) => (await request<T>(path)).data;

export const post = async <T>(path: string, body?: BodyInit) => (await request<T>(path, { method: "POST", body })).data;

export const send = async <T>(method: string, path: string, json?: unknown) => (await request<T>(path, { method, json })).data;

export const exampleUrl = (classId: number, index = 0) => `${API_BASE}/classes/${classId}/examples/${index}`;
