// React Query hooks: one cache shared by every page, refreshed in the background so the
// interface stays in sync with the API, the database and the model registry.
import { useQuery } from "@tanstack/react-query";
import { get } from "./client";
import type {
  ClassEntry, DatasetSummary, Evaluation, GateReport, Health, HistoryRow, Job, ModelVersion, RetrainStatus,
  ServiceMetrics, Uploads,
} from "./types";
import type { User } from "../auth";

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: () => get<Health>("/health"), refetchInterval: 15_000, retry: 1 });

export const useClasses = () =>
  useQuery({ queryKey: ["classes"], queryFn: () => get<ClassEntry[]>("/classes"), staleTime: Infinity });

export const useEvaluation = () =>
  useQuery({ queryKey: ["evaluation"], queryFn: () => get<Evaluation>("/results/evaluation"), staleTime: 60_000 });

export const useDataset = () =>
  useQuery({ queryKey: ["dataset"], queryFn: () => get<DatasetSummary>("/dataset/summary"), staleTime: Infinity });

export const useGate = () => useQuery({ queryKey: ["gate"], queryFn: () => get<GateReport>("/gate"), staleTime: 60_000 });

export const useVersions = () =>
  useQuery({ queryKey: ["versions"], queryFn: () => get<ModelVersion[]>("/model/versions"), refetchInterval: 30_000 });

export const useHistory = (version: string | undefined) =>
  useQuery({
    queryKey: ["history", version],
    queryFn: () => get<HistoryRow[]>(`/results/history/${version}`),
    enabled: Boolean(version),
    staleTime: Infinity,
  });

export const useServiceMetrics = (live = true) =>
  useQuery({ queryKey: ["metrics"], queryFn: () => get<ServiceMetrics>("/metrics"), refetchInterval: live ? 5_000 : false });

export const useRetrainStatus = () =>
  useQuery({ queryKey: ["retrain-status"], queryFn: () => get<RetrainStatus>("/retrain/status"), refetchInterval: 5_000 });

export const useJobs = () => useQuery({ queryKey: ["jobs"], queryFn: () => get<Job[]>("/retrain/jobs"), refetchInterval: 10_000 });

export const useJob = (jobId: string | null) =>
  useQuery({
    queryKey: ["job", jobId],
    queryFn: () => get<Job>(`/retrain/status/${jobId}`),
    enabled: Boolean(jobId),
    // Poll quickly while the job is active, then stop.
    refetchInterval: (query) => (query.state.data && ["completed", "failed"].includes(query.state.data.status) ? false : 2_000),
  });

export const useUploads = (enabled: boolean) =>
  useQuery({ queryKey: ["uploads"], queryFn: () => get<Uploads>("/uploads"), enabled, refetchInterval: 15_000 });

export type AuthStatus = {
  admin_key_required: boolean;
  accounts_exist: boolean;
  signup_open: boolean;
  first_account_becomes_admin: boolean;
  open_mode: boolean;
  min_password_length: number;
};

export const useAuthStatus = () =>
  useQuery({ queryKey: ["auth-status"], queryFn: () => get<AuthStatus>("/auth/status"), staleTime: 10_000 });

export type MongoStatus = {
  enabled: boolean;
  configured: boolean;
  cluster: string | null;
  database: string;
  store_images: boolean;
  sync_interval_seconds: number;
  last_sync: string | null;
  last_error: string | null;
  last_result: Record<string, number>;
  documents?: Record<string, number>;
};

export const useMongoStatus = (enabled: boolean) =>
  useQuery({ queryKey: ["mongo"], queryFn: () => get<MongoStatus>("/mongo/status"), enabled, refetchInterval: 20_000 });

export const useUsers = (enabled: boolean) =>
  useQuery({ queryKey: ["users"], queryFn: () => get<User[]>("/users"), enabled, refetchInterval: 30_000 });
