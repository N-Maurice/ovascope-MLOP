// Shapes of the API responses used by the interface (see api/main.py).

export type ClassEntry = { class_id: number; original_name: string; display_name: string; folder: string };

export type InputCheck = {
  enabled: boolean;
  is_parasite_egg?: boolean;
  egg_score?: number;
  threshold?: number;
  created_at?: string;
  test_egg_acceptance_rate?: number;
  test_non_egg_rejection_rate?: number;
};

export type Health = {
  status: string;
  service_uptime_seconds: number;
  model_loaded: boolean;
  model_version: string | null;
  model_error: string | null;
  instance: string;
  input_check: InputCheck;
};

export type Prediction = {
  predicted_class_id: number;
  predicted_class: string;
  class_key: string;
  original_label: string;
  confidence: number;
  top_k: { class_id: number; class_name: string; confidence: number }[];
  model_version: string;
  input_check: InputCheck;
  note: string;
};

export type NotEggDetail = { error: "not_a_parasite_egg"; message: string; is_parasite_egg: false; egg_score: number; threshold: number };

export type PerClass = { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number | null; pr_auc: number | null; log_loss: number; support: number };

export type Metrics = {
  samples: number;
  accuracy: number;
  log_loss: number;
  balanced_accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  roc_auc_macro: number | null;
  pr_auc_macro: number | null;
  per_class: Record<string, PerClass>;
  confusion_matrix: number[][];
  top_confusions: { true: string; predicted: string; count: number }[];
  model_version?: string;
};

export type ComparisonRow = Record<string, string | number | null>;

export type Evaluation = {
  model_version: string;
  test: Metrics;
  validation: Metrics | null;
  comparison: ComparisonRow[];
  by_resolution: { resolution: string; val_crops: number; val_accuracy: number; test_crops: number | null; test_accuracy: number | null }[];
  class_names: string[];
};

export type HistoryRow = Record<string, number | string | null> & { epoch: number; phase?: string };

export type ModelVersion = {
  version: string;
  kind: string;
  architecture: string;
  parent_version: string | null;
  created_at: string;
  validation_metrics: Partial<Metrics> | null;
  test_metrics: Partial<Metrics> | null;
  model_size_mb: number;
  active: boolean;
};

export type DatasetSummary = {
  crops_total: number;
  crops_per_split: Record<string, number>;
  crops_per_split_class: Record<string, Record<string, number>>;
  source_image_sizes: Record<string, number>;
  crop_side_px: { min: number; median: number; max: number; share_below_224: number };
  categories_identical_train_test: boolean;
  class_map_matches_annotations: boolean;
  source_images_in_multiple_splits: unknown[];
  exact_duplicates_across_splits: unknown[];
  exact_duplicate_crops_across_splits: number;
  near_duplicate_pairs_across_splits: Record<string, number>;
  near_duplicate_threshold_bits: number;
  test_to_trainval_closest_distance: { min: number; p5: number; median: number };
  non_square_crops: number;
  empty_or_flat_crops: number;
  source_photos: { train_and_val: number; test: number };
  classes: ClassEntry[];
};

export type GateSplitReport = {
  images: number;
  egg_images: number;
  non_egg_images: number;
  egg_acceptance_rate: number;
  non_egg_rejection_rate: number;
  roc_auc: number;
  confusion: Record<string, number>;
  per_source: Record<string, { images: number; rate: number; rate_meaning: string; median_score: number }>;
};

export type GateReport =
  | { enabled: false }
  | {
      enabled: true;
      created_at: string;
      architecture: string;
      threshold: number;
      target_egg_recall: number;
      negative_sources: string[];
      training_images: { egg: number; non_egg: number };
      epochs_completed: number;
      training_seconds: number;
      validation: GateSplitReport;
      test: GateSplitReport;
      history: Record<string, number[]> | { epoch: number; [k: string]: number }[];
      model_size_mb: number;
      classifier_on_non_eggs: { source: string; images: number; median_confidence: number; share_confidence_over_0_9: number; most_common_prediction: string }[];
    };

export type ServiceMetrics = {
  uptime_seconds: number;
  process_started_utc: string;
  prediction_requests_total: number;
  prediction_requests_successful: number;
  prediction_requests_failed: number;
  prediction_requests_since_start: number;
  average_latency_ms: number | null;
  recent_latency_ms: { count: number; p50: number | null; p95: number | null; max: number | null };
  recent_requests: { timestamp: string; status_code: number; latency_ms: number; model_version: string | null; predicted_class_id: number | null; confidence: number | null }[];
  recent_retraining: { id: string; status: string; created_at: string; finished_at: string | null; candidate_version: string | null; promotion_decision: string | null }[];
  model_version: string | null;
  instance: string;
};

export type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  progress: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  trigger: string;
  n_new_images: number | null;
  base_version: string | null;
  candidate_version: string | null;
  train_metrics: Record<string, number> | null;
  candidate_metrics: Record<string, number> | null;
  active_metrics: Record<string, number> | null;
  promotion_decision: string | null;
  promoted: boolean;
  error: string | null;
  requested_by: string | null;
};

export type RetrainStatus = {
  pending_new_images: number;
  trigger_threshold: number;
  retraining_recommended: boolean;
  min_new_images: number;
  active_model: string | null;
  job_running: boolean;
  latest_job: Job | null;
};

export type UploadRecord = {
  id: number;
  filename: string;
  class_id: number;
  class_label: string;
  file_hash: string;
  width: number;
  height: number;
  upload_timestamp: string;
  validation_status: string;
  training_status: string;
  used_in_job: string | null;
  uploaded_by: string | null;
  bytes: number;
};

export type Uploads = {
  counts: { pending: number; used: number; total: number; pending_per_class: Record<string, number> };
  records: UploadRecord[];
};

export type UploadResult = {
  accepted: { ok: true; id: number; filename: string; class_id: number; class_label: string }[];
  rejected: { ok: false; filename: string; reason: string }[];
  saved_to_database: number;
  retraining: RetrainStatus & { auto_started_job?: string; auto_start_skipped?: string };
};
