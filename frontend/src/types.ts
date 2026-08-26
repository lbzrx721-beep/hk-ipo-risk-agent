export type RiskCandidate = {
  risk_type: string;
  title: string;
  owner: "finance" | "legal" | "business" | "market";
  severity: "low" | "medium" | "high" | "critical";
  ai_recommended: boolean;
  rationale: string;
  review_question: string;
  source: string;
  confidence?: number | null;
  source_candidate_count?: number;
  evidence_count?: number;
};

export type RiskScopeReview = {
  run_id: string;
  industry: string;
  business_model: string;
  revenue_model: string;
  candidates: RiskCandidate[];
  requested_at?: string;
  funnel?: {
    document_batches?: number;
    raw_candidates?: number;
    consolidated_candidates?: number;
    catalog_added?: number;
    catalog_merged?: number;
    scoped_in?: number;
    scoped_out?: number;
  };
};

export type LlmFailureRecovery = {
  request_id: string;
  run_id: string;
  stage: string;
  agent: string;
  provider?: string | null;
  model?: string | null;
  reason: string;
  technical_detail: string;
  attempt: number;
  requested_at?: string;
};

export type PredictionKeyInput = {
  feature: string;
  label: string;
  value: number | boolean | string | null;
};

export type RecallGuardSignal = {
  feature: string;
  label: string;
  value: number | null;
  historical_percentile: number | null;
  risk_contribution: number;
  available: boolean;
};

export type HighRiskRecallGuard = {
  schema_version: string;
  enabled: boolean;
  policy_version?: string;
  watchlist: boolean;
  caught_beyond_primary_model: boolean;
  score?: number;
  threshold?: number;
  available_feature_count?: number;
  feature_count?: number;
  reasons: string[];
  feature_signals?: RecallGuardSignal[];
  note: string;
};

export type RuntimePrediction = {
  schema_version: string;
  prediction_type: "significant_drop_5d";
  prediction_variant?: "base" | "agent_enhanced" | string;
  run_id: string;
  company_id: string;
  company_name: string;
  generated_at: string;
  model_name: string;
  probability: number;
  predicted_significant_drop_5d: boolean;
  decision: "high_risk" | "low_risk" | "uncertain" | string;
  requires_review: boolean;
  risk_level: "low" | "medium" | "high" | string;
  confidence: number;
  feature_count: number;
  available_feature_count: number;
  feature_completeness: number;
  candidate_feature_count?: number;
  available_candidate_feature_count?: number;
  feature_readiness?: "ready" | "review_required" | "blocked" | string;
  feature_group_completeness?: Record<string, number>;
  missing_model_features?: string[];
  matched_reference_features: boolean;
  key_inputs: PredictionKeyInput[];
  recall_guard?: HighRiskRecallGuard;
  warnings: string[];
  disclaimer: string;
};

export type BatchCompany = {
  company_key: string;
  segment_id: string;
  company_id: string;
  company_folder: string;
  company_name: string;
  prospectus_date: string;
  split: string;
  state: string;
  started_at?: string | null;
  ended_at?: string | null;
  duration_seconds?: number | null;
  duration_updated_at?: string | null;
  review_wait_seconds?: number | null;
  run_id?: string | null;
  finding_count?: number | null;
  evidence_count?: number | null;
  error?: string | null;
  preparation_state?: string | null;
  preparation_error?: string | null;
  risk_scope_review?: RiskScopeReview | null;
  llm_failure_recovery?: LlmFailureRecovery | null;
  feature_path?: string | null;
  base_feature_path?: string | null;
  base_prediction?: RuntimePrediction | null;
  base_prediction_path?: string | null;
  base_prediction_status?: string | null;
  base_prediction_error?: string | null;
  prediction?: RuntimePrediction | null;
  prediction_path?: string | null;
  prediction_status?: string | null;
  prediction_error?: string | null;
  progress?: CompanyProgress | null;
};

export type AgentProgress = {
  state: "pending" | "running" | "complete" | "failed" | "degraded" | "skipped" | string;
  message: string;
  progress: number;
  progress_text: string;
  sequence_current?: number;
};

export type CompanyProgress = {
  max_percent: number;
  overall_text: string;
  current_activity: string;
  agents: Record<string, AgentProgress>;
  risk_titles?: Record<string, Record<string, string>>;
  discovery_failures?: string[];
  background_tasks?: Record<string, {
    state: string;
    progress: number;
    message: string;
  }>;
  funnel?: {
    document_batches?: number;
    raw_candidates?: number;
    consolidated_candidates?: number;
    scoped_in?: number;
    scoped_out?: number;
    human_selected?: number;
    role_assignments?: Record<string, number>;
    final_accepted?: number;
  };
};

export type BatchSnapshot = {
  batch_id: string;
  created_at: string;
  state: string;
  company_count: number;
  counts: Record<string, number>;
  companies: BatchCompany[];
  segments: Array<{
    segment_id: string;
    state: string;
    phase: string;
    phase_detail?: string | null;
    active: boolean;
    error?: string | null;
  }>;
};

export type AnalysisEvent = {
  stage: string;
  detail: string;
  occurred_at: string;
};

export type UploadSuggestion = {
  file_name: string;
  company_id: string | null;
  company_name: string | null;
  document_date: string | null;
  source: string;
  page_count: number;
  source_hash: string;
  existing_run_id?: string | null;
  existing_completed_at?: string | null;
};

export type PendingUpload = UploadSuggestion & {
  id: string;
  file: File;
  force_reanalysis: boolean;
};

export type ReviewRun = {
  run_id: string;
  company_id: string;
  company_name: string;
  as_of_date: string;
  total_findings: number;
  pending_findings: number;
  verified_findings: number;
  rejected_findings: number;
  reviewed_findings: number;
  complete: boolean;
  report_path?: string | null;
  verified_report_path?: string | null;
  base_feature_available: boolean;
  agent_feature_available: boolean;
  base_prediction?: RuntimePrediction | null;
  prediction?: RuntimePrediction | null;
};

export type StoredFinding = {
  finding_id: string;
  company_id: string;
  risk_type: string;
  title: string;
  severity: "low" | "medium" | "high" | "critical";
  claim: string;
  evidence_ids: string[];
  status: string;
  confidence: number;
  created_by: string;
  uncertainty?: string | null;
  specialist_status?: string | null;
};

export type RetryItem = {
  finding_id: string;
  state: "queued" | "running" | "succeeded" | "failed";
  attempt: number;
  stage: string;
  error?: string | null;
};

export type RetryJob = {
  job_id: string;
  run_id: string;
  scope: string;
  state: "queued" | "running" | "succeeded" | "failed";
  max_attempts: number;
  created_at: string;
  updated_at: string;
  detail: string;
  items: Record<string, RetryItem>;
};

export type StoredEvidence = {
  evidence_id: string;
  document_id: string;
  source_file: string;
  page: number;
  printed_page_label?: string | null;
  bbox: [number, number, number, number];
  quote: string;
  evidence_type: string;
  extractor: string;
  confidence: number;
  screenshot_path?: string | null;
  table_id?: string | null;
  table_cell_refs: string[];
  table_cell_values: string[];
};

export type VerificationCheck = {
  name: string;
  passed: boolean;
  detail: string;
};

export type FindingDetail = {
  finding: StoredFinding;
  retryable_failure: boolean;
  evidences: StoredEvidence[];
  verification: {
    passed: boolean;
    checks: VerificationCheck[];
    evidence_verifications: Array<{
      evidence_id: string;
      passed: boolean;
      checks: VerificationCheck[];
    }>;
  };
  reviews: Array<{
    review_id: string;
    decision: string;
    reviewer: string;
    comment?: string | null;
    created_at: string;
  }>;
  agent_traces: Array<{
    trace_id: string;
    agent: string;
    decision: string;
    status: string;
    started_at: string;
    ended_at: string;
    input_refs: string[];
  }>;
};
