import type {
  AnalysisEvent,
  BatchSnapshot,
  FindingDetail,
  PendingUpload,
  ReviewRun,
  RetryJob,
  StoredFinding,
  UploadSuggestion,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

const DEFAULT_TIMEOUT_MS = 30_000;
const UPLOAD_TIMEOUT_MS = 10 * 60_000;

export class ApiConnectionError extends Error {
  constructor(message = "无法连接分析服务，请确认后端已启动。") {
    super(message);
    this.name = "ApiConnectionError";
  }
}

async function responseJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string"
      ? detail
      : typeof detail?.message === "string"
        ? detail.message
        : "请求失败，请检查服务日志。";
    throw new Error(message);
  }
  return payload as T;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      cache: init?.cache ?? "no-store",
      signal: controller.signal,
    });
    return await responseJson<T>(response);
  } catch (error) {
    if (error instanceof ApiConnectionError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiConnectionError("分析服务响应超时，请检查后端状态后重试。");
    }
    if (error instanceof TypeError) {
      throw new ApiConnectionError();
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const payload = await requestJson<{ status: string }>("/health", undefined, 4_000);
    return payload.status === "ok";
  } catch {
    return false;
  }
}

export async function inspectFiles(files: File[]): Promise<UploadSuggestion[]> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const payload = await requestJson<{ files: UploadSuggestion[] }>("/uploads/inspect", {
    method: "POST",
    body: form,
  }, UPLOAD_TIMEOUT_MS);
  return payload.files;
}

export type ReasoningModel = "deepseek_flash" | "deepseek_pro" | "kimi";
export type AgentRoleKey = "discovery" | "finance" | "legal" | "business" | "market" | "decision";
export type AgentModelAssignments = Record<AgentRoleKey, ReasoningModel>;

function uploadForm(items: PendingUpload[], agentModels: AgentModelAssignments): FormData {
  const form = new FormData();
  items.forEach((item) => form.append("files", item.file));
  form.append(
    "metadata",
    JSON.stringify(
      items.map((item) => ({
        file_name: item.file_name,
        company_id: item.company_id,
        company_name: item.company_name,
        document_date: item.document_date,
        listing_date: item.listing_date,
        force_reanalysis: item.force_reanalysis,
      })),
    ),
  );
  form.append("reasoning_model", agentModels.discovery);
  form.append("agent_models", JSON.stringify(agentModels));
  return form;
}

export async function createBatch(
  items: PendingUpload[],
  agentModels: AgentModelAssignments,
): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>("/batches", {
    method: "POST",
    body: uploadForm(items, agentModels),
  }, UPLOAD_TIMEOUT_MS);
}

export async function appendBatch(
  batchId: string,
  items: PendingUpload[],
  agentModels: AgentModelAssignments,
): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>(`/batches/${batchId}/append`, {
    method: "POST",
    body: uploadForm(items, agentModels),
  }, UPLOAD_TIMEOUT_MS);
}

export async function getBatch(batchId: string): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>(`/batches/${batchId}`);
}

export async function batchAction(
  batchId: string,
  action: "pause" | "resume",
): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>(`/batches/${batchId}/${action}`, {
    method: "POST",
  });
}

export async function submitRiskScope(
  batchId: string,
  companyKey: string,
  payload: unknown,
): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>(
    `/batches/${batchId}/companies/${encodeURIComponent(companyKey)}/risk-scope`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function resolveLlmFailure(
  batchId: string,
  companyKey: string,
  payload: {
    request_id: string;
    run_id: string;
    action: "retry_llm" | "fallback_bm25";
  },
): Promise<BatchSnapshot> {
  return requestJson<BatchSnapshot>(
    `/batches/${batchId}/companies/${encodeURIComponent(companyKey)}/llm-recovery`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function rerunPrediction(
  batchId: string,
  companyKey: string,
  variant: "auto" | "base" | "agent_enhanced" = "auto",
): Promise<{ prediction: import("./types").RuntimePrediction; batch: BatchSnapshot }> {
  return requestJson(
    `/batches/${batchId}/companies/${encodeURIComponent(companyKey)}/prediction`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ variant }),
    },
  );
}

export function connectBatchStream(
  batchId: string,
  onSnapshot: (snapshot: BatchSnapshot) => void,
  onEvents: (companyKey: string, events: AnalysisEvent[]) => void,
  onConnection: (connected: boolean) => void,
): EventSource {
  const source = new EventSource(`${API_BASE}/batches/${batchId}/stream`);
  source.addEventListener("snapshot", (event) => {
    onSnapshot(JSON.parse((event as MessageEvent).data) as BatchSnapshot);
    onConnection(true);
  });
  source.addEventListener("company-events", (event) => {
    const payload = JSON.parse((event as MessageEvent).data) as {
      company_key: string;
      events: AnalysisEvent[];
    };
    onEvents(payload.company_key, payload.events);
  });
  source.addEventListener("company-structured-events", (event) => {
    const payload = JSON.parse((event as MessageEvent).data) as {
      company_key: string;
      events: AnalysisEvent[];
    };
    onEvents(payload.company_key, payload.events);
  });
  source.onerror = () => onConnection(false);
  return source;
}

export async function listReviewRuns(): Promise<ReviewRun[]> {
  const payload = await requestJson<{ runs: ReviewRun[] }>("/reports/runs");
  return payload.runs;
}

export async function deleteAnalysisRun(runId: string): Promise<{
  deleted: boolean;
  run_id: string;
  recovery_available: boolean;
  moved_file_count: number;
  shared_file_count: number;
}> {
  return requestJson(`/reports/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
}

export async function rerunStoredPrediction(
  runId: string,
  variant: "auto" | "base" | "agent_enhanced" = "auto",
): Promise<{ prediction: import("./types").RuntimePrediction; run: ReviewRun }> {
  return requestJson(`/reports/runs/${encodeURIComponent(runId)}/prediction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ variant }),
  });
}

export async function listRunFindings(runId: string): Promise<StoredFinding[]> {
  const payload = await requestJson<{ findings: StoredFinding[] }>(`/reports/runs/${encodeURIComponent(runId)}/findings`);
  return payload.findings;
}

export async function getAuditMaterials(
  runId: string,
): Promise<import("./types").AuditMaterials> {
  return requestJson(
    `/reports/runs/${encodeURIComponent(runId)}/audit-materials`,
  );
}

export async function getFindingDetail(
  runId: string,
  findingId: string,
): Promise<FindingDetail> {
  return requestJson<FindingDetail>(
    `/reports/runs/${encodeURIComponent(runId)}/findings/${encodeURIComponent(findingId)}`,
  );
}

export async function reviewFinding(
  runId: string,
  findingId: string,
  payload: {
    decision: "confirmed" | "rejected" | "edited";
    reviewer: string;
    comment?: string;
    manual_override?: boolean;
    title?: string;
    severity?: string;
    claim?: string;
    confidence?: number;
  },
): Promise<void> {
  await requestJson(
    `/reports/runs/${encodeURIComponent(runId)}/findings/${encodeURIComponent(findingId)}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function listRetryJobs(runId: string): Promise<RetryJob[]> {
  const payload = await requestJson<{ jobs: RetryJob[] }>(`/reports/runs/${encodeURIComponent(runId)}/retry-jobs`);
  return payload.jobs;
}

export async function retryFinding(runId: string, findingId: string): Promise<RetryJob> {
  const payload = await requestJson<{ job: RetryJob }>(
    `/reports/runs/${encodeURIComponent(runId)}/findings/${encodeURIComponent(findingId)}/retry`,
    { method: "POST" },
  );
  return payload.job;
}

export async function retryAgentFailures(runId: string, agent: string): Promise<RetryJob> {
  const payload = await requestJson<{ job: RetryJob }>(
    `/reports/runs/${encodeURIComponent(runId)}/agents/${encodeURIComponent(agent)}/retry`,
    { method: "POST" },
  );
  return payload.job;
}

export async function retryRunFailures(runId: string): Promise<RetryJob> {
  const payload = await requestJson<{ job: RetryJob }>(
    `/reports/runs/${encodeURIComponent(runId)}/retry-technical-failures`,
    { method: "POST" },
  );
  return payload.job;
}

export async function exportRun(runId: string): Promise<Record<string, string>> {
  const payload = await requestJson<{ files: Record<string, string> }>(
    `/reports/runs/${encodeURIComponent(runId)}/export`,
    { method: "POST" },
  );
  return payload.files;
}

export function evidenceImageUrl(runId: string, evidenceId: string): string {
  return `${API_BASE}/reports/runs/${encodeURIComponent(runId)}/evidence/${encodeURIComponent(evidenceId)}/image`;
}
