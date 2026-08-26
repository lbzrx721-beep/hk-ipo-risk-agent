import type { AnalysisEvent, BatchCompany } from "@/types";

export type AppView = "workspace" | "predictions" | "results";
export type FindingFilter = "all" | "pending" | "verified" | "rejected";

export const ACTIVE_BATCH_KEY = "ipo-risk-active-batch";

export const stateLabel: Record<string, string> = {
  pending: "等待中",
  preparing: "解析与索引",
  ready: "等待 Agent",
  awaiting_recovery: "等待处理模型异常",
  awaiting_review: "待确认风险",
  running: "分析中",
  pausing: "准备暂停",
  paused: "已暂停",
  completed: "已完成",
  failed: "异常",
};

export const roleMeta = [
  ["discovery", "风险发现 Agent", "定位原文并形成候选风险"],
  ["finance", "财务 Agent", "财务质量、现金流与集中度"],
  ["legal", "法律 Agent", "条款、诉讼、合规与权利义务"],
  ["business", "业务 Agent", "商业模式、客户与供应链"],
  ["market", "市场 Agent", "发行结构、流动性与市场情绪"],
  ["decision", "决策 Agent", "整合证据、冲突和不确定性"],
] as const;

export const severityLabel: Record<string, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
  critical: "极高风险",
};

export const findingStatusLabel: Record<string, string> = {
  proposed: "待核验",
  pending_verification: "待复核",
  verified: "已确认",
  rejected: "已驳回",
};

export function compactDetail(detail: string, limit = 150): string {
  const normalized = detail.replace(/\[\[batch:[^\]]+\]\]/g, "").replace(/\s+/g, " ").trim();
  if (normalized.startsWith("{") || normalized.startsWith("[")) {
    return "模型正在生成结构化结论，完整内容已写入审计事件。";
  }
  return normalized.length > limit ? `${normalized.slice(0, limit)}…` : normalized;
}

export function eventRole(stage: string): string | null {
  for (const [key] of roleMeta) {
    if (stage.includes(`agent:${key}`) || stage.includes(key)) return key;
  }
  if (stage.includes("risk_discovery") || stage.includes("discovery")) return "discovery";
  return null;
}

export function roleState(events: AnalysisEvent[], key: string, company: BatchCompany) {
  const projected = company.progress?.agents?.[key];
  if (projected) {
    const display = {
      complete: ["已完成", "success"],
      degraded: ["部分降级", "warning"],
      failed: ["异常", "danger"],
      skipped: [key === "discovery" ? "未启用" : "本次无任务", "muted"],
      running: ["分析中", "active"],
      pending: ["等待", "muted"],
    } as const;
    const [state, tone] = display[projected.state as keyof typeof display] ?? [projected.state, "muted"];
    return {
      state,
      tone,
      detail: projected.message || "等待上游任务分派。",
      progress: Math.max(0, Math.min(projected.progress ?? 0, 100)),
      progressText: projected.progress_text || "尚未开始",
    };
  }
  const matches = events.filter((event) => eventRole(event.stage) === key);
  const latest = matches.at(-1);
  const completed = matches.some((event) => event.stage.endsWith(":completed"));
  const failed = matches.some((event) => event.stage.includes("failed"));
  const degraded = matches.some((event) => event.stage.endsWith(":degraded"));
  const skipped = matches.some((event) => event.stage.endsWith(":skipped"));
  if (failed) return { state: "异常", tone: "danger" as const, detail: compactDetail(latest?.detail ?? ""), progress: 100, progressText: "执行失败" };
  if (degraded) return { state: "部分降级", tone: "warning" as const, detail: compactDetail(latest?.detail ?? ""), progress: 100, progressText: "部分批次失败" };
  if (skipped) return { state: key === "discovery" ? "未启用" : "本次无任务", tone: "muted" as const, detail: compactDetail(latest?.detail ?? ""), progress: 100, progressText: "无需调用模型" };
  if (completed) return { state: "已完成", tone: "success" as const, detail: compactDetail(latest?.detail ?? ""), progress: 100, progressText: "任务完成" };
  if (latest) return { state: "分析中", tone: "active" as const, detail: compactDetail(latest.detail), progress: 50, progressText: "正在处理" };
  if (company.state === "completed") return { state: "已完成", tone: "success" as const, detail: "结果已写入分析产物。", progress: 100, progressText: "任务完成" };
  return { state: "等待", tone: "muted" as const, detail: "等待上游任务分派。", progress: 0, progressText: "尚未开始" };
}

export function statusProgress(company: BatchCompany): number {
  if (company.progress) return Math.max(0, Math.min(100, company.progress.max_percent));
  const values: Record<string, number> = {
    pending: 3,
    preparing: 18,
    ready: 28,
    awaiting_recovery: 38,
    awaiting_review: 42,
    running: 68,
    pausing: 75,
    paused: 75,
    completed: 100,
    failed: 100,
  };
  return values[company.state] ?? 5;
}

function durableEvent(event: AnalysisEvent): boolean {
  const action = event.stage.slice(event.stage.lastIndexOf(":") + 1);
  return [
    "structured",
    "risk_catalog",
    "raw_completed",
    "consolidated",
    "completed",
    "degraded",
    "failed",
    "skipped",
  ].includes(action) || [
    "features_completed",
    "features_failed",
    "prediction_started",
    "prediction_completed",
    "prediction_failed",
    "features_started",
  ].includes(event.stage) || event.stage.startsWith("risk_scope_review_");
}

function eventIdentity(event: AnalysisEvent): string {
  const marker = /^\[\[(?:batch|consolidation):\d+\/\d+]]/.exec(event.detail)?.[0];
  const isModelStream = event.stage.endsWith(":delta") || event.stage.endsWith(":structured");
  if (isModelStream) {
    const roleStream = event.stage.split(":").slice(0, 2).join(":");
    return `${roleStream}:stream:${marker ?? "default"}`;
  }
  return `${event.stage}:${event.occurred_at}:${event.detail}`;
}

export function mergeAnalysisEvents(
  previous: AnalysisEvent[],
  incoming: AnalysisEvent[],
): AnalysisEvent[] {
  const merged = new Map(previous.map((event) => [eventIdentity(event), event]));
  incoming.forEach((event) => {
    const identity = eventIdentity(event);
    const existing = merged.get(identity);
    if (existing?.stage.endsWith(":structured") && event.stage.endsWith(":delta")) return;
    merged.set(identity, event);
  });
  const ordered = [...merged.values()].sort(
    (left, right) => new Date(left.occurred_at).getTime() - new Date(right.occurred_at).getTime(),
  );
  const durable = ordered.filter(durableEvent);
  const transient = ordered.filter((event) => !durableEvent(event)).slice(-240);
  return [...durable, ...transient].sort(
    (left, right) => new Date(left.occurred_at).getTime() - new Date(right.occurred_at).getTime(),
  );
}

export function elapsedSeconds(company: BatchCompany, nowMs: number): number | null {
  if (company.duration_seconds != null) {
    if (!["running", "preparing", "ready", "pausing"].includes(company.state)) {
      return Math.max(0, company.duration_seconds);
    }
    const updatedAt = company.duration_updated_at
      ? new Date(company.duration_updated_at).getTime()
      : Number.NaN;
    if (Number.isFinite(updatedAt)) {
      return Math.max(0, company.duration_seconds + (nowMs - updatedAt) / 1000);
    }
  }
  if (!company.started_at) return company.duration_seconds ?? null;
  const startedAt = new Date(company.started_at).getTime();
  if (!Number.isFinite(startedAt)) return company.duration_seconds ?? null;
  const endedAt = company.ended_at ? new Date(company.ended_at).getTime() : nowMs;
  return Math.max(0, (endedAt - startedAt) / 1000);
}

export function batchProgress(companies: BatchCompany[]): number {
  if (companies.length === 0) return 0;
  return Math.round(
    companies.reduce((total, company) => total + statusProgress(company), 0) / companies.length,
  );
}

export function companyStage(company: BatchCompany, events: AnalysisEvent[]): number {
  if (company.state === "completed") return 8;
  if (events.some((event) => event.stage === "completed")) return 8;
  if (
    company.prediction
    || events.some((event) => ["prediction_started", "prediction_completed", "prediction_failed"].includes(event.stage))
  ) return 7;
  if (events.some((event) => ["features_started", "features_completed", "features_failed"].includes(event.stage))) return 6;
  const agents = company.progress?.agents;
  if (agents?.decision && agents.decision.state !== "pending") return 5;
  if (["finance", "legal", "business", "market"].some((role) => agents?.[role] && agents[role].state !== "pending")) return 4;
  if (events.some((event) => eventRole(event.stage) === "decision")) return 5;
  if (events.some((event) => ["finance", "legal", "business", "market"].includes(eventRole(event.stage) ?? ""))) return 4;
  if (company.state === "awaiting_recovery") return 2;
  if (company.state === "awaiting_review") return 3;
  if (events.some((event) => eventRole(event.stage) === "discovery")) return 2;
  if (["ready", "running", "pausing", "paused"].includes(company.state)) return 2;
  if (company.state === "preparing") return 1;
  return 0;
}

export function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleTimeString("zh-CN", { hour12: false });
}

export function formatDuration(seconds?: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${Math.round(seconds)}秒`;
  const minutes = Math.floor(seconds / 60);
  const remain = Math.round(seconds % 60);
  return `${minutes}分${remain}秒`;
}

export function downloadFromUrl(url: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.target = "_blank";
  anchor.rel = "noreferrer";
  anchor.click();
}
