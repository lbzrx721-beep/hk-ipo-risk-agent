import {
  CheckCircle2,
  Database,
  Download,
  FileCheck2,
  FileSearch,
  History,
  Search,
  ShieldAlert,
  RefreshCw,
  Trash2,
  UserCheck,
  WifiOff,
  XCircle,
} from "lucide-react";
import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { VirtualList } from "@/components/virtual-list";
import { DebateTimeline } from "@/components/debate-timeline";
import {
  TableEvidenceViewer,
  tableEvidenceReviewLabel,
  type TableEvidenceReview,
} from "@/features/reports/table-evidence-viewer";
import {
  deleteAnalysisRun,
  evidenceImageUrl,
  exportRun,
  getAuditMaterials,
  getFindingDetail,
  listReviewRuns,
  listRetryJobs,
  listRunFindings,
  retryAgentFailures,
  retryFinding,
  retryRunFailures,
  reviewFinding,
} from "@/api";
import {
  downloadFromUrl,
  findingStatusLabel,
  formatTime,
  severityLabel,
  type FindingFilter,
} from "@/lib/presentation";
import { cn } from "@/lib/utils";
import type { AuditMaterials, AuditToolCall, FindingDetail, RetryItem, RetryJob, ReviewRun, StoredFinding } from "@/types";

type ReportsWorkspaceProps = {
  active: boolean;
  serviceAvailable: boolean;
  initialRunId?: string | null;
  initialFindingId?: string | null;
  onReconnect: () => Promise<void>;
  onSelectionChange?: (runId: string | null, findingId: string | null) => void;
  setError: (value: string | null) => void;
};

function severityVariant(severity: StoredFinding["severity"]): "default" | "secondary" | "destructive" | "outline" {
  if (severity === "critical" || severity === "high") return "destructive";
  if (severity === "medium") return "default";
  return "secondary";
}

function hasTechnicalFailure(finding: StoredFinding): boolean {
  if (finding.specialist_status === "technical_failure") return true;
  const uncertainty = finding.uncertainty ?? "";
  return (uncertainty.includes("专业Agent批量核验失败") || uncertainty.includes("Agent技术失败"))
    && ["incomplete chunked read", "peer closed connection", "connection reset", "server overloaded", "serveroverloaded", "too many requests", "error code: 429"]
      .some((marker) => uncertainty.toLowerCase().includes(marker));
}

function summarizeToolCalls(calls: AuditToolCall[]) {
  const grouped = new Map<string, { count: number; failed: number; first: string; last: string; error?: string | null }>();
  for (const call of calls) {
    const current = grouped.get(call.tool_name);
    if (current) {
      current.count += 1;
      current.failed += call.status === "failed" ? 1 : 0;
      current.last = call.ended_at;
      current.error ||= call.error;
    } else {
      grouped.set(call.tool_name, { count: 1, failed: call.status === "failed" ? 1 : 0, first: call.started_at, last: call.ended_at, error: call.error });
    }
  }
  return [...grouped.entries()].map(([tool, summary]) => ({ tool, ...summary }));
}

type FindingRetryStatus = RetryItem & {
  maxAttempts: number;
  updatedAt: string;
};

function latestRetryItemsByFinding(jobs: RetryJob[]): Map<string, FindingRetryStatus> {
  const result = new Map<string, FindingRetryStatus>();
  const orderedJobs = [...jobs].sort((left, right) => (
    Date.parse(left.updated_at) - Date.parse(right.updated_at)
  ));
  for (const job of orderedJobs) {
    for (const item of Object.values(job.items)) {
      result.set(item.finding_id, {
        ...item,
        maxAttempts: job.max_attempts,
        updatedAt: job.updated_at,
      });
    }
  }
  return result;
}

function RunItem({ run, selected, onSelect }: { run: ReviewRun; selected: boolean; onSelect: (id: string) => void }) {
  const percent = run.total_findings
    ? Math.round((run.reviewed_findings / run.total_findings) * 100)
    : 0;
  return (
    <Button
      variant="outline"
      className={cn(
        "h-auto w-full justify-start rounded-xl px-3 py-3 text-left",
        selected && "border-primary/50 bg-primary/8 ring-1 ring-primary/15",
      )}
      onClick={() => onSelect(run.run_id)}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex items-start justify-between gap-2">
          <strong className="truncate text-sm">{run.company_id} · {run.company_name}</strong>
          <Badge variant={run.complete ? "secondary" : "outline"}>
            {run.complete ? "已完成" : `${run.pending_findings} 待复核`}
          </Badge>
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{run.as_of_date || "日期待确认"}</span>
          <span>{run.total_findings} 条风险</span>
        </div>
        <Progress value={percent} />
      </div>
    </Button>
  );
}

function FindingItem({ finding, index, retry, selected, onSelect }: {
  finding: StoredFinding;
  index: number;
  retry?: FindingRetryStatus;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  const statusBadge = retry?.state === "queued"
    ? <Badge variant="outline">等待重试</Badge>
    : retry?.state === "running"
      ? (
        <Badge variant="secondary">
          <Spinner data-icon="inline-start" />
          重试中 · {Math.max(1, retry.attempt)}/{retry.maxAttempts}
        </Badge>
      )
      : retry?.state === "failed"
        ? <Badge variant="destructive">重试失败</Badge>
        : (
          <Badge variant={hasTechnicalFailure(finding) && retry?.state !== "succeeded" ? "destructive" : "outline"}>
            {hasTechnicalFailure(finding) && retry?.state !== "succeeded"
              ? "技术失败"
              : findingStatusLabel[finding.status] ?? finding.status}
          </Badge>
        );
  return (
    <Button
      variant="outline"
      className={cn(
        "h-auto w-full justify-start rounded-xl px-3 py-3 text-left",
        selected && "border-primary/50 bg-primary/8 ring-1 ring-primary/15",
      )}
      onClick={() => onSelect(finding.finding_id)}
    >
      <span className="w-7 shrink-0 self-start pt-0.5 font-mono text-xs text-muted-foreground">
        {String(index + 1).padStart(2, "0")}
      </span>
      <span className="min-w-0 flex-1">
        <strong className="line-clamp-2 text-sm leading-5">{finding.title}</strong>
        <small className="mt-1 block truncate text-muted-foreground">
          {finding.risk_type.replaceAll("_", " · ")} · {finding.evidence_ids.length} 条证据
        </small>
      </span>
      <span className="flex shrink-0 flex-col items-end gap-1">
        <Badge variant={severityVariant(finding.severity)}>{severityLabel[finding.severity]}</Badge>
        {statusBadge}
      </span>
    </Button>
  );
}

function SummaryCard({ label, value, suffix = "", emphasized = false }: { label: string; value: number | string; suffix?: string; emphasized?: boolean }) {
  return (
    <div className={cn("rounded-xl border bg-muted/15 px-3 py-2", emphasized && "border-primary/45 bg-primary/5")}>
      <span className="text-xs text-muted-foreground">{label}</span>
      <strong className="mt-1 block text-xl tabular-nums">{value}{suffix}</strong>
    </div>
  );
}

function PredictionSummary({ run }: { run: ReviewRun }) {
  const prediction = run.prediction;
  if (!prediction) return null;
  const horizons = [
    ["h1", "上市首日", prediction.horizon_predictions?.h1, false],
    ["h5", "5个交易日", prediction.horizon_predictions?.h5 ?? prediction.probability, true],
    ["h20", "20个交易日", prediction.horizon_predictions?.h20, false],
    ["h60", "60个交易日", prediction.horizon_predictions?.h60, false],
  ] as const;
  const horizonStatus = (key: "h1" | "h5" | "h20" | "h60", value: number | null | undefined) => {
    if (value == null) return "暂不可用";
    const alert = prediction.horizon_alerts?.[key];
    if (alert != null) return alert ? "触发预警" : "未触发预警";
    const threshold = prediction.horizon_thresholds?.[key];
    if (threshold != null) return value >= threshold ? "触发预警" : "未触发预警";
    if (key === "h5") return prediction.predicted_significant_drop_5d ? "触发预警" : "未触发预警";
    return "暂不可用";
  };
  const decisionLabel = prediction.decision === "high_risk"
    ? "高风险预警"
    : prediction.recall_guard?.watchlist
      ? "风险观察名单"
    : prediction.decision === "low_risk"
      ? "低风险"
      : "未触发显著下跌预警";
  return (
    <Card className="border-primary/25 bg-primary/5">
      <CardHeader className="border-b border-border/60">
        <CardTitle>上市后多时间窗口风险预测</CardTitle>
        <CardDescription>四个模型独立判断不同时间窗口；5个交易日为核心预警窗口。</CardDescription>
        <CardAction>
          <Badge variant={prediction.decision === "high_risk" ? "destructive" : prediction.decision === "low_risk" ? "secondary" : "outline"}>
            {decisionLabel}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {horizons.map(([key, label, value, emphasized]) => (
          <SummaryCard
            key={key}
            label={label}
            value={horizonStatus(key, value)}
            emphasized={emphasized}
          />
        ))}
        {prediction.recall_guard?.watchlist && (
          <Alert className="sm:col-span-2 xl:col-span-4">
            <ShieldAlert />
            <AlertTitle>高风险召回保护层已命中</AlertTitle>
            <AlertDescription>
              {prediction.recall_guard.reasons.join("、") || "多项稳定风险因子叠加"}。该标记用于减少漏报，不会覆盖主模型结论。
            </AlertDescription>
          </Alert>
        )}
        <p className="text-xs text-muted-foreground sm:col-span-2 xl:col-span-4">{prediction.disclaimer}</p>
      </CardContent>
    </Card>
  );
}

function TextEmbellishmentSummary({ assessment }: { assessment: NonNullable<AuditMaterials["text_embellishment"]> }) {
  if (assessment.evidence_ids.length === 0) return null;
  const levelLabel = { low: "低", medium: "中", high: "高", critical: "严重" }[assessment.level];
  return (
    <Card className="border-border/70 bg-card/75">
      <CardHeader className="border-b border-border/60">
        <CardTitle>招股书文本粉饰分析</CardTitle>
        <CardDescription>对照业务概览、风险因素与财务披露，识别宣传性表述和选择性披露。</CardDescription>
        <CardAction>
          <Badge variant={assessment.detected ? "destructive" : "secondary"}>粉饰程度：{levelLabel}</Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
        <p className="text-sm leading-6 text-muted-foreground">{assessment.summary}</p>
        <div className="flex flex-wrap gap-2 text-xs">
          <Badge variant="outline">原文证据 {assessment.evidence_ids.length} 条</Badge>
          {assessment.debate_used && <Badge variant="outline">已完成补证辩论</Badge>}
        </div>
      </CardContent>
    </Card>
  );
}

function AgentModelRoutingSummary({ assignments }: { assignments: Record<string, string> }) {
  const roles = {
    discovery: "风险发现 Agent", finance: "财务 Agent", legal: "法律 Agent",
    business: "业务 Agent", market: "市场 Agent", decision: "决策 Agent",
  } as const;
  const models: Record<string, string> = {
    deepseek_flash: "DeepSeek Flash", deepseek_pro: "DeepSeek Pro", kimi: "Kimi K3",
  };
  return (
    <Card className="border-border/70 bg-card/75">
      <CardHeader className="border-b border-border/60">
        <CardTitle>Agent模型编排</CardTitle>
        <CardDescription>本次分析启动前已固化模型分配，重试与补证均沿用原模型。</CardDescription>
        <CardAction><Badge variant="secondary">运行中不可切换</Badge></CardAction>
      </CardHeader>
      <CardContent className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {Object.entries(roles).map(([role, label]) => (
          <div key={role} className="flex items-center justify-between gap-3 rounded-xl border bg-background/35 px-3 py-2.5">
            <span className="text-sm font-medium">{label}</span>
            <Badge variant="outline">{models[assignments[role]] ?? assignments[role] ?? "历史记录未保存"}</Badge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export function ReportsWorkspace({ active, serviceAvailable, initialRunId, initialFindingId, onReconnect, onSelectionChange, setError }: ReportsWorkspaceProps) {
  const [runs, setRuns] = useState<ReviewRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [loadedRunId, setLoadedRunId] = useState<string | null>(null);
  const [findings, setFindings] = useState<StoredFinding[]>([]);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [detail, setDetail] = useState<FindingDetail | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingFindings, setLoadingFindings] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [exportingKind, setExportingKind] = useState<"pdf" | "audit_pdf" | "audit" | "prediction_csv" | "trace_json" | null>(null);
  const [reviewJustCompleted, setReviewJustCompleted] = useState(false);
  const [query, setQuery] = useState("");
  const [reviewer, setReviewer] = useState("张明");
  const [comment, setComment] = useState("无");
  const [editTitle, setEditTitle] = useState("");
  const [editClaim, setEditClaim] = useState("");
  const [editSeverity, setEditSeverity] = useState("medium");
  const [editConfidence, setEditConfidence] = useState(0.5);
  const [findingFilter, setFindingFilter] = useState<FindingFilter>("all");
  const [retryJobs, setRetryJobs] = useState<RetryJob[]>([]);
  const [retrySubmitting, setRetrySubmitting] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deletingRun, setDeletingRun] = useState(false);
  const [tableEvidenceReviews, setTableEvidenceReviews] = useState<Record<string, TableEvidenceReview>>({});
  const [auditMaterials, setAuditMaterials] = useState<AuditMaterials | null>(null);
  const [auditView, setAuditView] = useState<"toolchain" | "evidence">("toolchain");
  const [auditQuery, setAuditQuery] = useState("");
  const [loadingAuditMaterials, setLoadingAuditMaterials] = useState(false);
  const requestedRunIdRef = useRef(initialRunId);
  const requestedFindingIdRef = useRef(initialFindingId);
  const selectedRunIdRef = useRef<string | null>(null);
  const selectedFindingIdRef = useRef<string | null>(null);

  useEffect(() => {
    selectedRunIdRef.current = selectedRunId;
    selectedFindingIdRef.current = selectedFindingId;
  }, [selectedFindingId, selectedRunId]);

  useEffect(() => {
    requestedRunIdRef.current = initialRunId;
    requestedFindingIdRef.current = initialFindingId;
    if (!initialRunId && !initialFindingId) return;
    if (initialRunId && runs.length > 0 && runs.some((run) => run.run_id === initialRunId)) {
      setSelectedRunId(initialRunId);
    }
    if (initialFindingId && findings.length > 0 && findings.some((finding) => finding.finding_id === initialFindingId)) {
      setSelectedFindingId(initialFindingId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only react to external navigation, not internal state
  }, [initialFindingId, initialRunId]);

  const refreshRuns = useCallback(async () => {
    const rows = await listReviewRuns();
    setRuns(rows);
    setSelectedRunId((current) => current && rows.some((row) => row.run_id === current)
      ? current
      : requestedRunIdRef.current && rows.some((row) => row.run_id === requestedRunIdRef.current)
        ? requestedRunIdRef.current
        : rows[0]?.run_id ?? null);
  }, []);

  useEffect(() => {
    if (!active) return;
    setLoadingRuns(true);
    void refreshRuns()
      .catch((error) => setError(error instanceof Error ? error.message : "分析记录加载失败。"))
      .finally(() => setLoadingRuns(false));
  }, [active, refreshRuns, setError]);

  useEffect(() => {
    if (!selectedRunId) {
      setFindings([]);
      setLoadedRunId(null);
      setSelectedFindingId(null);
      setDetail(null);
      return;
    }
    let active = true;
    setLoadingFindings(true);
    setFindings([]);
    setLoadedRunId(null);
    setSelectedFindingId(null);
    setDetail(null);
    setDetailError(null);
    setError(null);
    void listRunFindings(selectedRunId)
      .then((rows) => {
        if (!active) return;
        setFindings(rows);
        setLoadedRunId(selectedRunId);
        const requested = requestedFindingIdRef.current
          ? rows.find((row) => row.finding_id === requestedFindingIdRef.current)
          : null;
        const pending = rows.find((row) => !["verified", "rejected"].includes(row.status));
        setSelectedFindingId(requested?.finding_id ?? pending?.finding_id ?? rows[0]?.finding_id ?? null);
      })
      .catch((error) => active && setError(error instanceof Error ? error.message : "风险结论加载失败。"))
      .finally(() => active && setLoadingFindings(false));
    return () => { active = false; };
  }, [selectedRunId, setError]);

  useEffect(() => {
    if (!selectedRunId) {
      setAuditMaterials(null);
      return;
    }
    setReviewJustCompleted(false);
    let mounted = true;
    setLoadingAuditMaterials(true);
    setAuditMaterials(null);
    void getAuditMaterials(selectedRunId)
      .then((payload) => mounted && setAuditMaterials(payload))
      .catch((error) => mounted && setError(error instanceof Error ? error.message : "审计材料加载失败。"))
      .finally(() => mounted && setLoadingAuditMaterials(false));
    return () => { mounted = false; };
  }, [selectedRunId, setError]);

  useEffect(() => {
    onSelectionChange?.(selectedRunId, selectedFindingId);
  }, [onSelectionChange, selectedFindingId, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || loadedRunId !== selectedRunId || !selectedFindingId) {
      setDetail(null);
      return;
    }
    let active = true;
    setLoadingDetail(true);
    setDetail(null);
    setDetailError(null);
    void getFindingDetail(selectedRunId, selectedFindingId)
      .then((payload) => {
        if (!active) return;
        setDetail(payload);
        setEditTitle(payload.finding.title);
        setEditClaim(payload.finding.claim);
        setEditSeverity(payload.finding.severity);
        setEditConfidence(payload.finding.confidence);
      })
      .catch((error) => {
        if (!active) return;
        const message = error instanceof Error ? error.message : "风险详情加载失败。";
        setDetailError(message);
        setError(message);
      })
      .finally(() => active && setLoadingDetail(false));
    return () => { active = false; };
  }, [loadedRunId, selectedFindingId, selectedRunId, setError]);

  const reloadSelectedDetail = async () => {
    if (!selectedRunId || !selectedFindingId) return;
    setLoadingDetail(true);
    setDetailError(null);
    setError(null);
    try {
      const payload = await getFindingDetail(selectedRunId, selectedFindingId);
      setDetail(payload);
      setEditTitle(payload.finding.title);
      setEditClaim(payload.finding.claim);
      setEditSeverity(payload.finding.severity);
      setEditConfidence(payload.finding.confidence);
    } catch (error) {
      const message = error instanceof Error ? error.message : "风险详情加载失败。";
      setDetailError(message);
      setError(message);
    } finally {
      setLoadingDetail(false);
    }
  };

  const synchronizeRetrySnapshot = useCallback(async (runId: string, jobs: RetryJob[]) => {
    const hasActiveJob = jobs.some((job) => ["queued", "running"].includes(job.state));
    if (hasActiveJob) {
      if (selectedRunIdRef.current === runId) setRetryJobs(jobs);
      return;
    }

    // Keep the previous active snapshot visible until the refreshed business data is ready.
    // Otherwise retryActive becomes false first and React cleans up this effect mid-refresh.
    const rows = await listRunFindings(runId);
    if (selectedRunIdRef.current !== runId) return;

    const findingId = selectedFindingIdRef.current;
    let payload: FindingDetail | null = null;
    if (findingId && rows.some((row) => row.finding_id === findingId)) {
      try {
        payload = await getFindingDetail(runId, findingId);
      } catch (error) {
        if (selectedRunIdRef.current === runId && selectedFindingIdRef.current === findingId) {
          const message = error instanceof Error ? error.message : "重试完成，但风险详情刷新失败。";
          setDetailError(message);
          setError(message);
        }
      }
      if (selectedRunIdRef.current !== runId || selectedFindingIdRef.current !== findingId) return;
    }

    setFindings(rows);
    setLoadedRunId(runId);
    if (payload) {
      setDetail(payload);
      setEditTitle(payload.finding.title);
      setEditClaim(payload.finding.claim);
      setEditSeverity(payload.finding.severity);
      setEditConfidence(payload.finding.confidence);
    }
    try {
      await refreshRuns();
    } catch (error) {
      setError(error instanceof Error ? error.message : "重试完成，但分析记录刷新失败。");
    }
    if (selectedRunIdRef.current === runId) setRetryJobs(jobs);
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedRunId) {
      setRetryJobs([]);
      return;
    }
    let active = true;
    void listRetryJobs(selectedRunId)
      .then((jobs) => active && setRetryJobs(jobs))
      .catch((error) => active && setError(error instanceof Error ? error.message : "重试状态加载失败。"));
    return () => { active = false; };
  }, [selectedRunId, setError]);

  const retryActive = retryJobs.some((job) => ["queued", "running"].includes(job.state));
  useEffect(() => {
    if (!retryActive || !selectedRunId) return;
    let active = true;
    let polling = false;
    const timer = window.setInterval(() => {
      if (polling) return;
      polling = true;
      void listRetryJobs(selectedRunId)
        .then(async (jobs) => {
          if (!active) return;
          await synchronizeRetrySnapshot(selectedRunId, jobs);
        })
        .catch((error) => active && setError(error instanceof Error ? error.message : "重试状态刷新失败。"))
        .finally(() => { polling = false; });
    }, 1_200);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [retryActive, selectedRunId, setError, synchronizeRetrySnapshot]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );
  const filteredRuns = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return runs.filter((run) => !normalized || `${run.company_id} ${run.company_name} ${run.run_id}`.toLowerCase().includes(normalized));
  }, [query, runs]);
  const findingCounts = useMemo(() => ({
    all: findings.length,
    pending: findings.filter((finding) => !["verified", "rejected"].includes(finding.status)).length,
    verified: findings.filter((finding) => finding.status === "verified").length,
    rejected: findings.filter((finding) => finding.status === "rejected").length,
  }), [findings]);
  const filteredFindings = useMemo(() => findings.filter((finding) => {
    if (findingFilter === "all") return true;
    if (findingFilter === "pending") return !["verified", "rejected"].includes(finding.status);
    return finding.status === findingFilter;
  }), [findingFilter, findings]);
  const filteredAuditEvidence = useMemo(() => {
    const normalized = auditQuery.trim().toLowerCase();
    return (auditMaterials?.evidences ?? []).filter((evidence) => !normalized || [
      evidence.evidence_id,
      evidence.quote,
      evidence.evidence_type,
      String(evidence.page),
      ...evidence.findings.flatMap((finding) => [finding.title, finding.risk_type]),
    ].join(" ").toLowerCase().includes(normalized));
  }, [auditMaterials, auditQuery]);

  const selectRun = useCallback((runId: string) => {
    if (runId === selectedRunId) return;
    startTransition(() => {
      setError(null);
      setSelectedRunId(runId);
    });
  }, [selectedRunId, setError]);

  const selectFindingFilter = (value: string[]) => {
    const filter = (value[0] ?? "all") as FindingFilter;
    setFindingFilter(filter);
    const candidates = findings.filter((finding) => {
      if (filter === "all") return true;
      if (filter === "pending") return !["verified", "rejected"].includes(finding.status);
      return finding.status === filter;
    });
    if (!candidates.some((finding) => finding.finding_id === selectedFindingId)) {
      setSelectedFindingId(candidates[0]?.finding_id ?? null);
    }
  };

  const submitDecision = async (decision: "confirmed" | "rejected" | "edited") => {
    if (!selectedRunId || !selectedFindingId) return;
    if (!serviceAvailable) return setError("分析服务当前未连接，重新连接后才能提交复核。");
    if (!reviewer.trim()) return setError("请填写复核人姓名或工号。");
    const tableReviewLines = detail?.evidences.flatMap((evidence) => {
      const status = tableEvidenceReviews[evidence.evidence_id];
      return status ? [`[表格核对] ${evidence.evidence_id}：${tableEvidenceReviewLabel[status]}`] : [];
    }) ?? [];
    const hasTableIssue = detail?.evidences.some((evidence) => {
      const status = tableEvidenceReviews[evidence.evidence_id];
      return status !== undefined && status !== "matched";
    }) ?? false;
    const manualOverride = decision !== "rejected" && detail?.verification.passed === false;
    const overrideNote = manualOverride
      ? "[人工确认] 自动一致性校验存在待核项，复核人已查看原文证据并确认该风险结论。"
      : "";
    const finalComment = [comment.trim(), overrideNote, ...tableReviewLines].filter(Boolean).join("\n");
    if (decision !== "confirmed" && !comment.trim() && !hasTableIssue) {
      return setError("驳回或编辑结论时必须填写复核意见，或标记具体的表格问题。");
    }
    setSubmitting(true);
    setError(null);
    try {
      await reviewFinding(selectedRunId, selectedFindingId, {
        decision,
        reviewer: reviewer.trim(),
        comment: finalComment || undefined,
        manual_override: manualOverride,
        ...(decision === "edited" ? {
          title: editTitle.trim(),
          claim: editClaim.trim(),
          severity: editSeverity,
          confidence: editConfidence,
        } : {}),
      });
      await refreshRuns();
      const rows = await listRunFindings(selectedRunId);
      setFindings(rows);
      const next = rows.find((row) => !["verified", "rejected"].includes(row.status));
      setReviewJustCompleted(!next && rows.length > 0);
      if (!next && rows.length > 0) setFindingFilter("all");
      const nextId = next?.finding_id ?? selectedFindingId;
      setSelectedFindingId(nextId);
      setComment("无");
      setTableEvidenceReviews((current) => Object.fromEntries(
        Object.entries(current).filter(([evidenceId]) => !detail?.evidences.some((evidence) => evidence.evidence_id === evidenceId)),
      ));
      const nextDetail = await getFindingDetail(selectedRunId, nextId);
      setDetail(nextDetail);
      setEditTitle(nextDetail.finding.title);
      setEditClaim(nextDetail.finding.claim);
      setEditSeverity(nextDetail.finding.severity);
      setEditConfidence(nextDetail.finding.confidence);
    } catch (error) {
      setError(error instanceof Error ? error.message : "复核提交失败。");
    } finally {
      setSubmitting(false);
    }
  };

  const handleExport = async (kind: "pdf" | "audit_pdf" | "audit" | "prediction_csv" | "trace_json") => {
    if (!selectedRunId) return;
    if (!serviceAvailable) return setError("分析服务当前未连接，重新连接后才能导出报告。");
    setExportingKind(kind);
    setError(null);
    try {
      const files = await exportRun(selectedRunId);
      if (files[kind]) downloadFromUrl(files[kind]);
      await refreshRuns();
    } catch (error) {
      setError(error instanceof Error ? error.message : "正式报告导出失败。");
    } finally {
      setExportingKind(null);
    }
  };

  const handleDeleteRun = async () => {
    if (!selectedRunId) return;
    if (!serviceAvailable) return setError("分析服务当前未连接，重新连接后才能删除记录。");
    if (retryActive) return setError("该分析正在重新处理，请等待任务结束后再删除。");
    const deletedRunId = selectedRunId;
    setDeletingRun(true);
    setError(null);
    try {
      await deleteAnalysisRun(deletedRunId);
      requestedRunIdRef.current = null;
      requestedFindingIdRef.current = null;
      setRuns((current) => current.filter((run) => run.run_id !== deletedRunId));
      setSelectedRunId(null);
      setLoadedRunId(null);
      setFindings([]);
      setSelectedFindingId(null);
      setDetail(null);
      setRetryJobs([]);
      setDeleteDialogOpen(false);
      await refreshRuns();
    } catch (error) {
      setError(error instanceof Error ? error.message : "分析记录删除失败。");
    } finally {
      setDeletingRun(false);
    }
  };

  const startRetry = async (scope: "finding" | "agent" | "run") => {
    if (!selectedRunId || !detail) return;
    if (!serviceAvailable) return setError("分析服务当前未连接，重新连接后才能重新分析。");
    setRetrySubmitting(true);
    setError(null);
    try {
      const job = scope === "finding"
        ? await retryFinding(selectedRunId, detail.finding.finding_id)
        : scope === "agent"
          ? await retryAgentFailures(selectedRunId, detail.finding.created_by)
          : await retryRunFailures(selectedRunId);
      const initialJobs = [job, ...retryJobs.filter((item) => item.job_id !== job.job_id)];
      if (["queued", "running"].includes(job.state)) setRetryJobs(initialJobs);
      const latestJobs = await listRetryJobs(selectedRunId);
      await synchronizeRetrySnapshot(selectedRunId, latestJobs.length ? latestJobs : initialJobs);
    } catch (error) {
      setError(error instanceof Error ? error.message : "重新分析启动失败。");
    } finally {
      setRetrySubmitting(false);
    }
  };

  const technicalFailureCount = findings.filter(hasTechnicalFailure).length;
  const retryItemsByFinding = useMemo(() => latestRetryItemsByFinding(retryJobs), [retryJobs]);
  const selectedRetryItem = selectedFindingId ? retryItemsByFinding.get(selectedFindingId) : undefined;
  const activeRetryItem = selectedRetryItem && ["queued", "running"].includes(selectedRetryItem.state)
    ? selectedRetryItem
    : undefined;
  const findingConfirmed = detail?.finding.status === "verified";
  const findingRejected = detail?.finding.status === "rejected";
  const reviewComplete = reviewJustCompleted || Boolean(selectedRun?.complete);

  return (
    <div className="space-y-5">
      <section className="flex flex-col gap-4 rounded-2xl border bg-card/65 p-5 shadow-sm backdrop-blur lg:flex-row lg:items-end lg:justify-between">
        <div>
          <Badge variant="outline" className="mb-3"><Database data-icon="inline-start" />分析与复核一体化</Badge>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">分析结果中心</h1>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">查看风险、核验证据、完成人工复核并导出正式报告，避免在两个重复页面之间切换。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">{runs.length} 次分析运行</Badge>
          <Button
            variant="outline"
            disabled={!serviceAvailable || loadingRuns}
            onClick={() => {
              setLoadingRuns(true);
              void refreshRuns()
                .catch((error) => setError(error instanceof Error ? error.message : "分析记录加载失败。"))
                .finally(() => setLoadingRuns(false));
            }}
          >
            {loadingRuns ? <Spinner data-icon="inline-start" /> : <RefreshCw data-icon="inline-start" />}
            刷新结果
          </Button>
          <Button
            variant="outline"
            disabled={!serviceAvailable || !selectedRun || deletingRun || retryActive}
            onClick={() => setDeleteDialogOpen(true)}
          >
            <Trash2 data-icon="inline-start" />
            删除当前记录
          </Button>
          {technicalFailureCount > 0 && (
            <Button variant="outline" disabled={!serviceAvailable || retrySubmitting || retryActive} onClick={() => void startRetry("run")}>
              {retrySubmitting || retryActive ? <Spinner data-icon="inline-start" /> : <RefreshCw data-icon="inline-start" />}
              重试本公司 {technicalFailureCount} 条技术失败
            </Button>
          )}
        </div>
      </section>

      {selectedRun && <PredictionSummary run={selectedRun} />}
      {auditMaterials?.text_embellishment && (
        <TextEmbellishmentSummary assessment={auditMaterials.text_embellishment} />
      )}
      {auditMaterials?.agent_models && Object.keys(auditMaterials.agent_models).length > 0 && (
        <AgentModelRoutingSummary assignments={auditMaterials.agent_models} />
      )}
      {selectedRun && !loadingAuditMaterials && (
        <DebateTimeline records={auditMaterials?.debates ?? []} />
      )}

      <div className="grid min-w-0 gap-4 2xl:grid-cols-[280px_390px_minmax(520px,1fr)]">
        <Card className="min-w-0 bg-card/75">
          <CardHeader className="border-b border-border/60">
            <CardTitle>分析记录</CardTitle>
            <CardDescription>按公司或运行编号查找。</CardDescription>
            <CardAction><Badge variant="outline">{filteredRuns.length}</Badge></CardAction>
          </CardHeader>
          <CardContent>
            <div className="relative mb-3">
              <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input className="pl-9" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索公司或运行编号" />
            </div>
            {filteredRuns.length ? (
              <VirtualList
                items={filteredRuns}
                estimateSize={112}
                getKey={(run) => run.run_id}
                renderItem={(run) => <RunItem run={run} selected={run.run_id === selectedRunId} onSelect={selectRun} />}
                className="h-[min(660px,calc(100vh-285px))] min-h-72"
              />
            ) : (
              <Empty className="min-h-64 border">
                <EmptyHeader>
                  <EmptyMedia variant="icon">{loadingRuns ? <Spinner /> : <FileSearch />}</EmptyMedia>
                  <EmptyTitle>{loadingRuns ? "正在加载" : "没有匹配记录"}</EmptyTitle>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 bg-card/75">
          <CardHeader className="border-b border-border/60">
            <CardTitle className="truncate">{selectedRun ? `${selectedRun.company_id} · ${selectedRun.company_name}` : "风险清单"}</CardTitle>
            <CardDescription>筛选并选择要核验的风险。</CardDescription>
            {selectedRun && <CardAction><Badge variant="secondary">{selectedRun.reviewed_findings}/{selectedRun.total_findings}</Badge></CardAction>}
          </CardHeader>
          <CardContent>
            {selectedRun && (
              <div className="mb-3 grid grid-cols-4 gap-2">
                <SummaryCard label="总数" value={selectedRun.total_findings} />
                <SummaryCard label="确认" value={selectedRun.verified_findings} />
                <SummaryCard label="驳回" value={selectedRun.rejected_findings} />
                <SummaryCard label="待处理" value={selectedRun.pending_findings} />
              </div>
            )}
            <ToggleGroup value={[findingFilter]} onValueChange={selectFindingFilter} className="mb-3 grid grid-cols-4">
              {([['all', '全部'], ['pending', '待复核'], ['verified', '已确认'], ['rejected', '已驳回']] as const).map(([value, label]) => (
                <ToggleGroupItem key={value} value={value}>{label} {findingCounts[value]}</ToggleGroupItem>
              ))}
            </ToggleGroup>
            {loadingFindings ? (
              <div className="space-y-3">{Array.from({ length: 5 }, (_, index) => <Skeleton className="h-20 w-full" key={index} />)}</div>
            ) : filteredFindings.length ? (
              <VirtualList
                items={filteredFindings}
                estimateSize={104}
                getKey={(finding) => finding.finding_id}
                renderItem={(finding, index) => (
                  <FindingItem
                    finding={finding}
                    index={index}
                    retry={retryItemsByFinding.get(finding.finding_id)}
                    selected={finding.finding_id === selectedFindingId}
                    onSelect={setSelectedFindingId}
                  />
                )}
                className="h-[min(590px,calc(100vh-360px))] min-h-64"
              />
            ) : (
              <Empty className="min-h-64 border">
                <EmptyHeader>
                  <EmptyMedia variant="icon"><FileCheck2 /></EmptyMedia>
                  <EmptyTitle>当前筛选没有风险</EmptyTitle>
                  <EmptyDescription>切换状态筛选或选择其他运行记录。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 bg-card/75">
          <CardHeader className="border-b border-border/60">
            <CardTitle>证据与人工复核</CardTitle>
            <CardDescription>原文、表格、截图、核验结果和审计记录集中展示。</CardDescription>
            {detail && <CardAction><Badge variant={detail.verification.passed ? "secondary" : "destructive"}>{detail.verification.passed ? "核验通过" : "存在待核项"}</Badge></CardAction>}
          </CardHeader>
          <CardContent className="p-0">
            {loadingDetail ? (
              <div className="space-y-4 p-5"><Skeleton className="h-8 w-2/3" /><Skeleton className="h-24 w-full" /><Skeleton className="h-72 w-full" /></div>
            ) : detailError && selectedFindingId ? (
              <Empty className="min-h-[520px] border-0">
                <EmptyHeader>
                  <EmptyMedia variant="icon"><WifiOff /></EmptyMedia>
                  <EmptyTitle>风险详情暂时无法加载</EmptyTitle>
                  <EmptyDescription>{detailError}</EmptyDescription>
                </EmptyHeader>
                <EmptyContent className="flex-row flex-wrap justify-center">
                  {!serviceAvailable && (
                    <Button variant="outline" onClick={() => void onReconnect()}>
                      <RefreshCw data-icon="inline-start" />重新连接服务
                    </Button>
                  )}
                  <Button disabled={!serviceAvailable} onClick={() => void reloadSelectedDetail()}>
                    <RefreshCw data-icon="inline-start" />重新加载详情
                  </Button>
                </EmptyContent>
              </Empty>
            ) : detail && selectedRunId ? (
              <ScrollArea className="h-[min(700px,calc(100vh-250px))] min-h-[520px]">
                <div className="space-y-5 p-5">
                  <section>
                    <div className="mb-3 flex flex-wrap gap-2">
                      <Badge variant={severityVariant(detail.finding.severity)}>{severityLabel[detail.finding.severity]}</Badge>
                      <Badge variant="outline">置信度 {Math.round(detail.finding.confidence * 100)}%</Badge>
                      <Badge variant="outline">{detail.finding.created_by}</Badge>
                    </div>
                    <h2 className="text-xl font-semibold leading-tight">{detail.finding.title}</h2>
                    <p className="mt-3 text-sm leading-7 text-foreground/90">{detail.finding.claim}</p>
                    {/* 证据边界区块已移除 */}
                  </section>
                  <Separator />
                  <section className="space-y-4">
                    <div className="flex items-center justify-between"><h3 className="font-semibold">原文证据</h3><Badge variant="outline">{detail.evidences.length} 条</Badge></div>
                    {detail.evidences.map((evidence, index) => (
                      <Card key={evidence.evidence_id} size="sm" className="overflow-hidden bg-muted/10">
                        <CardHeader className="border-b border-border/50">
                          <CardTitle>证据 {index + 1}</CardTitle>
                          <CardDescription>物理页 {evidence.page}{evidence.printed_page_label ? ` · 印刷页 ${evidence.printed_page_label}` : ""}</CardDescription>
                          <CardAction><Badge variant="outline">{Math.round(evidence.confidence * 100)}%</Badge></CardAction>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {evidence.display_table || evidence.table_id || evidence.table_cell_refs.length > 0 || evidence.evidence_type === "table" ? (
                            <TableEvidenceViewer
                              evidence={evidence}
                              imageUrl={evidence.screenshot_path ? evidenceImageUrl(selectedRunId, evidence.evidence_id) : undefined}
                              reviewStatus={tableEvidenceReviews[evidence.evidence_id]}
                              onReviewStatusChange={(status) => setTableEvidenceReviews((current) => {
                                if (!status) {
                                  const next = { ...current };
                                  delete next[evidence.evidence_id];
                                  return next;
                                }
                                return { ...current, [evidence.evidence_id]: status };
                              })}
                            />
                          ) : (
                            <>
                              <blockquote className="border-l-2 border-primary/60 pl-4 text-sm leading-7 text-foreground/90">{evidence.quote}</blockquote>
                              {evidence.screenshot_path && (
                                <img className="max-h-[520px] w-full rounded-xl border bg-white object-contain" src={evidenceImageUrl(selectedRunId, evidence.evidence_id)} alt={`证据 ${evidence.evidence_id} 高亮截图`} loading="lazy" />
                              )}
                            </>
                          )}
                          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                            <span>{evidence.evidence_id}</span><span>坐标 {evidence.bbox.map((value) => Math.round(value)).join(", ")}</span>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </section>
                  <Separator />
                  <section className="space-y-3">
                    <h3 className="font-semibold">确定性核验</h3>
                    {detail.verification.checks.map((check) => (
                      <div className="flex gap-3 rounded-xl border p-3" key={check.name}>
                        {check.passed ? <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-400" /> : <XCircle className="mt-0.5 size-4 shrink-0 text-destructive" />}
                        <div><strong className="text-sm">{check.name}</strong><p className="mt-1 text-xs leading-5 text-muted-foreground">{check.detail}</p></div>
                      </div>
                    ))}
                  </section>
                  <Separator />
                  <section className="space-y-4">
                    <div className="flex items-center justify-between"><h3 className="font-semibold">人工复核</h3><Badge variant="outline">{findingStatusLabel[detail.finding.status] ?? detail.finding.status}</Badge></div>
                    {reviewComplete && (
                      <div className="flex gap-3 rounded-xl border border-emerald-500/25 bg-emerald-500/[0.06] p-4">
                        <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-emerald-500" />
                        <div>
                          <p className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">本次复核已完成</p>
                          <p className="mt-1 text-xs leading-5 text-muted-foreground">所有风险均已确认或驳回，可以导出正式报告；已复核结论仍可通过“保存编辑”继续修改。</p>
                        </div>
                      </div>
                    )}
                    <FieldGroup>
                      <Field><FieldLabel>复核人</FieldLabel><Input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="姓名或工号" /></Field>
                      <Field><FieldLabel>复核意见</FieldLabel><Textarea value={comment} onChange={(event) => setComment(event.target.value)} placeholder="驳回或编辑时必填" /></Field>
                      <details className="rounded-xl border p-4">
                        <summary className="cursor-pointer text-sm font-medium">编辑风险结论</summary>
                        <div className="mt-4 space-y-3">
                          <Field><FieldLabel>标题</FieldLabel><Input value={editTitle} onChange={(event) => setEditTitle(event.target.value)} /></Field>
                          <div className="grid gap-3 sm:grid-cols-2">
                            <Field><FieldLabel>严重程度</FieldLabel><Select value={editSeverity} onValueChange={(value) => setEditSeverity(value ?? "medium")}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectGroup>{[["low", "低风险"], ["medium", "中风险"], ["high", "高风险"], ["critical", "极高风险"]].map(([value, label]) => <SelectItem key={value} value={value}>{label}</SelectItem>)}</SelectGroup></SelectContent></Select></Field>
                            <Field><FieldLabel>置信度 {Math.round(editConfidence * 100)}%</FieldLabel><Input type="range" min="0" max="1" step="0.01" value={editConfidence} onChange={(event) => setEditConfidence(Number(event.target.value))} /></Field>
                          </div>
                          <Field><FieldLabel>风险结论</FieldLabel><Textarea className="min-h-28" value={editClaim} onChange={(event) => setEditClaim(event.target.value)} /></Field>
                        </div>
                      </details>
                    </FieldGroup>
                    <div className="grid gap-2 sm:grid-cols-3">
                      <Button disabled={!serviceAvailable || submitting || Boolean(activeRetryItem) || findingConfirmed} title={!detail.verification.passed ? "将作为人工确认写入审计记录" : undefined} onClick={() => void submitDecision("confirmed")}>{submitting ? <Spinner data-icon="inline-start" /> : <CheckCircle2 data-icon="inline-start" />}{findingConfirmed ? "已确认" : detail.verification.passed ? "确认" : "人工确认"}</Button>
                      <Button variant="destructive" disabled={!serviceAvailable || submitting || Boolean(activeRetryItem) || findingRejected} onClick={() => void submitDecision("rejected")}><XCircle data-icon="inline-start" />{findingRejected ? "已驳回" : "驳回"}</Button>
                      <Button variant="outline" disabled={!serviceAvailable || submitting || Boolean(activeRetryItem)} onClick={() => void submitDecision("edited")}><FileCheck2 data-icon="inline-start" />保存编辑</Button>
                    </div>
                  </section>
                  {detail.reviews.length > 0 && (
                    <section className="space-y-3">
                      <Separator />
                      <h3 className="flex items-center gap-2 font-semibold"><History className="size-4 text-primary" />审计记录</h3>
                      {detail.reviews.map((review) => (
                        <div className="flex gap-3 rounded-xl border p-3" key={review.review_id}>
                          <UserCheck className="mt-0.5 size-4 shrink-0 text-primary" />
                          <div><strong className="text-sm">{review.reviewer} · {review.decision}{review.manual_override ? " · 人工覆盖" : ""}</strong><p className="mt-1 text-xs text-muted-foreground">{review.comment || "未填写补充意见"} · {formatTime(review.created_at)}</p></div>
                        </div>
                      ))}
                    </section>
                  )}
                  {detail.agent_traces.some((trace) => trace.trace_id.includes("-retry-") && trace.input_refs.includes(detail.finding.finding_id)) && (
                    <section className="space-y-3">
                      <Separator />
                      <h3 className="flex items-center gap-2 font-semibold"><RefreshCw className="size-4 text-primary" />模型重试审计</h3>
                      {detail.agent_traces.filter((trace) => trace.trace_id.includes("-retry-") && trace.input_refs.includes(detail.finding.finding_id)).map((trace) => (
                        <div className="rounded-xl border p-3" key={trace.trace_id}>
                          <div className="flex items-center justify-between gap-2"><strong className="text-sm">{trace.agent} Agent</strong><Badge variant={trace.status === "succeeded" ? "secondary" : "destructive"}>{trace.status === "succeeded" ? "成功" : "失败"}</Badge></div>
                          <p className="mt-1 text-xs leading-5 text-muted-foreground">{trace.decision} · {formatTime(trace.ended_at)}</p>
                        </div>
                      ))}
                    </section>
                  )}
                </div>
              </ScrollArea>
            ) : (
              <Empty className="min-h-[520px]">
                <EmptyHeader>
                  <EmptyMedia variant="icon"><FileSearch /></EmptyMedia>
                  <EmptyTitle>选择一条风险查看证据</EmptyTitle>
                  <EmptyDescription>原文、表格、截图和确定性核验将在这里展示。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </CardContent>
        </Card>
      </div>

      {selectedRun && (
        <Card className="bg-card/75">
          <CardHeader className="border-b border-border/60">
            <CardTitle>审计材料中心</CardTitle>
            <CardDescription>内容直接读取数据库；JSON仅用于下载交付。</CardDescription>
            <CardAction>
              <div className="flex rounded-lg border p-1">
                <Button size="sm" variant={auditView === "toolchain" ? "secondary" : "ghost"} onClick={() => setAuditView("toolchain")}>推理与工具链</Button>
                <Button size="sm" variant={auditView === "evidence" ? "secondary" : "ghost"} onClick={() => setAuditView("evidence")}>关键证据库</Button>
              </div>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-4">
            {loadingAuditMaterials ? (
              <div className="space-y-3"><Skeleton className="h-20 w-full" /><Skeleton className="h-20 w-full" /></div>
            ) : auditView === "toolchain" ? (
              <div className="space-y-3">
                {(auditMaterials?.business_toolchain ?? []).map((stage, index) => (
                  <div key={`${stage.stage}-${index}`} className="rounded-xl border bg-background/40 p-4">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="flex size-6 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">{index + 1}</span>
                      <strong className="text-sm">{stage.stage}</strong>
                      <Badge variant="secondary">已完成</Badge>
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">{stage.summary}</p>
                    <div className="mt-2 flex flex-wrap gap-1.5">{stage.tools.map((tool) => <Badge key={tool} variant="outline" className="font-normal">{tool}</Badge>)}</div>
                  </div>
                ))}
                {(auditMaterials?.agent_traces ?? []).map((trace) => (
                  <details key={trace.trace_id} className="rounded-xl border p-4">
                    <summary className="cursor-pointer text-sm font-semibold">{trace.agent} Agent · {trace.tool_calls.length} 次工具调用 · {formatTime(trace.started_at)} 至 {formatTime(trace.ended_at)}</summary>
                    <p className="mt-2 text-sm text-muted-foreground">{trace.decision}</p>
                    <div className="mt-3 space-y-2">{summarizeToolCalls(trace.tool_calls).map((call) => (
                      <div key={call.tool} className="flex flex-wrap items-center gap-2 rounded-lg bg-muted/35 px-3 py-2 text-xs">
                        <Badge variant={call.failed === 0 ? "secondary" : "destructive"}>{call.failed === 0 ? "成功" : `${call.failed} 次异常`}</Badge>
                        <strong>{call.tool}</strong>
                        <span className="text-muted-foreground">调用 {call.count} 次 · {formatTime(call.first)} 至 {formatTime(call.last)}</span>
                        {call.error && <span className="w-full text-destructive">{call.error}</span>}
                      </div>
                    ))}</div>
                  </details>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative"><Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" /><Input value={auditQuery} onChange={(event) => setAuditQuery(event.target.value)} placeholder="搜索风险、证据原文、页码或证据编号" className="pl-9" /></div>
                <p className="text-xs text-muted-foreground">共 {filteredAuditEvidence.length} 条证据</p>
                <div className="grid gap-3 lg:grid-cols-2">{filteredAuditEvidence.map((evidence) => (
                  <details key={evidence.evidence_id} className="rounded-xl border bg-background/40 p-4">
                    <summary className="cursor-pointer list-none">
                      <div className="flex flex-wrap items-center gap-2"><strong className="text-sm">{evidence.evidence_id}</strong><Badge variant="outline">第 {evidence.page} 页</Badge><Badge variant="secondary">{evidence.evidence_type}</Badge></div>
                      <p className="mt-2 line-clamp-3 text-sm leading-6 text-muted-foreground">{evidence.quote}</p>
                    </summary>
                    <div className="mt-3 border-t pt-3">
                      <p className="whitespace-pre-wrap text-sm leading-6">{evidence.quote}</p>
                      <div className="mt-3 flex flex-wrap gap-1.5">{evidence.findings.map((finding) => <Badge key={finding.finding_id} variant="outline">{finding.title}</Badge>)}</div>
                      {evidence.screenshot_path && selectedRunId && <Button className="mt-3" size="sm" variant="outline" onClick={() => window.open(evidenceImageUrl(selectedRunId, evidence.evidence_id), "_blank")}>查看证据截图</Button>}
                    </div>
                  </details>
                ))}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {selectedRun && (
        <Card className="border-primary/25 bg-card/75">
          <CardHeader className="border-b border-border/60">
            <CardTitle>最终报告与交付材料</CardTitle>
            <CardDescription>复核闭环后，可下载正式报告、完整审计材料、四个时间窗口预测结果和推理日志。</CardDescription>
            <CardAction>
              <Badge variant={selectedRun.complete ? "secondary" : "outline"}>
                {selectedRun.complete ? "复核完成" : `还需处理 ${selectedRun.pending_findings} 条`}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={selectedRun.total_findings ? (selectedRun.reviewed_findings / selectedRun.total_findings) * 100 : 0} />
            <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
              <p className="text-sm text-muted-foreground">
                {selectedRun.complete
                  ? "人工复核已闭环，报告将包含多时间窗口风险预测、最终风险结论、原文证据和审计引用。"
                  : "请先处理风险清单中的待复核项；预测结果已经生成，无需等待复核才能查看。"}
              </p>
              <div className="flex flex-wrap justify-end gap-2">
                <Button disabled={!serviceAvailable || submitting || Boolean(exportingKind) || !selectedRun.complete} onClick={() => void handleExport("pdf")}>
                  {exportingKind === "pdf" ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
                  风险报告 PDF
                </Button>
                <Button variant="outline" disabled={!serviceAvailable || submitting || Boolean(exportingKind) || !selectedRun.complete} onClick={() => void handleExport("audit_pdf")}>
                  {exportingKind === "audit_pdf" ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
                  完整审计 PDF
                </Button>
                <Button variant="outline" disabled={!serviceAvailable || submitting || Boolean(exportingKind) || !selectedRun.complete} onClick={() => void handleExport("audit")}>
                  {exportingKind === "audit" ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
                  原始审计 JSON
                </Button>
                <Button variant="outline" disabled={!serviceAvailable || submitting || Boolean(exportingKind) || !selectedRun.complete} onClick={() => void handleExport("prediction_csv")}>
                  {exportingKind === "prediction_csv" ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
                  最终预测结果
                </Button>
                <Button variant="outline" disabled={!serviceAvailable || submitting || Boolean(exportingKind) || !selectedRun.complete} onClick={() => void handleExport("trace_json")}>
                  {exportingKind === "trace_json" ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
                  推理日志
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Dialog
        open={deleteDialogOpen}
        onOpenChange={(open) => {
          if (!deletingRun) setDeleteDialogOpen(open);
        }}
      >
        <DialogContent showCloseButton={!deletingRun}>
          <DialogHeader>
            <DialogTitle>删除这次分析记录？</DialogTitle>
            <DialogDescription>
              {selectedRun
                ? `将删除 ${selectedRun.company_id} · ${selectedRun.company_name} 的本次分析、风险、证据、复核、预测和已生成报告。`
                : "将删除当前选择的分析记录。"}
            </DialogDescription>
          </DialogHeader>
          <Alert variant="destructive">
            <Trash2 />
            <AlertTitle>原始招股书不会删除</AlertTitle>
            <AlertDescription>
              该记录将从系统列表和数据库中移除；独占的派生文件会移入项目回收区，共享文件继续保留。
            </AlertDescription>
          </Alert>
          <DialogFooter>
            <Button variant="outline" disabled={deletingRun} onClick={() => setDeleteDialogOpen(false)}>
              取消
            </Button>
            <Button variant="destructive" disabled={deletingRun} onClick={() => void handleDeleteRun()}>
              {deletingRun ? <Spinner data-icon="inline-start" /> : <Trash2 data-icon="inline-start" />}
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
