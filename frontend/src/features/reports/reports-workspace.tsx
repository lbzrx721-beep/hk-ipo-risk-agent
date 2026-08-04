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
import {
  TableEvidenceViewer,
  tableEvidenceReviewLabel,
  type TableEvidenceReview,
} from "@/features/reports/table-evidence-viewer";
import {
  evidenceImageUrl,
  exportRun,
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
import type { FindingDetail, RetryItem, RetryJob, ReviewRun, StoredFinding } from "@/types";

type ReportsWorkspaceProps = {
  serviceAvailable: boolean;
  onReconnect: () => Promise<void>;
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

function SummaryCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border bg-muted/15 px-3 py-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      <strong className="mt-1 block text-xl tabular-nums">{value}</strong>
    </div>
  );
}

export function ReportsWorkspace({ serviceAvailable, onReconnect, setError }: ReportsWorkspaceProps) {
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
  const [query, setQuery] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [comment, setComment] = useState("");
  const [editTitle, setEditTitle] = useState("");
  const [editClaim, setEditClaim] = useState("");
  const [editSeverity, setEditSeverity] = useState("medium");
  const [editConfidence, setEditConfidence] = useState(0.5);
  const [findingFilter, setFindingFilter] = useState<FindingFilter>("all");
  const [retryJobs, setRetryJobs] = useState<RetryJob[]>([]);
  const [retrySubmitting, setRetrySubmitting] = useState(false);
  const [tableEvidenceReviews, setTableEvidenceReviews] = useState<Record<string, TableEvidenceReview>>({});
  const selectedRunIdRef = useRef<string | null>(null);
  const selectedFindingIdRef = useRef<string | null>(null);

  useEffect(() => {
    selectedRunIdRef.current = selectedRunId;
    selectedFindingIdRef.current = selectedFindingId;
  }, [selectedFindingId, selectedRunId]);

  const refreshRuns = useCallback(async () => {
    const rows = await listReviewRuns();
    setRuns(rows);
    setSelectedRunId((current) => current && rows.some((row) => row.run_id === current)
      ? current
      : rows[0]?.run_id ?? null);
  }, []);

  useEffect(() => {
    setLoadingRuns(true);
    void refreshRuns()
      .catch((error) => setError(error instanceof Error ? error.message : "分析记录加载失败。"))
      .finally(() => setLoadingRuns(false));
  }, [refreshRuns, setError]);

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
        const pending = rows.find((row) => !["verified", "rejected"].includes(row.status));
        setSelectedFindingId(pending?.finding_id ?? rows[0]?.finding_id ?? null);
      })
      .catch((error) => active && setError(error instanceof Error ? error.message : "风险结论加载失败。"))
      .finally(() => active && setLoadingFindings(false));
    return () => { active = false; };
  }, [selectedRunId, setError]);

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
    const finalComment = [comment.trim(), ...tableReviewLines].filter(Boolean).join("\n");
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
      const nextId = next?.finding_id ?? selectedFindingId;
      setSelectedFindingId(nextId);
      setComment("");
      setTableEvidenceReviews((current) => Object.fromEntries(
        Object.entries(current).filter(([evidenceId]) => !detail?.evidences.some((evidence) => evidence.evidence_id === evidenceId)),
      ));
      setDetail(await getFindingDetail(selectedRunId, nextId));
    } catch (error) {
      setError(error instanceof Error ? error.message : "复核提交失败。");
    } finally {
      setSubmitting(false);
    }
  };

  const handleExport = async () => {
    if (!selectedRunId) return;
    if (!serviceAvailable) return setError("分析服务当前未连接，重新连接后才能导出报告。");
    setSubmitting(true);
    try {
      const files = await exportRun(selectedRunId);
      if (files.pdf) downloadFromUrl(files.pdf);
      await refreshRuns();
    } catch (error) {
      setError(error instanceof Error ? error.message : "正式报告导出失败。");
    } finally {
      setSubmitting(false);
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
          {technicalFailureCount > 0 && (
            <Button variant="outline" disabled={!serviceAvailable || retrySubmitting || retryActive} onClick={() => void startRetry("run")}>
              {retrySubmitting || retryActive ? <Spinner data-icon="inline-start" /> : <RefreshCw data-icon="inline-start" />}
              重试本公司 {technicalFailureCount} 条技术失败
            </Button>
          )}
          {selectedRun?.complete && (
            <Button disabled={!serviceAvailable || submitting} onClick={() => void handleExport()}>
              {submitting ? <Spinner data-icon="inline-start" /> : <Download data-icon="inline-start" />}
              导出正式报告
            </Button>
          )}
        </div>
      </section>

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
                    {detail.finding.uncertainty && (
                      <Alert variant={detail.retryable_failure ? "destructive" : "default"} className="mt-4">
                        <ShieldAlert />
                        <AlertTitle>{detail.retryable_failure ? "模型连接中断，风险结论尚未完成" : "不确定性说明"}</AlertTitle>
                        <AlertDescription className="space-y-3">
                          <p>{detail.finding.uncertainty}</p>
                          {activeRetryItem ? (
                            <div className="rounded-lg border border-current/15 bg-background/30 p-3">
                              <div className="flex items-center gap-2 font-medium"><Spinner />{activeRetryItem.stage}</div>
                              <Progress className="mt-2" value={Math.min(100, (activeRetryItem.attempt / activeRetryItem.maxAttempts) * 100)} />
                              <p className="mt-2 text-xs">第 {Math.max(1, activeRetryItem.attempt)} / {activeRetryItem.maxAttempts} 次尝试 · 复用现有证据，不重新解析PDF或向量化</p>
                            </div>
                          ) : detail.retryable_failure && (
                            <div className="flex flex-wrap gap-2">
                              <Button size="sm" disabled={!serviceAvailable || retrySubmitting} onClick={() => void startRetry("finding")}>
                                {retrySubmitting ? <Spinner data-icon="inline-start" /> : <RefreshCw data-icon="inline-start" />}重试当前风险
                              </Button>
                              <Button size="sm" variant="outline" disabled={!serviceAvailable || retrySubmitting} onClick={() => void startRetry("agent")}>
                                重试该 Agent 的失败项
                              </Button>
                              <Button size="sm" variant="outline" disabled={!serviceAvailable || retrySubmitting} onClick={() => void startRetry("run")}>
                                重试本公司全部技术失败
                              </Button>
                            </div>
                          )}
                        </AlertDescription>
                      </Alert>
                    )}
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
                          {evidence.table_id || evidence.table_cell_refs.length > 0 || evidence.evidence_type === "table" ? (
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
                      <Button disabled={!serviceAvailable || submitting || Boolean(activeRetryItem)} onClick={() => void submitDecision("confirmed")}>{submitting ? <Spinner data-icon="inline-start" /> : <CheckCircle2 data-icon="inline-start" />}确认</Button>
                      <Button variant="destructive" disabled={!serviceAvailable || submitting || Boolean(activeRetryItem)} onClick={() => void submitDecision("rejected")}><XCircle data-icon="inline-start" />驳回</Button>
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
                          <div><strong className="text-sm">{review.reviewer} · {review.decision}</strong><p className="mt-1 text-xs text-muted-foreground">{review.comment || "未填写补充意见"} · {formatTime(review.created_at)}</p></div>
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
    </div>
  );
}
