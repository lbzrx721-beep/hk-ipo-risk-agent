import {
  Activity,
  Bot,
  Check,
  ChevronRight,
  CirclePause,
  CirclePlay,
  FileSearch,
  ListTree,
  Network,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { memo, startTransition, useEffect, useMemo, useState } from "react";

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
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VirtualList } from "@/components/virtual-list";
import { parseAgentOutput, type AgentOutputCard as OutputCard } from "@/lib/agent-output";
import {
  compactDetail,
  companyStage,
  elapsedSeconds,
  formatDuration,
  formatTime,
  roleMeta,
  roleState,
  stateLabel,
  statusProgress,
} from "@/lib/presentation";
import { cn } from "@/lib/utils";
import type {
  AnalysisEvent,
  BatchCompany,
  BatchSnapshot,
  PendingUpload,
} from "@/types";
import { RiskScopePanel } from "./risk-scope-panel";
import { UploadPanel } from "./upload-panel";

type AnalysisWorkspaceProps = {
  batch: BatchSnapshot | null;
  pending: PendingUpload[];
  setPending: React.Dispatch<React.SetStateAction<PendingUpload[]>>;
  selected: BatchCompany | null;
  selectedKey: string | null;
  selectedEvents: AnalysisEvent[];
  visibleEvents: AnalysisEvent[];
  busy: boolean;
  serviceAvailable: boolean;
  setBusy: (value: boolean) => void;
  setError: (value: string | null) => void;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  onSelectCompany: (companyKey: string) => void;
  onAction: (kind: "pause" | "resume") => void;
};

const agentStateLabel: Record<string, string> = {
  pending: "等待",
  running: "分析中",
  complete: "已完成",
  failed: "异常",
  degraded: "部分降级",
  skipped: "本次无任务",
};

const outputKindLabel: Record<OutputCard["kind"], string> = {
  discovery: "原始发现",
  consolidated: "语义合并",
  analysis: "专业核验",
  decision: "裁决结果",
};

const severityText: Record<string, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
  critical: "极高风险",
  unknown: "待判断",
};

function publicAgentMessage(role: string, state: string | undefined, message: string | undefined): string {
  const value = message?.trim() ?? "";
  const internalIds = value.match(/\bopen_[0-9a-f]+\b/gi) ?? [];
  if (internalIds.length > 1) {
    return state === "running"
      ? "已接收批量风险任务，正在逐条核验证据并生成结构化结论。"
      : "批量风险任务状态已更新，技术详情可在事件流中查看。";
  }
  if (internalIds.length === 1) {
    return value.replace(/\bopen_[0-9a-f]+\b/gi, role === "decision" ? "当前裁决项" : "当前风险项");
  }
  return value;
}

function useNow(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, intervalMs]);
  return now;
}

function badgeVariant(state: string): "default" | "secondary" | "destructive" | "outline" {
  if (state === "failed") return "destructive";
  if (["running", "preparing", "pausing"].includes(state)) return "default";
  if (["completed", "awaiting_review"].includes(state)) return "secondary";
  return "outline";
}

function LabeledProgress({ value, label, ariaLabel }: { value: number; label: string; ariaLabel: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="truncate text-muted-foreground" title={label}>{label}</span>
        <span className="shrink-0 tabular-nums text-muted-foreground">{Math.round(value)}%</span>
      </div>
      <Progress value={value} aria-label={ariaLabel} />
    </div>
  );
}

const CompanyQueueItem = memo(function CompanyQueueItem({
  company,
  selected,
  nowMs,
  onSelect,
}: {
  company: BatchCompany;
  selected: boolean;
  nowMs: number;
  onSelect: (companyKey: string) => void;
}) {
  const percent = statusProgress(company);
  const elapsed = elapsedSeconds(company, nowMs);
  const activity = company.progress?.overall_text
    || company.progress?.current_activity
    || stateLabel[company.state]
    || company.state;
  return (
    <Button
      variant="outline"
      className={cn(
        "h-auto w-full justify-start rounded-xl px-3 py-3 text-left",
        selected && "border-primary/50 bg-primary/8 ring-1 ring-primary/15",
      )}
      onClick={() => onSelect(company.company_key)}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex items-center justify-between gap-2">
          <strong className="truncate text-sm">{company.company_id} · {company.company_name}</strong>
          <Badge variant={badgeVariant(company.state)}>{stateLabel[company.state] ?? company.state}</Badge>
        </div>
        <p className="truncate text-xs text-muted-foreground" title={activity}>{activity}</p>
        <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
          <span>{company.prospectus_date || "日期待确认"}</span>
          <span className="tabular-nums">已用时 {formatDuration(elapsed)}</span>
        </div>
        <LabeledProgress value={percent} label="当前进度" ariaLabel={`${company.company_name}分析进度`} />
      </div>
    </Button>
  );
});

const AgentCard = memo(function AgentCard({
  roleKey,
  label,
  description,
  events,
  company,
}: {
  roleKey: string;
  label: string;
  description: string;
  events: AnalysisEvent[];
  company: BatchCompany;
}) {
  const fallback = roleState(events, roleKey, company);
  const live = company.progress?.agents?.[roleKey];
  const state = live?.state ?? (
    fallback.tone === "danger"
      ? "failed"
      : fallback.tone === "success"
        ? "complete"
        : fallback.tone === "active"
          ? "running"
          : "pending"
  );
  const tone = state === "failed"
    ? "danger"
    : state === "degraded"
      ? "warning"
    : state === "running"
      ? "active"
      : state === "complete"
        ? "success"
        : "muted";
  const variant = tone === "danger"
    ? "destructive"
    : tone === "active"
      ? "default"
      : tone === "success"
        ? "secondary"
        : "outline";
  const progress = live?.progress ?? (state === "complete" || state === "skipped" ? 100 : 0);
  const detail = publicAgentMessage(roleKey, state, live?.message || fallback.detail);
  return (
    <Card size="sm" className={cn("min-h-44 bg-muted/10", state === "running" && "border-primary/30 bg-primary/[0.035]")}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-lg bg-primary/10 text-primary"><Bot className="size-4" /></span>
          {label}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <Badge variant={variant}>
            {state === "running" && <Spinner data-icon="inline-start" />}
            {agentStateLabel[state] ?? state}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <LabeledProgress value={progress} label={live?.progress_text ?? "尚未开始"} ariaLabel={`${label}执行进度`} />
        <p className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
});

function MetricCard({ label, value, helper }: { label: string; value: number; helper: string }) {
  return (
    <Card size="sm" className="bg-card/70">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
      </CardHeader>
      <CardContent><p className="text-xs text-muted-foreground">{helper}</p></CardContent>
    </Card>
  );
}

const severityBadgeVariant = {
  critical: "destructive",
  high: "destructive",
  medium: "secondary",
  low: "outline",
  unknown: "outline",
} as const;

function normalizedSeverity(severity: string): keyof typeof severityBadgeVariant {
  return severity in severityBadgeVariant
    ? severity as keyof typeof severityBadgeVariant
    : "unknown";
}

function AgentConclusionCard({ card }: { card: OutputCard }) {
  const confidence = card.confidence == null ? null : Math.round(card.confidence * 100);
  const severity = normalizedSeverity(card.severity);
  const outcome = card.kind === "decision"
    ? card.accepted === false ? "未采纳" : "建议采纳"
    : card.kind === "analysis"
      ? card.supported === false ? "证据不足" : "证据支持"
      : null;
  return (
    <Card
      size="sm"
      className="risk-output-card relative border animate-in fade-in slide-in-from-bottom-2 duration-300"
      data-severity={severity}
    >
      <span aria-hidden data-risk-accent className="absolute inset-y-0 left-0 w-1" />
      <CardHeader>
        <CardDescription>{outputKindLabel[card.kind]}</CardDescription>
        <CardTitle className="text-sm leading-snug">{card.title}</CardTitle>
        <CardAction className="flex flex-wrap justify-end gap-1">
          <Badge variant={severityBadgeVariant[severity]}>
            {severityText[card.severity] ?? "待判断"}
          </Badge>
          {confidence != null && <Badge variant="secondary">置信度 {confidence}%</Badge>}
          {outcome && <Badge variant={outcome === "未采纳" || outcome === "证据不足" ? "outline" : "secondary"}>{outcome}</Badge>}
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-xs">
        {card.claim && <p className="leading-relaxed">{card.claim}</p>}
        {card.rationale && <p className="leading-relaxed text-muted-foreground">裁决依据：{card.rationale}</p>}
        {card.uncertainty && <p className="leading-relaxed text-muted-foreground">待核实：{card.uncertainty}</p>}
        <div className="flex flex-wrap gap-1">
          {card.owner && <Badge variant="outline">负责：{card.owner}</Badge>}
          <Badge variant="outline">原文证据 {card.evidenceIds.length} 条</Badge>
        </div>
      </CardContent>
    </Card>
  );
}

function StreamingCardPlaceholder({ role, first = false }: { role: string; first?: boolean }) {
  const label = roleMeta.find(([key]) => key === role)?.[1] ?? "Agent";
  const detail = role === "discovery"
    ? "模型正在整理候选片段；完成一条后会立即显示为风险卡片。"
    : role === "decision"
      ? "模型正在整合专业意见；完成一条后会立即显示为裁决卡片。"
      : "正在复用现有证据批量核验；完成一条后会立即显示为专业结论。";
  return (
    <Card
      size="sm"
      className="border-dashed border-primary/25 bg-primary/[0.025] animate-in fade-in slide-in-from-bottom-2 duration-300"
    >
      <CardHeader>
        <CardDescription className="flex items-center gap-2 text-primary">
          <Spinner />{first ? "首条生成中" : "下一条生成中"}
        </CardDescription>
        <CardTitle className="text-sm">{label}正在生成{first ? "首条" : "下一条"}结构化结论</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Skeleton className="h-3 w-11/12" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-3/4" />
        <p className="text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  );
}

function AgentOutputPane({
  role,
  events,
  company,
}: {
  role: string;
  events: AnalysisEvent[];
  company: BatchCompany;
}) {
  const [page, setPage] = useState(1);
  const output = useMemo(
    () => parseAgentOutput(events, role, company.progress?.risk_titles?.[role] ?? {}),
    [company.progress?.risk_titles, events, role],
  );
  const cards = output.cards;
  const pageSize = 6;
  const pageCount = Math.max(1, Math.ceil(cards.length / pageSize));
  const safePage = Math.min(page, pageCount);
  const visible = cards.slice((safePage - 1) * pageSize, safePage * pageSize);
  const progress = company.progress?.agents?.[role];
  const isStreaming = progress?.state === "running" || output.incomplete;
  const rawCandidates = Math.max(
    company.progress?.funnel?.raw_candidates ?? 0,
    company.risk_scope_review?.funnel?.raw_candidates ?? 0,
    cards.filter((card) => card.kind === "discovery").length,
  );
  const consolidatedCandidates = Math.max(
    company.progress?.funnel?.consolidated_candidates ?? 0,
    company.risk_scope_review?.funnel?.consolidated_candidates ?? 0,
    cards.filter((card) => card.kind === "consolidated").length,
  );

  useEffect(() => {
    if (isStreaming) setPage(pageCount);
  }, [cards.length, isStreaming, pageCount]);

  if (cards.length === 0) {
    if (isStreaming) {
      return (
        <div className="grid gap-3 xl:grid-cols-2">
          <StreamingCardPlaceholder role={role} first />
        </div>
      );
    }
    return (
      <Empty className="min-h-52 border bg-background/20">
        <EmptyHeader>
          <EmptyMedia variant="icon">{progress?.state === "running" ? <Spinner /> : <FileSearch />}</EmptyMedia>
          <EmptyTitle>{progress?.state === "skipped" ? "本次未分配风险" : progress?.state === "running" ? "正在生成结构化卡片" : "等待结构化输出"}</EmptyTitle>
          <EmptyDescription>{publicAgentMessage(role, progress?.state, progress?.message) || "Agent产生完整结论后，风险卡片会在这里逐条出现。"}</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap gap-1">
          <Badge variant="secondary">已形成 {cards.length} 条结构化结论</Badge>
          {role === "discovery" && (
            <>
              <Badge variant="outline">原始发现 {rawCandidates}</Badge>
              <Badge variant="outline">全局合并 {consolidatedCandidates}</Badge>
            </>
          )}
        </div>
        {pageCount > 1 && (
          <div className="flex items-center gap-2">
            <Button size="sm" variant="outline" disabled={safePage <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>上一页</Button>
            <span className="text-xs tabular-nums text-muted-foreground">{safePage}/{pageCount}</span>
            <Button size="sm" variant="outline" disabled={safePage >= pageCount} onClick={() => setPage((value) => Math.min(pageCount, value + 1))}>下一页</Button>
          </div>
        )}
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        {visible.map((card, index) => (
          <AgentConclusionCard
            key={`${card.kind}-${card.riskType ?? card.title}-${index}`}
            card={card}
          />
        ))}
        {isStreaming && safePage === pageCount && (
          <StreamingCardPlaceholder role={role} />
        )}
      </div>
    </div>
  );
}

function StageNotice({ company, events }: { company: BatchCompany; events: AnalysisEvent[] }) {
  const agents = company.progress?.agents ?? {};
  const specialists = ["finance", "legal", "business", "market"].map((role) => agents[role]?.state ?? "pending");
  const reviewCompleted = events.some((event) => event.stage === "risk_scope_review_completed");
  let title = "";
  let detail = "";

  if (company.state === "awaiting_review" || company.risk_scope_review) {
    title = "风险发现与全局合并已完成，等待人工确认";
    detail = "专业Agent尚未启动是正常状态。请在下方确认风险范围，提交后财务、法律、业务和市场Agent会并行运行。";
  } else if (agents.discovery?.state === "running" && (agents.discovery.progress ?? 0) >= 92) {
    title = "批次扫描已完成，正在进行全局语义合并";
    detail = "系统正在合并同主题风险并整理候选范围，完成后会进入人工确认，不会直接跳过。";
  } else if (reviewCompleted && specialists.every((state) => state === "pending")) {
    title = "人工范围已确认，正在分派专业Agent";
    detail = "编排器正在把风险分派给财务、法律、业务和市场Agent；未匹配风险的角色会显示“本次无任务”。";
  } else if (specialists.some((state) => state === "running")) {
    title = "专业Agent正在并行核验证据";
    detail = "四个专业分支独立更新；全部结束后，决策Agent才会开始整合与裁决。";
  } else if (agents.decision?.state === "running") {
    title = "专业核验完成，决策Agent正在裁决";
    detail = "决策Agent只整合、合并或驳回已有提案，不会悄悄创建新的风险主题。";
  }

  if (!title) return null;
  return (
    <Alert>
      {company.state === "awaiting_review" ? <ShieldCheck /> : agents.decision?.state === "running" ? <Network /> : <Activity />}
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{detail}</AlertDescription>
    </Alert>
  );
}

function CompanyDetail({
  batch,
  company,
  events,
  visibleEvents,
  nowMs,
  serviceAvailable,
  onSnapshot,
  setError,
}: {
  batch: BatchSnapshot;
  company: BatchCompany;
  events: AnalysisEvent[];
  visibleEvents: AnalysisEvent[];
  nowMs: number;
  serviceAvailable: boolean;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setError: (value: string | null) => void;
}) {
  const currentStage = companyStage(company, events);
  const pipeline = ["文档登记", "解析与索引", "风险发现", "范围确认", "专业核验", "决策整合"];
  const percent = statusProgress(company);
  const elapsed = elapsedSeconds(company, nowMs);

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <Card className="bg-card/80">
        <CardHeader className="border-b border-border/60">
          <CardTitle className="text-xl">{company.company_id} · {company.company_name}</CardTitle>
          <CardDescription className="break-all">
            {company.prospectus_date} · {company.split} · {company.run_id ?? "等待生成 run_id"}
          </CardDescription>
          <CardAction><Badge variant={badgeVariant(company.state)}>{stateLabel[company.state] ?? company.state}</Badge></CardAction>
        </CardHeader>
        <CardContent className="grid gap-3 pt-1 sm:grid-cols-3">
          {[
            ["风险结论", company.finding_count ?? "—"],
            ["原文证据", company.evidence_count ?? "—"],
            ["分析用时", formatDuration(elapsed)],
          ].map(([label, value]) => (
            <div key={label} className="rounded-lg border border-border/60 bg-muted/15 p-3">
              <p className="text-xs text-muted-foreground">{label}</p>
              <strong className="mt-1 block text-lg font-medium tabular-nums">{value}</strong>
            </div>
          ))}
          <div className="sm:col-span-3">
            <LabeledProgress value={percent} label={company.progress?.overall_text || "等待实时分析事件"} ariaLabel="公司端到端分析进度" />
          </div>
        </CardContent>
      </Card>

      <Card size="sm" className="overflow-x-auto bg-card/70">
        <CardContent className="flex min-w-[760px] items-center gap-2">
          {pipeline.map((label, index) => (
            <div key={label} className="flex min-w-0 flex-1 items-center gap-2">
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <span className={cn(
                  "grid size-7 shrink-0 place-items-center rounded-full border text-xs",
                  index < currentStage && "border-primary bg-primary text-primary-foreground",
                  index === currentStage && "border-primary bg-primary/10 text-primary ring-4 ring-primary/10",
                )}>
                  {index < currentStage ? <Check className="size-3.5" /> : index + 1}
                </span>
                <span className={cn("truncate text-xs font-medium", index > currentStage && "text-muted-foreground")}>{label}</span>
              </div>
              {index < pipeline.length - 1 && <ChevronRight className="size-4 shrink-0 text-muted-foreground/50" />}
            </div>
          ))}
        </CardContent>
      </Card>

      {company.error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="text-sm text-destructive">{company.error}</CardContent>
        </Card>
      )}

      <StageNotice company={company} events={events} />

      <RiskScopePanel
        key={company.company_key}
        batch={batch}
        company={company}
        serviceAvailable={serviceAvailable}
        onSnapshot={onSnapshot}
        setError={setError}
      />

      <Card className="bg-card/75">
        <CardHeader className="border-b border-border/60">
          <CardTitle>Agent协同矩阵</CardTitle>
          <CardDescription>{company.progress?.current_activity || "每个角色独立更新，专业Agent在人工确认后并行启动。"}</CardDescription>
          <CardAction><Badge variant="outline">实时状态</Badge></CardAction>
        </CardHeader>
        <CardContent className="flex flex-col gap-5">
          <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {roleMeta.map(([key, label, description]) => (
              <AgentCard
                key={key}
                roleKey={key}
                label={label}
                description={description}
                events={events}
                company={company}
              />
            ))}
          </div>

          <Separator />

          <Tabs defaultValue="discovery">
            <div className="flex flex-col justify-between gap-3 xl:flex-row xl:items-center">
              <div>
                <h3 className="text-sm font-medium">结构化分析卡片</h3>
                <p className="mt-1 text-xs text-muted-foreground">风险发现、专业提案和决策裁决会随模型输出逐步增加。</p>
              </div>
              <TabsList variant="line" className="h-auto max-w-full flex-wrap justify-start">
                {roleMeta.map(([key, label]) => (
                  <TabsTrigger key={key} value={key}>{label}</TabsTrigger>
                ))}
                <TabsTrigger value="events"><Activity data-icon="inline-start" />事件流</TabsTrigger>
              </TabsList>
            </div>
            {roleMeta.map(([key]) => (
              <TabsContent key={key} value={key}>
                <AgentOutputPane role={key} events={events} company={company} />
              </TabsContent>
            ))}
            <TabsContent value="events">
              <div className="max-h-[480px] overflow-auto pr-2">
                {visibleEvents.length > 0 ? (
                  <div className="flex flex-col">
                    {visibleEvents.map((event, index) => (
                      <article key={`${event.stage}-${event.occurred_at}-${index}`} className="grid grid-cols-[70px_12px_minmax(0,1fr)] gap-3 border-b border-border/50 py-3 last:border-b-0">
                        <time className="text-xs tabular-nums text-muted-foreground">{formatTime(event.occurred_at)}</time>
                        <span className="mt-1.5 size-2 rounded-full bg-primary shadow-[0_0_12px_var(--primary)]" />
                        <div className="min-w-0">
                          <strong className="text-xs font-medium">{event.stage.replaceAll(":", " · ")}</strong>
                          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{compactDetail(event.detail, 260)}</p>
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <Empty className="min-h-64 border">
                    <EmptyHeader>
                      <EmptyMedia variant="icon"><FileSearch /></EmptyMedia>
                      <EmptyTitle>等待第一条分析事件</EmptyTitle>
                      <EmptyDescription>PDF 登记后，事件会在这里持续追加。</EmptyDescription>
                    </EmptyHeader>
                  </Empty>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

export function AnalysisWorkspace({
  batch,
  pending,
  setPending,
  selected,
  selectedKey,
  selectedEvents,
  visibleEvents,
  busy,
  serviceAvailable,
  setBusy,
  setError,
  onSnapshot,
  onSelectCompany,
  onAction,
}: AnalysisWorkspaceProps) {
  const clockActive = Boolean(batch?.companies.some((company) =>
    ["running", "preparing", "ready", "pausing"].includes(company.state),
  ));
  const nowMs = useNow(clockActive);
  const selectCompany = (companyKey: string) => startTransition(() => onSelectCompany(companyKey));
  const batchPercent = batch?.companies.length
    ? batch.companies.reduce((total, company) => total + statusProgress(company), 0) / batch.companies.length
    : 0;
  return (
    <div className="mx-auto flex w-full max-w-[1920px] flex-col gap-5 px-4 py-6 lg:px-6 2xl:px-8">
      <section className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div>
          <Badge variant="outline" className="mb-3">Hong Kong IPO intelligence</Badge>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">招股书风险分析工作台</h1>
          <p className="mt-2 text-sm text-muted-foreground">从 PDF 到风险裁决，每家公司独立运行、实时更新、全程可追溯。</p>
        </div>
        {batch && (
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={badgeVariant(batch.state)}>{stateLabel[batch.state] ?? batch.state}</Badge>
            {batch.state === "running" || batch.state === "pausing" ? (
              <Button variant="outline" disabled={busy || !serviceAvailable} onClick={() => onAction("pause")}>
                <CirclePause data-icon="inline-start" />当前公司完成后暂停
              </Button>
            ) : batch.state === "paused" || batch.state === "failed" ? (
              <Button variant="outline" disabled={busy || !serviceAvailable} onClick={() => onAction("resume")}>
                <CirclePlay data-icon="inline-start" />继续队列
              </Button>
            ) : null}
          </div>
        )}
      </section>

      <UploadPanel
        pending={pending}
        setPending={setPending}
        activeBatch={batch}
        busy={busy}
        serviceAvailable={serviceAvailable}
        setBusy={setBusy}
        setError={setError}
        onSnapshot={onSnapshot}
      />

      {batch ? (
        <>
          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <MetricCard label="公司总数" value={batch.company_count} helper="当前工作区" />
            <MetricCard label="分析中" value={(batch.counts.running ?? 0) + (batch.counts.preparing ?? 0)} helper="独立实时更新" />
            <MetricCard label="待确认风险" value={batch.counts.awaiting_review ?? 0} helper="可逐公司放行" />
            <MetricCard label="已完成" value={batch.counts.completed ?? 0} helper="结果持续保留" />
            <MetricCard label="异常" value={batch.counts.failed ?? 0} helper="支持恢复" />
          </section>

          <Card size="sm" className="bg-card/75">
            <CardHeader>
              <CardTitle>批次总体进度</CardTitle>
              <CardDescription>按每家公司真实端到端进度计算，进度只前进、不回跳。</CardDescription>
            </CardHeader>
            <CardContent>
              <LabeledProgress
                value={batchPercent}
                label={batch.segments.find((segment) => segment.active)?.phase_detail || stateLabel[batch.state] || batch.state}
                ariaLabel="批次总体进度"
              />
            </CardContent>
          </Card>

          <div className="grid min-w-0 gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
            <Card className="h-fit bg-card/75 xl:sticky xl:top-20">
              <CardHeader className="border-b border-border/60">
                <CardTitle className="flex items-center gap-2"><ListTree className="size-4 text-primary" />公司任务队列</CardTitle>
                <CardDescription>点击公司只切换右侧详情，不刷新整个页面。</CardDescription>
                <CardAction><Badge variant="outline">{batch.company_count} 家</Badge></CardAction>
              </CardHeader>
              <CardContent>
                <VirtualList
                  items={batch.companies}
                  estimateSize={148}
                  getKey={(company) => company.company_key}
                  renderItem={(company) => (
                    <CompanyQueueItem
                      company={company}
                      selected={selectedKey === company.company_key}
                      nowMs={nowMs}
                      onSelect={selectCompany}
                    />
                  )}
                  className="h-[min(560px,calc(100vh-270px))] min-h-60"
                />
              </CardContent>
            </Card>

            {selected ? (
              <CompanyDetail
                batch={batch}
                company={selected}
                events={selectedEvents}
                visibleEvents={visibleEvents}
                nowMs={nowMs}
                serviceAvailable={serviceAvailable}
                onSnapshot={onSnapshot}
                setError={setError}
              />
            ) : (
              <Empty className="min-h-[460px] border bg-card/60">
                <EmptyHeader>
                  <EmptyMedia variant="icon"><ChevronRight /></EmptyMedia>
                  <EmptyTitle>请选择一家公司</EmptyTitle>
                  <EmptyDescription>公司详情和 Agent 进度将在这里展示。</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </div>
        </>
      ) : pending.length === 0 ? (
        <Empty className="min-h-[360px] border bg-card/60">
          <EmptyHeader>
            <EmptyMedia variant="icon"><ShieldCheck /></EmptyMedia>
            <EmptyTitle>从一份招股书开始</EmptyTitle>
            <EmptyDescription>拖入 PDF 后，系统会识别公司信息并创建实时分析队列。</EmptyDescription>
          </EmptyHeader>
          <div className="flex flex-wrap justify-center gap-2">
            <Badge variant="outline"><Network data-icon="inline-start" />多公司并行</Badge>
            <Badge variant="outline"><Activity data-icon="inline-start" />Agent 流式输出</Badge>
            <Badge variant="outline"><Timer data-icon="inline-start" />局部实时更新</Badge>
          </div>
        </Empty>
      ) : null}
    </div>
  );
}
