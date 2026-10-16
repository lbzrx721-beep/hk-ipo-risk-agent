import {
  Activity,
  ArrowDownToLine,
  Bot,
  ChartNoAxesCombined,
  Check,
  ChevronDown,
  ChevronRight,
  CirclePause,
  CirclePlay,
  FileSearch,
  ListTree,
  LockKeyhole,
  MessageSquareWarning,
  Network,
  ShieldCheck,
  Timer,
} from "lucide-react";
import { memo, startTransition, useEffect, useMemo, useRef, useState } from "react";

import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { DebateTimeline } from "@/components/debate-timeline";
import { parseAgentOutput, type AgentOutputCard as OutputCard } from "@/lib/agent-output";
import { shouldShowStreamingPlaceholder } from "@/lib/streaming-state";
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
  DebateRecord,
  PendingUpload,
} from "@/types";
import { LlmRecoveryPanel } from "./llm-recovery-panel";
import { RiskScopePanel } from "./risk-scope-panel";
import { UploadPanel } from "./upload-panel";
import { CompanyReviewPlanView, parseCompanyReviewPlan } from "./company-review-plan";

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
  onOpenPredictions: () => void;
};

const agentStateLabel: Record<string, string> = {
  pending: "等待",
  running: "分析中",
  complete: "已完成",
  failed: "异常",
  degraded: "部分降级",
  skipped: "本次无任务",
};

const modelDisplayName: Record<string, string> = {
  deepseek_flash: "DeepSeek Flash",
  deepseek_pro: "DeepSeek Pro",
  kimi: "Kimi K3",
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
  if (state === "awaiting_recovery") return "destructive";
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
  const assignedModel = company.agent_models?.[roleKey];
  return (
    <Card size="sm" className={cn("min-h-44 bg-muted/10", state === "running" && "border-primary/30 bg-primary/[0.035]")}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-lg bg-primary/10 text-primary"><Bot className="size-4" /></span>
          {label}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction className="flex flex-col items-end gap-1">
          <Badge variant={variant}>
            {state === "running" && <Spinner data-icon="inline-start" />}
            {agentStateLabel[state] ?? state}
          </Badge>
          {assignedModel && <Badge variant="outline"><LockKeyhole data-icon="inline-start" />{modelDisplayName[assignedModel] ?? assignedModel}</Badge>}
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

const AgentConclusionCard = memo(function AgentConclusionCard({ card }: { card: OutputCard }) {
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
      className="risk-output-card relative border animate-in fade-in slide-in-from-bottom-1 duration-200"
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
        <div className="flex flex-wrap gap-1">
          {card.owner && <Badge variant="outline">负责：{card.owner}</Badge>}
          <Badge variant="outline">原文证据 {card.evidenceIds.length} 条</Badge>
        </div>
      </CardContent>
    </Card>
  );
}, (previous, next) => (
  previous.card.title === next.card.title
  && previous.card.claim === next.card.claim
  && previous.card.severity === next.card.severity
  && previous.card.confidence === next.card.confidence
  && previous.card.owner === next.card.owner
  && previous.card.supported === next.card.supported
  && previous.card.accepted === next.card.accepted
  && previous.card.rationale === next.card.rationale
  && previous.card.evidenceIds.join("|") === next.card.evidenceIds.join("|")
));

function agentCardKey(card: OutputCard): string {
  return card.riskType
    ? `${card.kind}:${card.riskType}`
    : [card.kind, card.owner ?? "", card.title].join(":");
}

function StreamingCardPlaceholder({ role, first = false }: { role: string; first?: boolean }) {
  const label = roleMeta.find(([key]) => key === role)?.[1] ?? "Agent";
  const detail = role === "discovery"
    ? "正在检索公司概况、核对原文引用并规划审查问题；本阶段不判断风险成立。"
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
          <Spinner />{role === "discovery" ? "画像与审查规划中" : first ? "首条生成中" : "下一条生成中"}
        </CardDescription>
        <CardTitle className="text-sm">{role === "discovery" ? "风险发现 Agent 正在构建公司画像与待查问题" : `${label}正在生成${first ? "首条" : "下一条"}结构化结论`}</CardTitle>
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

const stageLabel: Record<string, string> = {
  analysis_model_selected: "分析模型已确认",
  upload_started: "PDF 登记开始",
  upload_completed: "PDF 登记完成",
  parsing_started: "PDF 解析开始",
  parsing_progress: "PDF 解析中",
  parsing_completed: "PDF 解析完成",
  embedding_started: "向量索引创建",
  embedding_progress: "向量索引创建中",
  embedding_join_wait: "等待索引同步",
  embedding_join_completed: "索引同步完成",
  vector_store_indexed: "向量库已就绪",
  embedding_fallback: "索引降级（备用方案）",
  agents_started: "多 Agent 分析启动",
  agents_completed: "多 Agent 分析完成",
  risk_scoping_started: "风险发现开始",
  risk_scoping_completed: "风险发现完成",
  text_embellishment_registered: "文本粉饰模块已启用",
  text_embellishment_started: "文本粉饰分析开始",
  text_embellishment_completed: "文本粉饰分析完成",
  "agent:decision:debate_requested": "决策Agent发起质疑",
  "agent:finance:debate_retrieval": "财务Agent定向补证",
  "agent:legal:debate_retrieval": "法律Agent定向补证",
  "agent:business:debate_retrieval": "业务Agent定向补证",
  "agent:market:debate_retrieval": "市场Agent定向补证",
  "agent:finance:debate_response": "财务Agent提交补证",
  "agent:legal:debate_response": "法律Agent提交补证",
  "agent:business:debate_response": "业务Agent提交补证",
  "agent:market:debate_response": "市场Agent提交补证",
  "agent:decision:debate_completed": "决策Agent完成再次裁决",
  "agent:decision:debate_failed": "补证辩论异常",
  debate_record: "辩论记录已保存",
  risk_scope_review_started: "等待确认风险范围",
  risk_scope_review_submitted: "风险范围已提交",
  risk_scope_review_completed: "已确认风险范围",
  rag_features_started: "特征提取开始",
  rag_features_progress: "特征提取中",
  rag_features_completed: "特征提取完成",
  rag_features_failed: "特征提取异常",
  features_started: "上市前特征开始",
  features_progress: "上市前特征采集中",
  features_completed: "上市前特征采集完成",
  features_failed: "上市前特征异常",
  prediction_started: "风险预测开始",
  prediction_completed: "风险预测完成",
  prediction_failed: "风险预测异常",
  evidence_started: "证据链生成开始",
  evidence_completed: "证据链生成完成",
  agent_completed: "Agent 阶段完成",
  completed: "已完成",
  failed: "异常中止",
};

function stageCapabilities(stage: string): string[] {
  if (stage.startsWith("upload_")) return ["文件哈希", "PDF登记"];
  if (stage.startsWith("parsing_")) return ["PyMuPDF", "RapidOCR", "表格识别"];
  if (stage.startsWith("embedding_") || stage === "vector_store_indexed") return ["Qwen3-Embedding-4B", "pgvector"];
  if (stage.startsWith("risk_scop") || stage.startsWith("agent:discovery")) return ["RAG检索", "风险发现能力"];
  if (stage.startsWith("text_embellishment")) return ["向量检索", "披露语言风险分析"];
  if (stage.includes("debate")) return ["定向补证", "多Agent辩论"];
  if (/^agent:(finance|legal|business|market):/.test(stage)) return ["所选推理模型", "专业核验能力"];
  if (stage === "analysis_model_selected") return ["模型路由", "任务级配置"];
  if (stage.startsWith("agent:decision")) return ["所选推理模型", "风险裁决能力"];
  if (stage.startsWith("rag_features")) return ["pgvector检索", "财务字段提取"];
  if (stage.startsWith("features_")) return ["特征工程", "时点校验"];
  if (stage.startsWith("prediction_")) return ["1/5/20/60日模型", "模型契约"];
  if (stage.startsWith("evidence_")) return ["原文定位", "证据截图"];
  if (stage === "completed") return ["审计落盘"];
  return [];
}

function stageLabelClean(stage: string): string {
  const direct = stageLabel[stage];
  if (direct) return direct;
  // "agent:finance:completed" → "财务 Agent 完成"
  const agentMatch = /^agent:(\w+):(.+)$/.exec(stage);
  if (agentMatch) {
    const roleMap: Record<string, string> = {
      discovery: "发现", finance: "财务", legal: "法律", business: "业务", market: "市场", decision: "决策",
    };
    const role = roleMap[agentMatch[1]] ?? agentMatch[1];
    const verb = agentMatch[2] === "completed" ? "完成" : agentMatch[2] === "running" ? "运行中" : agentMatch[2] === "failed" ? "异常" : agentMatch[2];
    return `${role} Agent ${verb}`;
  }
  return stage.replace(/_/g, " ").replace(/:/g, " · ");
}

function cleanDetail(detail: string): string {
  // Remove internal IDs and noisy tokens
  let cleaned = detail
    .replace(/\bopen_[0-9a-f]+\b/gi, "")
    .replace(/\[request .*?]\s*/gi, "")
    .replace(/概率\s*\d+(?:\.\d+)?%/g, "方向性预警已生成")
    .replace(/\s{2,}/g, " ")
    .trim();
  if (cleaned.length > 160) cleaned = cleaned.slice(0, 157) + "…";
  return cleaned || detail.slice(0, 120);
}

function groupEvents(events: AnalysisEvent[]): { stage: string; label: string; events: AnalysisEvent[] }[] {
  const groups: { stage: string; label: string; events: AnalysisEvent[] }[] = [];
  for (const event of events) {
    const last = groups[groups.length - 1];
    if (last && last.stage === event.stage) {
      last.events.push(event);
    } else {
      groups.push({
        stage: event.stage,
        label: stageLabelClean(event.stage),
        events: [event],
      });
    }
  }
  return groups;
}

function compactGroupEvents(stage: string, events: AnalysisEvent[]): AnalysisEvent[] {
  const noisy = stage.endsWith("_progress") || stage.includes("embedding_progress") || stage.includes("vectorization");
  if (!noisy || events.length <= 2) return events;
  return [events[0], events[events.length - 1]];
}

function parseDebateRecords(events: AnalysisEvent[]): DebateRecord[] {
  const records = new Map<string, DebateRecord>();
  for (const event of events) {
    if (event.stage !== "debate_record") continue;
    try {
      const record = JSON.parse(event.detail) as DebateRecord;
      if (!record.debate_id || !record.risk_type) continue;
      records.set(record.debate_id, record);
    } catch {
      // Older or interrupted runs may contain a partial business event.
    }
  }
  return [...records.values()];
}

function EventLog({ events }: { events: AnalysisEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [userScrolled, setUserScrolled] = useState(false);
  const displayEvents = useMemo(
    () => events.filter((event) => !event.stage.endsWith(":delta") && event.stage !== "debate_record" && event.stage !== "company_review_plan"),
    [events],
  );
  const groups = useMemo(() => groupEvents(displayEvents), [displayEvents]);

  useEffect(() => {
    if (!userScrolled && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events, userScrolled]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const el = scrollRef.current;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    setUserScrolled(!atBottom);
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setUserScrolled(false);
    }
  };

  if (events.length === 0) {
    return (
      <Empty className="min-h-64 border">
        <EmptyHeader>
          <EmptyMedia variant="icon"><FileSearch /></EmptyMedia>
          <EmptyTitle>等待第一条分析事件</EmptyTitle>
          <EmptyDescription>PDF 登记后，事件会在这里持续追加。</EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <div className="relative">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="max-h-[480px] overflow-auto pr-1"
      >
        {groups.map((group, gi) => (
          <div key={`${group.stage}-${gi}`} className="border-b border-border/40 last:border-b-0">
            <div className="flex items-center gap-2 px-1 py-2">
              <span className="size-2 rounded-full bg-primary/70" />
              <span className="text-xs font-semibold text-foreground/85">{group.label}</span>
              {group.events.length > 1 && (
                <Badge variant="secondary" className="ml-auto text-[10px]">{group.events.length}</Badge>
              )}
              {stageCapabilities(group.stage).map((capability) => (
                <Badge key={capability} variant="outline" className="text-[10px] font-normal text-muted-foreground">{capability}</Badge>
              ))}
            </div>
            {compactGroupEvents(group.stage, group.events).map((event, ei, shown) => (
              <div key={`${gi}-${ei}`} className="ml-2.5 border-l-2 border-muted pl-4 pb-2.5 last:pb-3">
                <time className="text-[11px] tabular-nums text-muted-foreground/70">
                  {formatTime(event.occurred_at)}
                  {group.events.length > 2 && shown.length === 2 && ei === 1 ? ` · 共 ${group.events.length} 次更新` : ""}
                </time>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{cleanDetail(event.detail)}</p>
              </div>
            ))}
          </div>
        ))}
      </div>

      {userScrolled && (
        <Button
          variant="secondary"
          size="sm"
          className="absolute bottom-3 right-3 shadow-lg"
          onClick={scrollToBottom}
        >
          <ArrowDownToLine className="size-3.5" />
          最新
        </Button>
      )}
    </div>
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
  const isStreaming = shouldShowStreamingPlaceholder(progress?.state, output.incomplete);
  const rawCandidates = Math.max(
    company.progress?.funnel?.raw_candidates ?? 0,
    company.risk_scope_review?.funnel?.raw_candidates ?? 0,
    output.rawCount,
  );
  const consolidatedCandidates = Math.max(
    company.progress?.funnel?.consolidated_candidates ?? 0,
    company.risk_scope_review?.funnel?.consolidated_candidates ?? 0,
    output.consolidatedCount,
  );

  useEffect(() => {
    setPage((current) => Math.min(current, pageCount));
  }, [pageCount]);

  if (role === "discovery") {
    const plan = parseCompanyReviewPlan(events);
    if (plan) return <CompanyReviewPlanView plan={plan} />;
  }

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
        {visible.map((card) => (
          <AgentConclusionCard
            key={agentCardKey(card)}
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

  if (company.state === "awaiting_recovery" || company.llm_failure_recovery) {
    title = "风险发现模型异常，等待选择恢复方式";
    detail = "当前公司已暂停；可重试同一模型阶段，或明确降级为BM25目录检索。";
  } else if (company.state === "awaiting_review" || company.risk_scope_review) {
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

type AutomaticStageState = "waiting" | "running" | "complete" | "failed";

const automaticStageLabel: Record<AutomaticStageState, string> = {
  waiting: "等待上游",
  running: "自动执行中",
  complete: "已完成",
  failed: "未完成",
};

function latestEvent(events: AnalysisEvent[], stages: string[]): AnalysisEvent | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (stages.includes(events[index].stage)) return events[index];
  }
  return undefined;
}

function progressMarker(detail: string | undefined): { current: number; total: number; detail: string } | null {
  const match = /^\[(\d+)\/(\d+)]\s*(.*)$/.exec(detail?.trim() ?? "");
  if (!match) return null;
  return {
    current: Number(match[1]),
    total: Math.max(1, Number(match[2])),
    detail: match[3] || "正在处理",
  };
}

function candidateCount(detail: string | undefined): number | null {
  const match = /快照已生成：(\d+)项/.exec(detail ?? "");
  return match ? Number(match[1]) : null;
}

function AutomaticStageCard({
  kind,
  title,
  description,
  state,
  progress,
  progressText,
  detail,
  metrics,
}: {
  kind: "features" | "prediction";
  title: string;
  description: string;
  state: AutomaticStageState;
  progress: number;
  progressText: string;
  detail: string;
  metrics: string[];
}) {
  const Icon = kind === "features" ? ListTree : ShieldCheck;
  return (
    <Card size="sm" className={cn("bg-card/75", state === "running" && "border-primary/35 bg-primary/[0.035]")}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-lg bg-primary/10 text-primary"><Icon /></span>
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <Badge variant={state === "failed" ? "destructive" : state === "complete" ? "secondary" : state === "running" ? "default" : "outline"}>
            {state === "running" && <Spinner data-icon="inline-start" />}
            {automaticStageLabel[state]}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <LabeledProgress value={progress} label={progressText} ariaLabel={`${title}进度`} />
        <p className="min-h-10 text-xs leading-relaxed text-muted-foreground">{detail}</p>
        {metrics.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {metrics.map((metric) => <Badge key={metric} variant="outline">{metric}</Badge>)}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function FeaturePredictionPanel({
  company,
  events,
  onOpenPredictions,
}: {
  company: BatchCompany;
  events: AnalysisEvent[];
  onOpenPredictions: () => void;
}) {
  const prediction = company.prediction;
  const decisionState = company.progress?.agents?.decision?.state;
  const featureActivity = latestEvent(events, [
    "rag_features_started",
    "rag_features_progress",
    "rag_features_completed",
    "rag_features_failed",
    "features_started",
    "features_progress",
  ]);
  const featureStarted = latestEvent(events, ["rag_features_started", "features_started"]);
  const featureProgressEvent = latestEvent(events, ["features_progress"]);
  const featureTerminal = latestEvent(events, ["features_completed", "features_failed"]);
  const featureCompleted = featureTerminal?.stage === "features_completed" ? featureTerminal : undefined;
  const featureFailed = featureTerminal?.stage === "features_failed" ? featureTerminal : undefined;
  const predictionStarted = latestEvent(events, ["prediction_started"]);
  const predictionTerminal = latestEvent(events, ["prediction_completed", "prediction_failed"]);
  const predictionCompleted = predictionTerminal?.stage === "prediction_completed" ? predictionTerminal : undefined;
  const predictionFailed = predictionTerminal?.stage === "prediction_failed" ? predictionTerminal : undefined;
  const visible = Boolean(
    prediction
    || featureActivity
    || featureStarted
    || featureCompleted
    || featureFailed
    || predictionStarted
    || ["complete", "degraded"].includes(decisionState ?? ""),
  );
  if (!visible) return null;

  const marker = progressMarker(featureProgressEvent?.detail);
  const featuresState: AutomaticStageState = featureFailed && !prediction
    ? "failed"
    : featureCompleted || predictionStarted || prediction
      ? "complete"
      : featureActivity || featureStarted
        ? "running"
        : "waiting";
  const predictionState: AutomaticStageState = predictionFailed
    ? "failed"
    : prediction || predictionCompleted
      ? "complete"
      : predictionStarted
        ? "running"
        : "waiting";
  const featurePercent = featuresState === "complete" || featuresState === "failed"
    ? 100
    : marker
      ? marker.current / marker.total * 100
      : featuresState === "running" ? 5 : 0;
  const predictionPercent = predictionState === "complete" || predictionState === "failed"
    ? 100
    : predictionState === "running" ? 55 : 0;
  const collectedCount = prediction?.candidate_feature_count
    ?? candidateCount(featureCompleted?.detail);
  return (
    <div className="flex flex-col gap-4">
      <Card className="bg-card/75">
        <CardHeader className="border-b border-border/60">
          <CardTitle>上市前特征与多时间窗口预测</CardTitle>
          <CardDescription>决策Agent结束后自动执行，分别生成上市首日、5日、20日和60日风险预测。</CardDescription>
          <CardAction><Badge variant="outline">自动流程</Badge></CardAction>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-2">
          <AutomaticStageCard
            kind="features"
            title="检索并采集上市前宽候选特征"
            description="本公司向量库、表格公式、上市前市场与Agent风险"
            state={featuresState}
            progress={featurePercent}
            progressText={marker ? `${marker.current}/${marker.total} · ${marker.detail}` : automaticStageLabel[featuresState]}
            detail={featureFailed?.detail ?? featureCompleted?.detail ?? featureActivity?.detail ?? featureStarted?.detail ?? "等待决策Agent完成后自动开始。"}
            metrics={[
              collectedCount != null ? `候选特征 ${collectedCount} 项` : "",
            ].filter(Boolean)}
          />
          <AutomaticStageCard
            kind="prediction"
            title="多时间窗口风险预测"
            description="使用四个独立模型生成不同时间窗口的方向性预警"
            state={predictionState}
            progress={predictionPercent}
            progressText={predictionState === "running" ? "正在校验并计算" : automaticStageLabel[predictionState]}
            detail={cleanDetail(predictionFailed?.detail ?? predictionCompleted?.detail ?? predictionStarted?.detail ?? "候选特征完成后自动校验当前模型输入。")}
            metrics={prediction ? ["1日、5日、20日和60日预测已生成"] : []}
          />
        </CardContent>
      </Card>

      {prediction && (
        <Alert>
          <ChartNoAxesCombined />
          <AlertTitle>多时间窗口预测已有可用结果</AlertTitle>
          <AlertDescription>四个时间窗口的预警状态和主要风险已移至独立预测页。</AlertDescription>
          <AlertAction>
            <Button variant="outline" size="sm" onClick={onOpenPredictions}>查看风险预测</Button>
          </AlertAction>
        </Alert>
      )}
    </div>
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
  onOpenPredictions,
}: {
  batch: BatchSnapshot;
  company: BatchCompany;
  events: AnalysisEvent[];
  visibleEvents: AnalysisEvent[];
  nowMs: number;
  serviceAvailable: boolean;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setError: (value: string | null) => void;
  onOpenPredictions: () => void;
}) {
  const currentStage = companyStage(company, events);
  const pipeline = ["文档登记", "解析与索引", "风险发现", "范围确认", "专业核验", "决策整合", "特征生成", "风险预测", "完成"];
  const percent = statusProgress(company);
  const elapsed = elapsedSeconds(company, nowMs);
  const debateRecords = useMemo(() => parseDebateRecords(events), [events]);
  const debateActive = company.state !== "completed" && company.state !== "failed";

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
        <CardContent className="flex min-w-[1120px] items-center gap-2">
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

      <LlmRecoveryPanel
        key={company.llm_failure_recovery?.request_id ?? `${company.company_key}-llm-recovery`}
        batch={batch}
        company={company}
        serviceAvailable={serviceAvailable}
        onSnapshot={onSnapshot}
        setError={setError}
      />

      <StageNotice company={company} events={events} />

      <FeaturePredictionPanel company={company} events={events} onOpenPredictions={onOpenPredictions} />

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
                <TabsTrigger value="debates"><MessageSquareWarning data-icon="inline-start" />辩论与补证</TabsTrigger>
                <TabsTrigger value="events"><Activity data-icon="inline-start" />事件流</TabsTrigger>
              </TabsList>
            </div>
            {roleMeta.map(([key]) => (
              <TabsContent key={key} value={key}>
                <AgentOutputPane role={key} events={events} company={company} />
              </TabsContent>
            ))}
            <TabsContent value="debates">
              <DebateTimeline records={debateRecords} active={debateActive} />
            </TabsContent>
            <TabsContent value="events">
              <EventLog events={visibleEvents} />
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
  onOpenPredictions,
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
          <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
            <MetricCard label="公司总数" value={batch.company_count} helper="当前工作区" />
            <MetricCard label="分析中" value={(batch.counts.running ?? 0) + (batch.counts.preparing ?? 0)} helper="独立实时更新" />
            <MetricCard label="待确认风险" value={batch.counts.awaiting_review ?? 0} helper="可逐公司放行" />
            <MetricCard label="待处理异常" value={batch.counts.awaiting_recovery ?? 0} helper="重试或BM25降级" />
            <MetricCard label="已完成" value={batch.counts.completed ?? 0} helper="结果持续保留" />
            <MetricCard label="异常" value={batch.counts.failed ?? 0} helper="支持恢复" />
          </section>

          <Card size="sm" className="bg-card/75">
            <CardHeader>
              <CardTitle>批次总体进度</CardTitle>
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
                onOpenPredictions={onOpenPredictions}
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
