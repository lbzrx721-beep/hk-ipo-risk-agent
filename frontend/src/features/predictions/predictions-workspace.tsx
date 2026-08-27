import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  ChartNoAxesCombined,
  CheckCircle2,
  Clock3,
  Database,
  FileWarning,
  History,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

import { listReviewRuns, listRunFindings, rerunPrediction, rerunStoredPrediction } from "@/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { severityLabel, stateLabel } from "@/lib/presentation";
import { cn } from "@/lib/utils";
import type { BatchCompany, BatchSnapshot, ReviewRun, RuntimePrediction, StoredFinding } from "@/types";

type PredictionsWorkspaceProps = {
  batch: BatchSnapshot | null;
  selectedKey: string | null;
  busy: boolean;
  serviceAvailable: boolean;
  initialRunId?: string | null;
  onSelectCompany: (companyKey: string) => void;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setBusy: (busy: boolean) => void;
  setError: (message: string | null) => void;
  onSelectionChange?: (runId: string | null) => void;
  onOpenResults: (runId?: string | null, findingId?: string | null) => void;
};

type PredictionEntry = {
  key: string;
  source: "current" | "history";
  runId?: string | null;
  companyKey?: string;
  companyId: string;
  companyName: string;
  asOfDate: string;
  status: string;
  featureAvailable: boolean;
  prediction?: RuntimePrediction | null;
  predictionError?: string | null;
};

const severityRank: Record<StoredFinding["severity"], number> = { critical: 4, high: 3, medium: 2, low: 1 };
const severityTone: Record<StoredFinding["severity"], string> = {
  critical: "border-l-red-500 bg-red-500/5",
  high: "border-l-orange-500 bg-orange-500/5",
  medium: "border-l-amber-400 bg-amber-400/5",
  low: "border-l-emerald-500 bg-emerald-500/5",
};
const agentLabel: Record<string, string> = {
  finance: "财务 Agent",
  legal: "法律 Agent",
  business: "业务 Agent",
  market: "市场 Agent",
  decision: "决策 Agent",
  discovery: "风险发现 Agent",
};

function percent(value: number | undefined): string {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function riskLabel(prediction: RuntimePrediction): string {
  if (prediction.decision === "high_risk") return "显著下跌预警";
  if (prediction.recall_guard?.watchlist) return "风险观察名单";
  if (prediction.decision === "low_risk") return "低风险";
  return "边界信号 · 暂不预警";
}

function riskBadge(prediction: RuntimePrediction): "destructive" | "secondary" | "outline" {
  if (prediction.decision === "high_risk") return "destructive";
  if (prediction.decision === "low_risk") return "secondary";
  return "outline";
}

function highestRisks(findings: StoredFinding[]): StoredFinding[] {
  return [...findings]
    .sort((left, right) => severityRank[right.severity] - severityRank[left.severity]
      || right.confidence - left.confidence
      || right.evidence_ids.length - left.evidence_ids.length)
    .slice(0, 5);
}

function PredictionResultCard({ prediction, pendingText, findings, findingsLoading, onOpenFinding }: {
  prediction?: RuntimePrediction | null;
  pendingText: string;
  findings: StoredFinding[];
  findingsLoading: boolean;
  onOpenFinding: (findingId?: string) => void;
}) {
  if (!prediction) {
    return (
      <Card className="min-h-80 border-dashed bg-card/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-primary" />统一五日风险预测</CardTitle>
          <CardDescription>融合发行、财务、市场与多 Agent 风险特征。</CardDescription>
          <CardAction><Badge variant="outline"><Clock3 data-icon="inline-start" />等待预测</Badge></CardAction>
        </CardHeader>
        <CardContent>
          <Empty className="min-h-52 border">
            <EmptyHeader>
              <EmptyMedia variant="icon"><Clock3 /></EmptyMedia>
              <EmptyTitle>{pendingText}</EmptyTitle>
              <EmptyDescription>重算会复用已保存特征，不会重新解析 PDF 或调用 Agent。</EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      </Card>
    );
  }

  const topRisks = highestRisks(findings);
  return (
    <Card className="overflow-hidden border-primary/30 bg-gradient-to-br from-primary/8 via-card to-card">
      <CardHeader className="border-b border-border/60">
        <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-primary" />统一五日风险预测</CardTitle>
        <CardDescription>最终模型给出方向性预警，风险卡片用于解释公司的主要风险背景。</CardDescription>
        <CardAction><Badge variant={riskBadge(prediction)}>{riskLabel(prediction)}</Badge></CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <section className="grid gap-4 rounded-2xl border bg-background/50 p-5 md:grid-cols-[220px_minmax(0,1fr)] md:items-center">
          <div>
            <p className="text-sm text-muted-foreground">上市后五日显著下跌概率</p>
            <strong className="mt-2 block text-5xl tabular-nums tracking-tight">{percent(prediction.probability)}</strong>
            <p className="mt-2 text-xs text-muted-foreground">模型置信度 {percent(prediction.confidence)}</p>
          </div>
          <div className="flex flex-col gap-3">
            <div className="flex justify-between text-xs text-muted-foreground"><span>低风险</span><span>高风险</span></div>
            <Progress value={prediction.probability * 100} aria-label="五日显著下跌概率" className="h-3" />
            <p className="text-sm leading-6 text-muted-foreground">该概率用于风险排序和预警，不代表收益率预测；正式结论仍需结合原文证据与人工复核。</p>
          </div>
        </section>

        <section className="flex flex-col gap-3">
          <div className="flex items-end justify-between gap-3">
            <div><h3 className="font-semibold">该公司最高风险</h3><p className="mt-1 text-xs text-muted-foreground">按严重程度、置信度和证据数量排序，展示前 5 条。</p></div>
            {findings.length > 0 && <Badge variant="outline">共 {findings.length} 条风险</Badge>}
          </div>
          {findingsLoading ? (
            <div className="grid gap-3 lg:grid-cols-2">{[0, 1, 2, 3].map((item) => <Skeleton key={item} className="h-36 rounded-xl" />)}</div>
          ) : topRisks.length > 0 ? (
            <div className="grid gap-3 lg:grid-cols-2">
              {topRisks.map((finding) => (
                <Card key={finding.finding_id} size="sm" className={cn("border-l-4", severityTone[finding.severity])}>
                  <CardHeader>
                    <CardTitle className="line-clamp-2 text-base">{finding.title}</CardTitle>
                    <CardDescription>{agentLabel[finding.created_by] ?? `${finding.created_by} Agent`}</CardDescription>
                    <CardAction><Badge variant={finding.severity === "critical" || finding.severity === "high" ? "destructive" : "secondary"}>{severityLabel[finding.severity]}</Badge></CardAction>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-3">
                    <p className="line-clamp-3 text-sm leading-6 text-foreground/85">{finding.claim}</p>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="outline">置信度 {percent(finding.confidence)}</Badge>
                      <Badge variant="outline">证据 {finding.evidence_ids.length} 条</Badge>
                      <Button variant="link" size="sm" className="ml-auto h-auto p-0" onClick={() => onOpenFinding(finding.finding_id)}>查看证据<ArrowRight data-icon="inline-end" /></Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Empty className="min-h-44 border">
              <EmptyHeader><EmptyMedia variant="icon"><ShieldAlert /></EmptyMedia><EmptyTitle>暂未读取到风险卡片</EmptyTitle><EmptyDescription>预测结果仍可使用；风险证据将在对应分析运行保存后显示。</EmptyDescription></EmptyHeader>
            </Empty>
          )}
        </section>

        <div className="flex flex-wrap items-center gap-2 border-t pt-4 text-xs text-muted-foreground">
          <Badge variant="outline">{prediction.model_name}</Badge>
          <Badge variant="outline">{new Date(prediction.generated_at).toLocaleString("zh-CN")}</Badge>
        </div>
        {prediction.warnings.length > 0 && <Alert><FileWarning /><AlertTitle>预测提示</AlertTitle><AlertDescription>{prediction.warnings.join("；")}</AlertDescription></Alert>}
      </CardContent>
    </Card>
  );
}

function currentEntry(company: BatchCompany): PredictionEntry {
  return {
    key: `current:${company.company_key}`,
    source: "current",
    runId: company.run_id,
    companyKey: company.company_key,
    companyId: company.company_id,
    companyName: company.company_name,
    asOfDate: company.prospectus_date,
    status: stateLabel[company.state] ?? company.state,
    featureAvailable: Boolean(company.feature_path),
    prediction: company.prediction,
    predictionError: company.prediction_error,
  };
}

function historyEntry(run: ReviewRun): PredictionEntry {
  return {
    key: `history:${run.run_id}`,
    source: "history",
    runId: run.run_id,
    companyId: run.company_id,
    companyName: run.company_name,
    asOfDate: run.as_of_date,
    status: "历史分析已完成",
    featureAvailable: run.agent_feature_available,
    prediction: run.prediction,
  };
}

export function PredictionsWorkspace({ batch, selectedKey, busy, serviceAvailable, initialRunId, onSelectCompany, onSnapshot, setBusy, setError, onSelectionChange, onOpenResults }: PredictionsWorkspaceProps) {
  const [historyRuns, setHistoryRuns] = useState<ReviewRun[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [scope, setScope] = useState("all");
  const [query, setQuery] = useState("");
  const [localSelectedKey, setLocalSelectedKey] = useState<string | null>(null);
  const [findingsByRun, setFindingsByRun] = useState<Record<string, StoredFinding[]>>({});
  const [findingsLoadingRunId, setFindingsLoadingRunId] = useState<string | null>(null);

  const refreshHistory = useCallback(async () => {
    if (!serviceAvailable) return;
    setHistoryLoading(true);
    try { setHistoryRuns(await listReviewRuns()); }
    catch (error) { setError(error instanceof Error ? error.message : "历史预测记录加载失败。"); }
    finally { setHistoryLoading(false); }
  }, [serviceAvailable, setError]);

  useEffect(() => { void refreshHistory(); }, [refreshHistory]);

  const entries = useMemo(() => {
    const current = (batch?.companies ?? []).map(currentEntry);
    const currentRunIds = new Set(current.map((item) => item.runId).filter(Boolean));
    return [...current, ...historyRuns.filter((run) => !currentRunIds.has(run.run_id)).map(historyEntry)];
  }, [batch, historyRuns]);

  const visibleEntries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => (scope === "all" || entry.source === scope)
      && (!needle || `${entry.companyId} ${entry.companyName} ${entry.runId ?? ""}`.toLowerCase().includes(needle)));
  }, [entries, query, scope]);

  useEffect(() => {
    if (!initialRunId) return;
    const match = entries.find((entry) => entry.runId === initialRunId);
    if (match) setLocalSelectedKey(match.key);
  }, [entries, initialRunId]);

  useEffect(() => {
    if (!selectedKey || initialRunId) return;
    const key = `current:${selectedKey}`;
    if (entries.some((entry) => entry.key === key)) setLocalSelectedKey(key);
  }, [entries, initialRunId, selectedKey]);

  useEffect(() => {
    if (!visibleEntries.some((entry) => entry.key === localSelectedKey)) setLocalSelectedKey(visibleEntries[0]?.key ?? null);
  }, [localSelectedKey, visibleEntries]);

  const selectedEntry = visibleEntries.find((entry) => entry.key === localSelectedKey) ?? visibleEntries[0] ?? null;
  const selectedRunId = selectedEntry?.runId ?? null;

  useEffect(() => { onSelectionChange?.(selectedRunId); }, [onSelectionChange, selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || findingsByRun[selectedRunId]) return;
    let active = true;
    setFindingsLoadingRunId(selectedRunId);
    void listRunFindings(selectedRunId)
      .then((rows) => active && setFindingsByRun((current) => ({ ...current, [selectedRunId]: rows })))
      .catch(() => active && setFindingsByRun((current) => ({ ...current, [selectedRunId]: [] })))
      .finally(() => active && setFindingsLoadingRunId((current) => current === selectedRunId ? null : current));
    return () => { active = false; };
  }, [findingsByRun, selectedRunId]);

  const selectEntry = (entry: PredictionEntry) => {
    setLocalSelectedKey(entry.key);
    if (entry.source === "current" && entry.companyKey) onSelectCompany(entry.companyKey);
  };

  const runPrediction = async () => {
    if (!selectedEntry) return;
    setBusy(true);
    try {
      if (selectedEntry.source === "current") {
        if (!batch || !selectedEntry.companyKey) return;
        onSnapshot((await rerunPrediction(batch.batch_id, selectedEntry.companyKey, "agent_enhanced")).batch);
      } else if (selectedEntry.runId) {
        const result = await rerunStoredPrediction(selectedEntry.runId, "agent_enhanced");
        setHistoryRuns((runs) => runs.map((run) => run.run_id === result.run.run_id ? result.run : run));
      }
    } catch (error) { setError(error instanceof Error ? error.message : "五日预测未完成。"); }
    finally { setBusy(false); }
  };

  return (
    <div className="mx-auto flex w-full max-w-[1920px] flex-col gap-5 px-4 py-6 lg:px-6 2xl:px-8">
      <section className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div><Badge variant="outline" className="mb-3">Day-5 downside intelligence</Badge><h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">上市后五日风险预测</h1><p className="mt-2 text-sm text-muted-foreground">选择任意已分析公司，复用保存的特征直接运行统一最终模型。</p></div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" disabled={historyLoading || !serviceAvailable} onClick={() => void refreshHistory()}><RefreshCw data-icon="inline-start" className={historyLoading ? "animate-spin" : ""} />刷新历史记录</Button>
          {selectedEntry && <Button disabled={busy || !serviceAvailable || !selectedEntry.featureAvailable} onClick={() => void runPrediction()}><Sparkles data-icon="inline-start" />重新运行五日预测</Button>}
        </div>
      </section>

      {entries.length === 0 ? (
        <Empty className="min-h-[520px] border"><EmptyHeader><EmptyMedia variant="icon"><ChartNoAxesCombined /></EmptyMedia><EmptyTitle>{historyLoading ? "正在读取历史分析" : "还没有可预测的公司"}</EmptyTitle><EmptyDescription>完成分析并保存特征后，公司会自动出现在这里。</EmptyDescription></EmptyHeader></Empty>
      ) : (
        <div className="grid min-h-[620px] gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
          <Card className="bg-card/70">
            <CardHeader className="border-b border-border/60"><CardTitle>可预测公司</CardTitle><CardDescription>{entries.length} 条运行 · 当前批次与历史分析统一选择</CardDescription></CardHeader>
            <CardContent className="flex min-h-0 flex-1 flex-col gap-3 p-3">
              <div className="relative"><Search className="pointer-events-none absolute left-2.5 top-2 size-4 text-muted-foreground" /><Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索公司、代码或运行编号" className="pl-8" /></div>
              <Tabs value={scope} onValueChange={setScope}><TabsList className="grid w-full grid-cols-3"><TabsTrigger value="all">全部</TabsTrigger><TabsTrigger value="current">当前批次</TabsTrigger><TabsTrigger value="history">历史分析</TabsTrigger></TabsList></Tabs>
              <ScrollArea className="h-[540px]"><div className="flex flex-col gap-2 pr-3">
                {visibleEntries.map((entry) => (
                  <Button key={entry.key} variant={selectedEntry?.key === entry.key ? "secondary" : "ghost"} className="h-auto justify-start px-3 py-3 text-left" onClick={() => selectEntry(entry)}>
                    <span className="flex min-w-0 flex-1 flex-col gap-1"><strong className="truncate">{entry.companyId} · {entry.companyName}</strong><span className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">{entry.source === "history" ? <History className="size-3" /> : <Database className="size-3" />}<span>{entry.source === "history" ? "历史分析" : "当前批次"}</span><span>·</span><span>{entry.prediction ? "预测已就绪" : entry.featureAvailable ? "特征已保存" : "等待特征"}</span></span></span>
                    {entry.prediction ? <CheckCircle2 className="size-4 shrink-0 text-emerald-500" /> : <Clock3 className="size-4 shrink-0 text-muted-foreground" />}
                  </Button>
                ))}
              </div></ScrollArea>
            </CardContent>
          </Card>

          {selectedEntry ? (
            <div className="flex min-w-0 flex-col gap-4">
              <Card size="sm" className="bg-card/70"><CardHeader><CardTitle>{selectedEntry.companyId} · {selectedEntry.companyName}</CardTitle><CardDescription>招股书日期 {selectedEntry.asOfDate || "待识别"}{selectedEntry.runId ? ` · ${selectedEntry.runId}` : ""}</CardDescription><CardAction><Badge variant="outline">{selectedEntry.source === "history" ? "历史特征复用" : selectedEntry.status}</Badge></CardAction></CardHeader></Card>
              {selectedEntry.predictionError && !selectedEntry.prediction && <Alert variant="destructive"><ShieldAlert /><AlertTitle>五日预测未完成</AlertTitle><AlertDescription>{selectedEntry.predictionError}</AlertDescription></Alert>}
              <PredictionResultCard prediction={selectedEntry.prediction} pendingText={selectedEntry.featureAvailable ? "特征已就绪，可立即预测" : "暂无统一模型所需特征"} findings={selectedRunId ? findingsByRun[selectedRunId] ?? [] : []} findingsLoading={findingsLoadingRunId === selectedRunId} onOpenFinding={(findingId) => onOpenResults(selectedRunId, findingId)} />
              <Alert><ShieldAlert /><AlertTitle>预测与正式报告分工</AlertTitle><AlertDescription>本页负责五日预警；证据确认、人工复核与报告导出在“结果与复核”完成。</AlertDescription><Button variant="outline" size="sm" onClick={() => onOpenResults(selectedRunId)}>前往结果与复核<ArrowRight data-icon="inline-end" /></Button></Alert>
            </div>
          ) : (
            <Empty className="min-h-[620px] border"><EmptyHeader><EmptyMedia variant="icon"><Search /></EmptyMedia><EmptyTitle>没有匹配的分析记录</EmptyTitle><EmptyDescription>调整筛选条件或搜索关键词后重试。</EmptyDescription></EmptyHeader></Empty>
          )}
        </div>
      )}
    </div>
  );
}
