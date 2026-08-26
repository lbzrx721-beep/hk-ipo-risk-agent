import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Bot,
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

import { listReviewRuns, rerunPrediction, rerunStoredPrediction } from "@/api";
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
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { stateLabel } from "@/lib/presentation";
import type {
  BatchCompany,
  BatchSnapshot,
  ReviewRun,
  RuntimePrediction,
} from "@/types";

type PredictionsWorkspaceProps = {
  batch: BatchSnapshot | null;
  selected: BatchCompany | null;
  selectedKey: string | null;
  busy: boolean;
  serviceAvailable: boolean;
  onSelectCompany: (companyKey: string) => void;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setBusy: (busy: boolean) => void;
  setError: (message: string | null) => void;
  onOpenResults: () => void;
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
  baseFeatureAvailable: boolean;
  agentFeatureAvailable: boolean;
  basePrediction?: RuntimePrediction | null;
  prediction?: RuntimePrediction | null;
  basePredictionError?: string | null;
  predictionError?: string | null;
};

function percent(value: number | undefined): string {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function riskLabel(prediction: RuntimePrediction): string {
  if (prediction.decision === "high_risk") return "显著下跌预警";
  if (prediction.recall_guard?.watchlist) return "风险观察名单";
  if (prediction.decision === "low_risk") return "低风险";
  return "不确定 · 待研判";
}

function riskBadge(prediction: RuntimePrediction): "destructive" | "secondary" | "outline" {
  if (prediction.decision === "high_risk") return "destructive";
  if (prediction.recall_guard?.watchlist) return "outline";
  if (prediction.decision === "low_risk") return "secondary";
  return "outline";
}

function PredictionResultCard({
  title,
  description,
  prediction,
  pendingText,
  accent,
}: {
  title: string;
  description: string;
  prediction?: RuntimePrediction | null;
  pendingText: string;
  accent: "base" | "agent";
}) {
  if (!prediction) {
    return (
      <Card className="min-h-72 border-dashed bg-card/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {accent === "base" ? (
              <ChartNoAxesCombined className="size-4 text-primary" />
            ) : (
              <Bot className="size-4 text-primary" />
            )}
            {title}
          </CardTitle>
          <CardDescription>{description}</CardDescription>
          <CardAction>
            <Badge variant="outline"><Clock3 data-icon="inline-start" />等待输入</Badge>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Empty className="min-h-44 border">
            <EmptyHeader>
              <EmptyMedia variant="icon"><Clock3 /></EmptyMedia>
              <EmptyTitle>{pendingText}</EmptyTitle>
              <EmptyDescription>已有特征可直接重算，不会重新解析PDF或调用Agent。</EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={accent === "agent" ? "border-primary/30 bg-primary/5" : "bg-card/75"}>
      <CardHeader className="border-b border-border/60">
        <CardTitle className="flex items-center gap-2">
          {accent === "base" ? (
            <ChartNoAxesCombined className="size-4 text-primary" />
          ) : (
            <Sparkles className="size-4 text-primary" />
          )}
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction><Badge variant={riskBadge(prediction)}>{riskLabel(prediction)}</Badge></CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-xl border bg-background/55 p-4">
            <p className="text-xs text-muted-foreground">五日显著下跌概率</p>
            <strong className="mt-2 block text-3xl tabular-nums">{percent(prediction.probability)}</strong>
          </div>
          <div className="rounded-xl border bg-background/55 p-4">
            <p className="text-xs text-muted-foreground">方向置信度</p>
            <strong className="mt-2 block text-3xl tabular-nums">{percent(prediction.confidence)}</strong>
          </div>
          <div className="rounded-xl border bg-background/55 p-4">
            <p className="text-xs text-muted-foreground">模型输入完整度</p>
            <strong className="mt-2 block text-3xl tabular-nums">{percent(prediction.feature_completeness)}</strong>
          </div>
        </div>
        <Progress value={prediction.feature_completeness * 100} aria-label={`${title}输入完整度`} />
        {prediction.recall_guard?.enabled && prediction.recall_guard.watchlist && (
          <Alert className="border-amber-500/35 bg-amber-500/10 text-amber-50">
            <ShieldAlert />
            <AlertTitle>高风险召回保护层已命中</AlertTitle>
            <AlertDescription>
              主模型结论不变；该公司因
              {prediction.recall_guard.reasons.join("、") || "多项稳定风险因子叠加"}
              进入风险观察名单，请结合原文证据核验。
            </AlertDescription>
          </Alert>
        )}
        {prediction.key_inputs.length > 0 && (
          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">本次主要输入</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {prediction.key_inputs.map((item) => (
                <div
                  key={item.feature}
                  className="flex items-center justify-between gap-4 rounded-lg border bg-background/45 px-3 py-2"
                >
                  <span className="truncate text-sm text-muted-foreground">{item.label}</span>
                  <strong className="shrink-0 text-sm tabular-nums">
                    {String(item.value ?? "缺失")}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        )}
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">{prediction.model_name}</Badge>
          <Badge variant="outline">
            输入 {prediction.available_feature_count}/{prediction.feature_count}
          </Badge>
          {prediction.recall_guard?.enabled && (
            <Badge variant="outline">
              召回保护 {prediction.recall_guard.available_feature_count ?? 0}/
              {prediction.recall_guard.feature_count ?? 0}
            </Badge>
          )}
          <Badge variant="outline">{new Date(prediction.generated_at).toLocaleString("zh-CN")}</Badge>
        </div>
        {prediction.warnings.length > 0 && (
          <Alert>
            <FileWarning />
            <AlertTitle>输入与使用提示</AlertTitle>
            <AlertDescription>{prediction.warnings.join("；")}</AlertDescription>
          </Alert>
        )}
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
    baseFeatureAvailable: Boolean(company.base_feature_path),
    agentFeatureAvailable: Boolean(company.feature_path),
    basePrediction: company.base_prediction,
    prediction: company.prediction,
    basePredictionError: company.base_prediction_error,
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
    baseFeatureAvailable: run.base_feature_available,
    agentFeatureAvailable: run.agent_feature_available,
    basePrediction: run.base_prediction,
    prediction: run.prediction,
  };
}

export function PredictionsWorkspace({
  batch,
  selected: _selected,
  selectedKey,
  busy,
  serviceAvailable,
  onSelectCompany,
  onSnapshot,
  setBusy,
  setError,
  onOpenResults,
}: PredictionsWorkspaceProps) {
  const [historyRuns, setHistoryRuns] = useState<ReviewRun[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [scope, setScope] = useState("all");
  const [query, setQuery] = useState("");
  const [localSelectedKey, setLocalSelectedKey] = useState<string | null>(null);

  const refreshHistory = async () => {
    if (!serviceAvailable) return;
    setHistoryLoading(true);
    try {
      setHistoryRuns(await listReviewRuns());
    } catch (error) {
      setError(error instanceof Error ? error.message : "历史预测记录加载失败。");
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    void refreshHistory();
  }, [serviceAvailable]);

  const entries = useMemo(() => {
    const current = (batch?.companies ?? []).map(currentEntry);
    const currentRunIds = new Set(current.map((item) => item.runId).filter(Boolean));
    const history = historyRuns
      .filter((run) => !currentRunIds.has(run.run_id))
      .map(historyEntry);
    return [...current, ...history];
  }, [batch, historyRuns]);

  const visibleEntries = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => {
      if (scope !== "all" && entry.source !== scope) return false;
      if (!needle) return true;
      return `${entry.companyId} ${entry.companyName} ${entry.runId ?? ""}`
        .toLowerCase()
        .includes(needle);
    });
  }, [entries, query, scope]);

  useEffect(() => {
    if (!selectedKey) return;
    const key = `current:${selectedKey}`;
    if (entries.some((entry) => entry.key === key)) setLocalSelectedKey(key);
  }, [entries, selectedKey]);

  useEffect(() => {
    if (visibleEntries.some((entry) => entry.key === localSelectedKey)) return;
    setLocalSelectedKey(visibleEntries[0]?.key ?? null);
  }, [localSelectedKey, visibleEntries]);

  const selectedEntry =
    visibleEntries.find((entry) => entry.key === localSelectedKey)
    ?? visibleEntries[0]
    ?? null;

  const selectEntry = (entry: PredictionEntry) => {
    setLocalSelectedKey(entry.key);
    if (entry.source === "current" && entry.companyKey) onSelectCompany(entry.companyKey);
  };

  const runPrediction = async (variant: "base" | "agent_enhanced") => {
    if (!selectedEntry) return;
    setBusy(true);
    try {
      if (selectedEntry.source === "current") {
        if (!batch || !selectedEntry.companyKey) return;
        const result = await rerunPrediction(batch.batch_id, selectedEntry.companyKey, variant);
        onSnapshot(result.batch);
      } else if (selectedEntry.runId) {
        const result = await rerunStoredPrediction(selectedEntry.runId, variant);
        setHistoryRuns((runs) => runs.map((run) => (
          run.run_id === result.run.run_id ? result.run : run
        )));
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "五日预测未完成。");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-[1920px] flex-col gap-5 px-4 py-6 lg:px-6 2xl:px-8">
      <section className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
        <div>
          <Badge variant="outline" className="mb-3">Day-5 downside intelligence</Badge>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">上市后五日风险预测</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            可选择任意历史分析直接预测；复用已保存特征，不重新解析PDF，也不调用Agent。
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            disabled={historyLoading || !serviceAvailable}
            onClick={() => void refreshHistory()}
          >
            <RefreshCw
              data-icon="inline-start"
              className={historyLoading ? "animate-spin" : ""}
            />
            刷新历史记录
          </Button>
          {selectedEntry && (
            <>
              <Button
                variant="outline"
                disabled={busy || !serviceAvailable || !selectedEntry.baseFeatureAvailable}
                onClick={() => void runPrediction("base")}
              >
                <ChartNoAxesCombined data-icon="inline-start" />重算基础预测
              </Button>
              <Button
                disabled={busy || !serviceAvailable || !selectedEntry.agentFeatureAvailable}
                onClick={() => void runPrediction("agent_enhanced")}
              >
                <Sparkles data-icon="inline-start" />重算增强预测
              </Button>
            </>
          )}
        </div>
      </section>

      {entries.length === 0 ? (
        <Empty className="min-h-[520px] border">
          <EmptyHeader>
            <EmptyMedia variant="icon"><ChartNoAxesCombined /></EmptyMedia>
            <EmptyTitle>{historyLoading ? "正在读取历史分析" : "还没有可预测的公司"}</EmptyTitle>
            <EmptyDescription>
              已完成的分析会自动出现在这里；新上传招股书也可以从分析工作台继续进入。
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <div className="grid min-h-[620px] gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
          <Card className="bg-card/70">
            <CardHeader className="border-b border-border/60">
              <CardTitle>可预测公司</CardTitle>
              <CardDescription>{entries.length} 条运行 · 历史结果与当前批次统一选择</CardDescription>
            </CardHeader>
            <CardContent className="flex min-h-0 flex-1 flex-col gap-3 p-3">
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-2 size-4 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="搜索公司、代码或运行编号"
                  className="pl-8"
                />
              </div>
              <Tabs value={scope} onValueChange={setScope}>
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="all">全部</TabsTrigger>
                  <TabsTrigger value="current">当前批次</TabsTrigger>
                  <TabsTrigger value="history">历史分析</TabsTrigger>
                </TabsList>
              </Tabs>
              <ScrollArea className="h-[490px]">
                <div className="flex flex-col gap-2 pr-3">
                  {visibleEntries.map((entry) => {
                    const ready = Boolean(entry.prediction || entry.basePrediction);
                    return (
                      <Button
                        key={entry.key}
                        variant={selectedEntry?.key === entry.key ? "secondary" : "ghost"}
                        className="h-auto justify-start px-3 py-3 text-left"
                        onClick={() => selectEntry(entry)}
                      >
                        <span className="flex min-w-0 flex-1 flex-col gap-1">
                          <strong className="truncate">
                            {entry.companyId} · {entry.companyName}
                          </strong>
                          <span className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
                            {entry.source === "history" ? (
                              <History className="size-3" />
                            ) : (
                              <Database className="size-3" />
                            )}
                            <span>{entry.source === "history" ? "历史分析" : "当前批次"}</span>
                            <span>·</span>
                            <span>{ready ? "预测已就绪" : "特征已保存"}</span>
                          </span>
                        </span>
                        {ready ? (
                          <CheckCircle2 className="size-4 shrink-0 text-emerald-500" />
                        ) : (
                          <Clock3 className="size-4 shrink-0 text-muted-foreground" />
                        )}
                      </Button>
                    );
                  })}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {selectedEntry ? (
            <div className="flex min-w-0 flex-col gap-4">
              <Card size="sm" className="bg-card/70">
                <CardHeader>
                  <CardTitle>{selectedEntry.companyId} · {selectedEntry.companyName}</CardTitle>
                  <CardDescription>
                    招股书日期 {selectedEntry.asOfDate || "待识别"}
                    {selectedEntry.runId ? ` · ${selectedEntry.runId}` : ""}
                  </CardDescription>
                  <CardAction>
                    <Badge variant="outline">
                      {selectedEntry.source === "history" ? "历史特征复用" : selectedEntry.status}
                    </Badge>
                  </CardAction>
                </CardHeader>
              </Card>

              {selectedEntry.basePredictionError && !selectedEntry.basePrediction && (
                <Alert variant="destructive">
                  <ShieldAlert />
                  <AlertTitle>快速预测未完成</AlertTitle>
                  <AlertDescription>{selectedEntry.basePredictionError}</AlertDescription>
                </Alert>
              )}
              {selectedEntry.predictionError && !selectedEntry.prediction && (
                <Alert variant="destructive">
                  <ShieldAlert />
                  <AlertTitle>Agent增强预测未完成</AlertTitle>
                  <AlertDescription>{selectedEntry.predictionError}</AlertDescription>
                </Alert>
              )}

              <div className="grid gap-4 2xl:grid-cols-2">
                <PredictionResultCard
                  title="快速预测"
                  description="使用发行、财务、文档和上市前市场输入。"
                  prediction={selectedEntry.basePrediction}
                  pendingText={
                    selectedEntry.baseFeatureAvailable
                      ? "基础特征已就绪，可立即预测"
                      : "暂无基础特征"
                  }
                  accent="base"
                />
                <PredictionResultCard
                  title="Agent增强预测"
                  description="融合风险发现、专业核验和决策裁决后的结构化特征。"
                  prediction={selectedEntry.prediction}
                  pendingText={
                    selectedEntry.agentFeatureAvailable
                      ? "Agent特征已就绪，可立即预测"
                      : "暂无Agent增强特征"
                  }
                  accent="agent"
                />
              </div>

              {selectedEntry.basePrediction && selectedEntry.prediction && (
                <Card className="border-primary/25 bg-primary/5">
                  <CardHeader>
                    <CardTitle>两阶段结果对照</CardTitle>
                    <CardDescription>Agent风险特征对基础判断的影响会在这里明确显示。</CardDescription>
                    <CardAction>
                      <Badge variant="secondary">
                        <ArrowRight data-icon="inline-start" />已完成对照
                      </Badge>
                    </CardAction>
                  </CardHeader>
                  <CardContent className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-xl border bg-background/55 p-4">
                      <p className="text-xs text-muted-foreground">基础概率</p>
                      <strong className="mt-2 block text-2xl">
                        {percent(selectedEntry.basePrediction.probability)}
                      </strong>
                    </div>
                    <div className="rounded-xl border bg-background/55 p-4">
                      <p className="text-xs text-muted-foreground">增强概率</p>
                      <strong className="mt-2 block text-2xl">
                        {percent(selectedEntry.prediction.probability)}
                      </strong>
                    </div>
                    <div className="rounded-xl border bg-background/55 p-4">
                      <p className="text-xs text-muted-foreground">Agent影响</p>
                      <strong className="mt-2 block text-2xl">
                        {selectedEntry.prediction.probability >= selectedEntry.basePrediction.probability ? "+" : ""}
                        {Math.round((selectedEntry.prediction.probability - selectedEntry.basePrediction.probability) * 100)} 个百分点
                      </strong>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Alert>
                <ShieldAlert />
                <AlertTitle>预测与正式报告分工</AlertTitle>
                <AlertDescription>
                  本页只做方向性五日预警。证据确认、人工复核与正式报告导出仍在“结果与复核”完成。
                </AlertDescription>
                <Button variant="outline" size="sm" onClick={onOpenResults}>
                  前往结果与复核<ArrowRight data-icon="inline-end" />
                </Button>
              </Alert>
            </div>
          ) : (
            <Empty className="min-h-[620px] border">
              <EmptyHeader>
                <EmptyMedia variant="icon"><Search /></EmptyMedia>
                <EmptyTitle>没有匹配的分析记录</EmptyTitle>
                <EmptyDescription>调整筛选条件或搜索关键词后重试。</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
      )}
    </div>
  );
}
