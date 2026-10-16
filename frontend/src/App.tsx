import {
  ChartNoAxesCombined,
  Database,
  FileCheck2,
  LayoutDashboard,
  ShieldCheck,
  RefreshCw,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AnalysisWorkspace } from "@/features/workspace/analysis-workspace";
import { PredictionsWorkspace } from "@/features/predictions/predictions-workspace";
import { ReportsWorkspace } from "@/features/reports/reports-workspace";
import { LandingPage } from "@/features/landing/landing-page";
import { batchAction, checkApiHealth, connectBatchStream, getBatch } from "@/api";
import {
  ACTIVE_BATCH_KEY,
  mergeAnalysisEvents,
  viewFromPath,
  viewPath,
  type AppView,
} from "@/lib/presentation";
import type { AnalysisEvent, BatchSnapshot, PendingUpload } from "@/types";

function App() {
  const [isWelcome, setIsWelcome] = useState(() => window.location.pathname === "/");
  const [view, setView] = useState<AppView>(() => viewFromPath(window.location.pathname));
  const [visitedViews, setVisitedViews] = useState<Set<AppView>>(() => new Set([
    "workspace",
    viewFromPath(window.location.pathname),
  ]));
  const initialSearch = new URLSearchParams(window.location.search);
  const [routeRunId, setRouteRunId] = useState<string | null>(() => initialSearch.get("run_id"));
  const [routeFindingId, setRouteFindingId] = useState<string | null>(() => initialSearch.get("finding_id"));
  const [batch, setBatch] = useState<BatchSnapshot | null>(null);
  const [pending, setPending] = useState<PendingUpload[]>([]);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [eventsByCompany, setEventsByCompany] = useState<Record<string, AnalysisEvent[]>>({});
  const [connected, setConnected] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<"checking" | "online" | "offline">("checking");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkService = useCallback(async () => {
    const available = await checkApiHealth();
    setServiceStatus(available ? "online" : "offline");
    return available;
  }, []);

  const applyRoute = useCallback((nextView: AppView, runId?: string | null, findingId?: string | null, replace = false) => {
    const params = new URLSearchParams();
    if (runId) params.set("run_id", runId);
    if (findingId) params.set("finding_id", findingId);
    const href = `${viewPath[nextView]}${params.size ? `?${params.toString()}` : ""}`;
    window.history[replace ? "replaceState" : "pushState"]({}, "", href);
    setRouteRunId(runId ?? null);
    setRouteFindingId(findingId ?? null);
    setVisitedViews((current) => new Set(current).add(nextView));
    setView(nextView);
    setIsWelcome(false);
  }, []);

  const showWelcome = useCallback(() => {
    window.history.pushState({}, "", "/");
    setIsWelcome(true);
  }, []);

  const updateRouteSelection = useCallback((nextView: AppView, runId?: string | null, findingId?: string | null) => {
    if (viewFromPath(window.location.pathname) !== nextView) return;
    const params = new URLSearchParams();
    if (runId) params.set("run_id", runId);
    if (findingId) params.set("finding_id", findingId);
    const href = `${viewPath[nextView]}${params.size ? `?${params.toString()}` : ""}`;
    window.history.replaceState({}, "", href);
    setRouteRunId(runId ?? null);
    setRouteFindingId(findingId ?? null);
  }, []);

  const updatePredictionRoute = useCallback((runId: string | null) => {
    updateRouteSelection("predictions", runId);
  }, [updateRouteSelection]);

  const updateReviewRoute = useCallback((runId: string | null, findingId: string | null) => {
    updateRouteSelection("results", runId, findingId);
  }, [updateRouteSelection]);

  useEffect(() => {
    const onPopState = () => {
      const welcome = window.location.pathname === "/";
      const nextView = viewFromPath(window.location.pathname);
      const search = new URLSearchParams(window.location.search);
      setIsWelcome(welcome);
      setView(nextView);
      setVisitedViews((current) => new Set(current).add(nextView));
      setRouteRunId(search.get("run_id"));
      setRouteFindingId(search.get("finding_id"));
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [applyRoute]);

  useEffect(() => {
    void checkService();
    const timer = window.setInterval(() => void checkService(), 5_000);
    const handleOnline = () => void checkService();
    const handleOffline = () => setServiceStatus("offline");
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, [checkService]);

  const acceptSnapshot = useCallback((snapshot: BatchSnapshot) => {
    setBatch(snapshot);
    localStorage.setItem(ACTIVE_BATCH_KEY, snapshot.batch_id);
    setSelectedKey((current) => current && snapshot.companies.some((company) => company.company_key === current)
      ? current
      : snapshot.companies[0]?.company_key ?? null);
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(ACTIVE_BATCH_KEY);
    if (!saved) return;
    void getBatch(saved)
      .then(acceptSnapshot)
      .catch(() => localStorage.removeItem(ACTIVE_BATCH_KEY));
  }, [acceptSnapshot]);

  useEffect(() => {
    if (!batch?.batch_id) return;
    const source = connectBatchStream(
      batch.batch_id,
      acceptSnapshot,
      (companyKey, events) => setEventsByCompany((current) => ({
        ...current,
        [companyKey]: mergeAnalysisEvents(current[companyKey] ?? [], events),
      })),
      setConnected,
    );
    return () => source.close();
  }, [batch?.batch_id, acceptSnapshot]);

  const selected = useMemo(
    () => batch?.companies.find((company) => company.company_key === selectedKey) ?? null,
    [batch, selectedKey],
  );
  const selectedEvents = useMemo(
    () => selected ? eventsByCompany[selected.company_key] ?? [] : [],
    [eventsByCompany, selected],
  );
  const visibleEvents = useMemo(
    () => selectedEvents.filter((event) => !event.stage.endsWith(":delta")),
    [selectedEvents],
  );

  const action = useCallback(async (kind: "pause" | "resume") => {
    if (!batch) return;
    if (serviceStatus !== "online") {
      setError("分析服务当前未连接，请重新连接后再操作。");
      return;
    }
    setBusy(true);
    try {
      acceptSnapshot(await batchAction(batch.batch_id, kind));
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : "任务操作失败。");
    } finally {
      setBusy(false);
    }
  }, [acceptSnapshot, batch, serviceStatus]);

  const reconnect = useCallback(async () => {
    const available = await checkService();
    setError(available ? null : "仍无法连接分析服务，请确认后端 8000 端口已经启动。");
  }, [checkService]);

  const serviceAvailable = serviceStatus === "online";

  if (isWelcome) {
    return (
      <TooltipProvider>
        <LandingPage onEnter={() => applyRoute("workspace")} />
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur-xl">
          <div className="mx-auto flex h-16 w-full max-w-[1920px] items-center gap-4 px-4 lg:px-6 2xl:px-8">
            <button type="button" className="flex min-w-0 flex-1 items-center gap-3 text-left" onClick={showWelcome} aria-label="返回系统首页">
              <span className="grid size-9 shrink-0 place-items-center rounded-xl border border-primary/30 bg-primary/10 text-primary shadow-sm">
                <ShieldCheck className="size-5" />
              </span>
              <div className="hidden min-w-0 sm:block">
                <strong className="block truncate text-sm">IPO 风险操作系统</strong>
                <span className="hidden text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:block">Agentic Intelligence Workspace</span>
              </div>
            </button>

            <nav className="flex rounded-xl border bg-card/70 p-1" aria-label="主导航">
              <Button variant={view === "workspace" ? "secondary" : "ghost"} size="sm" onClick={() => applyRoute("workspace")}>
                <LayoutDashboard data-icon="inline-start" />分析工作台
              </Button>
              <Button variant={view === "predictions" ? "secondary" : "ghost"} size="sm" onClick={() => applyRoute("predictions", selected?.run_id)}>
                <ChartNoAxesCombined data-icon="inline-start" />风险预测
              </Button>
              <Button variant={view === "results" ? "secondary" : "ghost"} size="sm" onClick={() => applyRoute("results", selected?.run_id)}>
                <FileCheck2 data-icon="inline-start" />结果与复核
              </Button>
            </nav>

            <div className="flex flex-1 justify-end">
              {view === "workspace" ? (
                <Badge variant={connected && serviceAvailable ? "secondary" : "outline"}>
                  {connected && serviceAvailable ? <Wifi data-icon="inline-start" /> : <WifiOff data-icon="inline-start" />}
                  <span className="hidden sm:inline">{serviceStatus === "checking" ? "正在检测" : connected && serviceAvailable ? "实时连接" : serviceAvailable ? "服务已连接" : "服务已断开"}</span>
                </Badge>
              ) : (
                <Badge variant={serviceAvailable ? "secondary" : "outline"}>
                  {serviceAvailable ? <Database data-icon="inline-start" /> : <WifiOff data-icon="inline-start" />}
                  <span className="hidden sm:inline">{serviceStatus === "checking" ? "正在检测" : serviceAvailable ? "分析服务已连接" : "分析服务已断开"}</span>
                </Badge>
              )}
            </div>
          </div>
        </header>

        {error && (
          <div className="fixed inset-x-4 top-20 z-50 mx-auto max-w-4xl">
            <Alert variant="destructive" className="bg-background/95 shadow-xl backdrop-blur">
              <AlertTitle>操作未完成</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
              <AlertAction>
                <div className="flex items-center gap-1">
                  {!serviceAvailable && (
                    <Button variant="ghost" size="sm" onClick={() => void reconnect()}>
                      <RefreshCw data-icon="inline-start" />重新连接
                    </Button>
                  )}
                  <Button variant="ghost" size="icon-sm" onClick={() => setError(null)} aria-label="关闭错误提示"><X /></Button>
                </div>
              </AlertAction>
            </Alert>
          </div>
        )}

        <main>
          <div hidden={view !== "workspace"}>
            <AnalysisWorkspace
              batch={batch}
              pending={pending}
              setPending={setPending}
              selected={selected}
              selectedKey={selectedKey}
              selectedEvents={selectedEvents}
              visibleEvents={visibleEvents}
              busy={busy}
              serviceAvailable={serviceAvailable}
              setBusy={setBusy}
              setError={setError}
              onSnapshot={acceptSnapshot}
              onSelectCompany={setSelectedKey}
              onAction={(kind) => void action(kind)}
              onOpenPredictions={() => applyRoute("predictions", selected?.run_id)}
            />
          </div>
          {visitedViews.has("predictions") && (
            <div hidden={view !== "predictions"}>
            <PredictionsWorkspace
              batch={batch}
              selectedKey={selectedKey}
              busy={busy}
              serviceAvailable={serviceAvailable}
              initialRunId={routeRunId}
              onSelectCompany={setSelectedKey}
              onSnapshot={acceptSnapshot}
              setBusy={setBusy}
              setError={setError}
              onSelectionChange={updatePredictionRoute}
              onOpenResults={(runId, findingId) => applyRoute("results", runId, findingId)}
            />
            </div>
          )}
          {visitedViews.has("results") && (
            <div hidden={view !== "results"}>
            <div className="mx-auto w-full max-w-[1920px] px-4 py-6 lg:px-6 2xl:px-8">
              <ReportsWorkspace
                active={view === "results"}
                serviceAvailable={serviceAvailable}
                initialRunId={routeRunId}
                initialFindingId={routeFindingId}
                onReconnect={reconnect}
                onSelectionChange={updateReviewRoute}
                setError={setError}
              />
            </div>
            </div>
          )}
        </main>
      </div>
    </TooltipProvider>
  );
}

export default App;
