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
import { batchAction, checkApiHealth, connectBatchStream, getBatch } from "@/api";
import { ACTIVE_BATCH_KEY, mergeAnalysisEvents, type AppView } from "@/lib/presentation";
import type { AnalysisEvent, BatchSnapshot, PendingUpload } from "@/types";

function App() {
  const [view, setView] = useState<AppView>("workspace");
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
    () => selectedEvents.filter((event) => !event.stage.endsWith(":delta")).slice(-40).reverse(),
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

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur-xl">
          <div className="mx-auto flex h-16 w-full max-w-[1920px] items-center gap-4 px-4 lg:px-6 2xl:px-8">
            <div className="flex min-w-0 flex-1 items-center gap-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-xl border border-primary/30 bg-primary/10 text-primary shadow-sm">
                <ShieldCheck className="size-5" />
              </span>
              <div className="hidden min-w-0 sm:block">
                <strong className="block truncate text-sm">IPO 风险操作系统</strong>
                <span className="hidden text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:block">Agentic Intelligence Workspace</span>
              </div>
            </div>

            <nav className="flex rounded-xl border bg-card/70 p-1" aria-label="主导航">
              <Button variant={view === "workspace" ? "secondary" : "ghost"} size="sm" onClick={() => setView("workspace")}>
                <LayoutDashboard data-icon="inline-start" />分析工作台
              </Button>
              <Button variant={view === "predictions" ? "secondary" : "ghost"} size="sm" onClick={() => setView("predictions")}>
                <ChartNoAxesCombined data-icon="inline-start" />五日预测
              </Button>
              <Button variant={view === "results" ? "secondary" : "ghost"} size="sm" onClick={() => setView("results")}>
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
          {view === "workspace" ? (
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
              onOpenPredictions={() => setView("predictions")}
            />
          ) : view === "predictions" ? (
            <PredictionsWorkspace
              batch={batch}
              selected={selected}
              selectedKey={selectedKey}
              busy={busy}
              serviceAvailable={serviceAvailable}
              onSelectCompany={setSelectedKey}
              onSnapshot={acceptSnapshot}
              setBusy={setBusy}
              setError={setError}
              onOpenResults={() => setView("results")}
            />
          ) : (
            <div className="mx-auto w-full max-w-[1920px] px-4 py-6 lg:px-6 2xl:px-8">
              <ReportsWorkspace serviceAvailable={serviceAvailable} onReconnect={reconnect} setError={setError} />
            </div>
          )}
        </main>
      </div>
    </TooltipProvider>
  );
}

export default App;
