import {
  FileText,
  History,
  LockKeyhole,
  Plus,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { appendBatch, createBatch, inspectFiles } from "@/api";
import type { AgentModelAssignments, AgentRoleKey, ReasoningModel } from "@/api";
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
import { Checkbox } from "@/components/ui/checkbox";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { BatchSnapshot, PendingUpload } from "@/types";

type UploadPanelProps = {
  pending: PendingUpload[];
  setPending: React.Dispatch<React.SetStateAction<PendingUpload[]>>;
  activeBatch: BatchSnapshot | null;
  busy: boolean;
  serviceAvailable: boolean;
  setBusy: (value: boolean) => void;
  setError: (value: string | null) => void;
  onSnapshot: (snapshot: BatchSnapshot) => void;
};

const agentRoles: Array<{ key: AgentRoleKey; label: string; description: string }> = [
  { key: "discovery", label: "风险发现 Agent", description: "全文扫描与候选风险发现" },
  { key: "finance", label: "财务 Agent", description: "财务质量、现金流与集中度" },
  { key: "legal", label: "法律 Agent", description: "条款、诉讼与合规" },
  { key: "business", label: "业务 Agent", description: "商业模式、供应链与文本粉饰" },
  { key: "market", label: "市场 Agent", description: "发行结构、流动性与市场情绪" },
  { key: "decision", label: "决策 Agent", description: "冲突整合、辩论与最终裁决" },
];

const modelOptions: Array<{ value: ReasoningModel; label: string; hint: string }> = [
  { value: "deepseek_flash", label: "DeepSeek Flash", hint: "最快" },
  { value: "deepseek_pro", label: "DeepSeek Pro", hint: "均衡" },
  { value: "kimi", label: "Kimi K3", hint: "深度" },
];

const modelPresets: Record<"fast" | "balanced" | "deep", AgentModelAssignments> = {
  fast: {
    discovery: "deepseek_flash", finance: "deepseek_flash", legal: "deepseek_flash",
    business: "deepseek_flash", market: "deepseek_flash", decision: "deepseek_flash",
  },
  balanced: {
    discovery: "deepseek_flash", finance: "deepseek_pro", legal: "deepseek_pro",
    business: "deepseek_pro", market: "deepseek_flash", decision: "kimi",
  },
  deep: {
    discovery: "kimi", finance: "kimi", legal: "kimi",
    business: "kimi", market: "kimi", decision: "kimi",
  },
};

export function UploadPanel({
  pending,
  setPending,
  activeBatch,
  busy,
  serviceAvailable,
  setBusy,
  setError,
  onSnapshot,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [agentModels, setAgentModels] = useState<AgentModelAssignments>({ ...modelPresets.balanced });

  const addFiles = useCallback(async (incoming: File[]) => {
    if (!serviceAvailable) {
      setError("分析服务当前未连接，重新连接后才能上传招股书。");
      return;
    }
    const pdfs = incoming.filter(
      (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"),
    );
    if (!pdfs.length) {
      setError("请拖入 PDF 格式的招股书。");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const suggestions = await inspectFiles(pdfs);
      setPending((current) => [
        ...current,
        ...suggestions.map((suggestion, index) => ({
          ...suggestion,
          file: pdfs[index],
          id: `${suggestion.source_hash}-${crypto.randomUUID()}`,
          force_reanalysis: false,
        })),
      ]);
    } catch (error) {
      setError(error instanceof Error ? error.message : "PDF 识别失败。");
    } finally {
      setBusy(false);
    }
  }, [serviceAvailable, setBusy, setError, setPending]);

  const updatePending = (id: string, patch: Partial<PendingUpload>) => {
    setPending((current) => current.map((row) => row.id === id ? { ...row, ...patch } : row));
  };

  const submit = async () => {
    if (!serviceAvailable) {
      setError("分析服务当前未连接，重新连接后才能创建分析任务。");
      return;
    }
    if (pending.some((item) => !item.company_id || !item.company_name || !item.document_date)) {
      setError("请补全每份 PDF 的股票代码、公司名称和招股书日期。");
      return;
    }
    const protectedHistory = pending.find((item) => item.existing_run_id && !item.force_reanalysis);
    if (protectedHistory) {
      setError(
        `${protectedHistory.file_name} 已有完整分析（${protectedHistory.existing_run_id}）。` +
        "默认不会重复产生模型费用；如确需重跑，请勾选重新分析。",
      );
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const snapshot = activeBatch
        ? await appendBatch(activeBatch.batch_id, pending, agentModels)
        : await createBatch(pending, agentModels);
      setPending([]);
      onSnapshot(snapshot);
    } catch (error) {
      setError(error instanceof Error ? error.message : "创建分析任务失败。");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="border-primary/10 bg-card/80 shadow-2xl shadow-black/10 backdrop-blur-xl">
      <CardHeader className="border-b border-border/60">
        <CardTitle>{activeBatch ? "继续添加公司" : "创建公司分析队列"}</CardTitle>
        <CardDescription>拖入一份或多份招股书，系统会识别公司资料并创建独立任务。</CardDescription>
        {activeBatch && (
          <CardAction>
            <Badge variant="outline">
              <Plus data-icon="inline-start" />
              追加到当前工作区
            </Badge>
          </CardAction>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-4 pt-1">
        <div
          role="button"
          tabIndex={0}
          aria-disabled={!serviceAvailable}
          className={cn(
            "group flex min-h-28 cursor-pointer items-center gap-4 rounded-xl border border-dashed border-border bg-muted/20 p-5 transition-colors outline-none hover:border-primary/40 hover:bg-primary/5 focus-visible:ring-3 focus-visible:ring-ring/50",
            dragging && "border-primary bg-primary/10",
          )}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void addFiles(Array.from(event.dataTransfer.files));
          }}
          onClick={() => serviceAvailable && inputRef.current?.click()}
          onKeyDown={(event) => {
            if (serviceAvailable && (event.key === "Enter" || event.key === " ")) inputRef.current?.click();
          }}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept="application/pdf,.pdf"
            hidden
            disabled={!serviceAvailable}
            onChange={(event) => {
              void addFiles(Array.from(event.target.files ?? []));
              event.currentTarget.value = "";
            }}
          />
          <div className="grid size-11 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary">
            <UploadCloud className="size-5" />
          </div>
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <strong className="text-sm font-medium">拖拽 PDF 到这里，或点击选择文件</strong>
            <span className="text-xs text-muted-foreground">支持批量上传；运行期间也可继续追加公司</span>
          </div>
          {busy && <Spinner className="text-primary" />}
        </div>

        {pending.length > 0 && (
          <div className="flex flex-col gap-3">
            <div className="overflow-hidden rounded-xl border border-primary/15 bg-primary/[0.035]">
              <div className="flex flex-col gap-3 border-b border-border/60 p-4 lg:flex-row lg:items-center lg:justify-between">
                <div className="min-w-0">
                  <p className="flex items-center gap-2 text-sm font-medium"><SlidersHorizontal className="size-4 text-primary" />Agent模型编排</p>
                  <p className="mt-1 text-xs text-muted-foreground">六个Agent独立选用推理模型；提交任务后立即锁定，本次运行不可切换。</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {([
                    ["fast", "快速模式"],
                    ["balanced", "均衡模式"],
                    ["deep", "深度模式"],
                  ] as const).map(([key, label]) => (
                    <Button key={key} type="button" size="sm" variant="outline" disabled={busy} onClick={() => setAgentModels({ ...modelPresets[key] })}>{label}</Button>
                  ))}
                </div>
              </div>
              <div className="grid gap-px bg-border/60 sm:grid-cols-2 xl:grid-cols-3">
                {agentRoles.map((role) => (
                  <div key={role.key} className="flex items-center justify-between gap-3 bg-card/95 p-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{role.label}</p>
                      <p className="mt-0.5 truncate text-[11px] text-muted-foreground">{role.description}</p>
                    </div>
                    <Select
                      value={agentModels[role.key]}
                      disabled={busy}
                      onValueChange={(value) => setAgentModels((current) => ({ ...current, [role.key]: value as ReasoningModel }))}
                    >
                      <SelectTrigger size="sm" className="w-40 shrink-0"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((model) => <SelectItem key={model.value} value={model.value}>{model.label} · {model.hint}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-2 border-t border-border/60 px-4 py-2.5 text-xs text-muted-foreground">
                <LockKeyhole className="size-3.5 text-primary" />启动后前端只读，后端同时固化模型分配；异常重试仍使用原模型。
              </div>
            </div>
            {pending.map((item) => {
              const historyProtected = Boolean(item.existing_run_id && !item.force_reanalysis);
              return (
                <Card key={item.id} size="sm" className="bg-muted/15">
                  <CardHeader className="border-b border-border/50">
                    <CardTitle className="flex min-w-0 items-center gap-2">
                      <FileText className="size-4 shrink-0 text-primary" />
                      <span className="truncate">{item.file_name}</span>
                    </CardTitle>
                    <CardDescription>{item.page_count} 页 · 信息可在提交前修正</CardDescription>
                    <CardAction>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="移除文件"
                        onClick={() => setPending((current) => current.filter((row) => row.id !== item.id))}
                      >
                        <X />
                      </Button>
                    </CardAction>
                  </CardHeader>
                  <CardContent className="flex flex-col gap-3">
                    {item.existing_run_id && (
                      <Alert className="border-emerald-500/20 bg-emerald-500/5">
                        <History />
                        <AlertTitle>检测到历史分析</AlertTitle>
                        <AlertDescription className="flex flex-wrap items-center gap-3">
                          <span className="break-all">{item.existing_run_id}</span>
                          <Field orientation="horizontal" className="w-auto">
                            <Checkbox
                              id={`rerun-${item.id}`}
                              checked={item.force_reanalysis}
                              onCheckedChange={(checked) => updatePending(item.id, { force_reanalysis: checked === true })}
                            />
                            <FieldLabel htmlFor={`rerun-${item.id}`}>使用当前模型重新分析</FieldLabel>
                          </Field>
                        </AlertDescription>
                      </Alert>
                    )}
                    <FieldGroup className="grid gap-3 md:grid-cols-[140px_minmax(220px,1fr)_170px_170px]">
                      <Field>
                        <FieldLabel htmlFor={`code-${item.id}`}>股票代码</FieldLabel>
                        <Input
                          id={`code-${item.id}`}
                          value={item.company_id ?? ""}
                          disabled={historyProtected}
                          onChange={(event) => updatePending(item.id, { company_id: event.target.value })}
                        />
                      </Field>
                      <Field>
                        <FieldLabel htmlFor={`company-${item.id}`}>公司名称</FieldLabel>
                        <Input
                          id={`company-${item.id}`}
                          value={item.company_name ?? ""}
                          disabled={historyProtected}
                          onChange={(event) => updatePending(item.id, { company_name: event.target.value })}
                        />
                      </Field>
                      <Field>
                        <FieldLabel htmlFor={`date-${item.id}`}>招股书日期</FieldLabel>
                        <Input
                          id={`date-${item.id}`}
                          type="date"
                          value={item.document_date ?? ""}
                          disabled={historyProtected}
                          onChange={(event) => updatePending(item.id, { document_date: event.target.value })}
                        />
                      </Field>
                      <Field>
                        <FieldLabel htmlFor={`listing-date-${item.id}`}>预计上市日期</FieldLabel>
                        <Input
                          id={`listing-date-${item.id}`}
                          type="date"
                          value={item.listing_date ?? ""}
                          disabled={historyProtected}
                          onChange={(event) => updatePending(item.id, { listing_date: event.target.value || null })}
                        />
                      </Field>
                    </FieldGroup>
                  </CardContent>
                </Card>
              );
            })}
            <Button className="self-end" disabled={busy || !serviceAvailable} onClick={() => void submit()}>
              {busy ? <Spinner data-icon="inline-start" /> : <Sparkles data-icon="inline-start" />}
              {activeBatch ? `添加并分析 ${pending.length} 家公司` : `开始分析 ${pending.length} 家公司`}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
