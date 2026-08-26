import { DatabaseZap, RefreshCw, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { resolveLlmFailure } from "@/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import type { BatchCompany, BatchSnapshot } from "@/types";

type RecoveryAction = "retry_llm" | "fallback_bm25";

export function LlmRecoveryPanel({
  batch,
  company,
  serviceAvailable,
  onSnapshot,
  setError,
}: {
  batch: BatchSnapshot;
  company: BatchCompany;
  serviceAvailable: boolean;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setError: (value: string | null) => void;
}) {
  const request = company.llm_failure_recovery;
  const [submitting, setSubmitting] = useState<RecoveryAction | null>(null);
  if (!request) return null;

  const submit = async (action: RecoveryAction) => {
    setSubmitting(action);
    setError(null);
    try {
      const snapshot = await resolveLlmFailure(batch.batch_id, company.company_key, {
        request_id: request.request_id,
        run_id: request.run_id,
        action,
      });
      onSnapshot(snapshot);
    } catch (error) {
      setError(error instanceof Error ? error.message : "模型故障恢复操作失败。" );
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <Card className="border-destructive/40 bg-destructive/5">
      <CardHeader className="border-b border-destructive/20">
        <div className="flex flex-wrap items-center gap-2">
          <TriangleAlert className="size-5 text-destructive" />
          <CardTitle>风险发现 Agent 调用失败，当前公司已暂停</CardTitle>
          <Badge variant="destructive">第 {request.attempt} 次失败</Badge>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertTitle>{request.reason}</AlertTitle>
          <AlertDescription className="break-words font-mono text-xs">
            {request.technical_detail}
          </AlertDescription>
        </Alert>

        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <Badge variant="outline">阶段：风险发现</Badge>
          {request.provider && <Badge variant="outline">服务：{request.provider}</Badge>}
          {request.model && <Badge variant="outline">模型：{request.model}</Badge>}
        </div>

        <p className="text-sm text-muted-foreground">
          两种操作都会复用已经完成的 PDF 解析和向量索引，不会重复预处理文档。
          重试只重新调用风险发现模型；BM25 会使用固定风险目录检索原文，并记录本次人工降级选择。
        </p>
      </CardContent>
      <CardFooter className="flex flex-col gap-2 border-t border-destructive/20 sm:flex-row">
        <Button
          disabled={!serviceAvailable || submitting !== null}
          onClick={() => void submit("retry_llm")}
        >
          {submitting === "retry_llm" ? <Spinner /> : <RefreshCw />}
          重试 LLM
        </Button>
        <Button
          variant="outline"
          disabled={!serviceAvailable || submitting !== null}
          onClick={() => void submit("fallback_bm25")}
        >
          {submitting === "fallback_bm25" ? <Spinner /> : <DatabaseZap />}
          降级使用 BM25 检索
        </Button>
      </CardFooter>
    </Card>
  );
}
