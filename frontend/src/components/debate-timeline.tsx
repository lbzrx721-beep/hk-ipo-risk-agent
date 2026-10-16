import { ArrowRight, MessageSquareWarning, Scale, SearchCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { DebateRecord } from "@/types";

const roleLabel: Record<string, string> = {
  discovery: "风险发现 Agent",
  finance: "财务 Agent",
  legal: "法律 Agent",
  business: "业务 Agent",
  market: "市场 Agent",
  decision: "决策 Agent",
};

function decisionLabel(value: boolean) {
  return value ? "纳入正式风险" : "不纳入正式风险";
}

export function DebateTimeline({
  records,
  active = false,
  className,
}: {
  records: DebateRecord[];
  active?: boolean;
  className?: string;
}) {
  return (
    <Card className={cn("border-primary/20 bg-card/75", className)}>
      <CardHeader className="border-b border-border/60">
        <CardTitle className="flex items-center gap-2">
          <MessageSquareWarning className="size-4 text-primary" />
          Agent辩论与补证
        </CardTitle>
        <CardDescription>决策Agent提出质疑，专业Agent定向检索原文，随后完成第二轮裁决。</CardDescription>
        <CardAction>
          <Badge variant={records.length > 0 ? "secondary" : "outline"}>
            {records.length > 0 ? `${records.length} 项辩论` : active ? "等待裁决" : "未触发辩论"}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-4">
        {records.length === 0 ? (
          <div className="rounded-xl border border-dashed bg-muted/10 px-4 py-6 text-center">
            <p className="text-sm font-medium">
              {active ? "决策Agent正在判断是否存在需要补证的实质争议" : "本次分析未发现需要二轮补证的实质争议"}
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              {active ? "触发后会在此实时展示质疑、补证问题与最终裁决。" : "各Agent结论与现有证据能够直接完成裁决。"}
            </p>
          </div>
        ) : records.map((record, index) => {
          const changed = record.first_round_accepted !== record.final_accepted;
          return (
            <section key={record.debate_id || `${record.risk_type}-${index}`} className="rounded-xl border bg-background/35 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground">辩论 {String(index + 1).padStart(2, "0")}</p>
                  <h4 className="mt-1 text-sm font-semibold leading-6">{record.risk_name || record.risk_type}</h4>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">
                    {record.trigger === "material_risk_review" ? "重大风险对抗性复核" : "决策分歧质疑"}
                  </Badge>
                  <Badge variant={record.status === "failed" ? "destructive" : "secondary"}>
                    {record.status === "failed" ? "补证异常" : "辩论完成"}
                  </Badge>
                  <Badge variant={changed ? "default" : "outline"}>{changed ? "裁决已调整" : "维持原裁决"}</Badge>
                </div>
              </div>

              <div className="mt-4 grid gap-3 xl:grid-cols-[1fr_auto_1.15fr_auto_1fr] xl:items-stretch">
                <div className="rounded-lg border border-primary/15 bg-primary/[0.035] p-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-primary">
                    <MessageSquareWarning className="size-3.5" />1. 决策质疑
                  </div>
                  <p className="mt-2 text-sm leading-6 text-foreground/85">{record.challenge || "决策Agent要求进一步核验该风险。"}</p>
                  <Badge variant="outline" className="mt-3 font-normal">初次：{decisionLabel(record.first_round_accepted)}</Badge>
                </div>

                <ArrowRight className="hidden size-4 self-center text-muted-foreground xl:block" />

                <div className="rounded-lg border p-3">
                  <div className="flex items-center gap-2 text-xs font-semibold">
                    <SearchCheck className="size-3.5 text-primary" />2. 专业补证
                  </div>
                  <p className="mt-2 text-sm">{roleLabel[record.challenge_role] ?? `${record.challenge_role} Agent`}重新检索招股书向量库</p>
                  {record.challenge_queries.length > 0 && (
                    <ol className="mt-2 space-y-1.5 pl-4 text-xs leading-5 text-muted-foreground">
                      {record.challenge_queries.map((query, queryIndex) => <li key={`${queryIndex}-${query}`}>{query}</li>)}
                    </ol>
                  )}
                  <div className="mt-3 flex flex-wrap gap-2">
                    <Badge variant="outline">召回证据 {record.recalled_evidence_count} 条</Badge>
                    <Badge variant="outline">新增证据 {record.new_evidence_count} 条</Badge>
                  </div>
                </div>

                <ArrowRight className="hidden size-4 self-center text-muted-foreground xl:block" />

                <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/[0.035] p-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    <Scale className="size-3.5" />3. 再次裁决
                  </div>
                  <p className="mt-2 text-sm font-medium">最终：{decisionLabel(record.final_accepted)}</p>
                  {record.final_rationale && <p className="mt-2 text-xs leading-5 text-muted-foreground">{record.final_rationale}</p>}
                  {record.error && <p className="mt-2 text-xs leading-5 text-destructive">{record.error}</p>}
                </div>
              </div>
            </section>
          );
        })}
      </CardContent>
    </Card>
  );
}
