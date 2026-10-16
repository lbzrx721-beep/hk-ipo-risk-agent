import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { AnalysisEvent } from "@/types";

type ProfileFact = {
  field: string;
  value: string;
  sources: { chunk_id: string; quote: string }[];
};
type ReviewPlan = {
  status: string;
  profile: { facts: ProfileFact[] };
  unknown_fields: string[];
  checks: { title: string; owner: string; question: string; fact_indices: number[]; queries: string[] }[];
  base_checks: { id: string; title: string; owner: string; question: string }[];
  retrieval_rounds?: { mode: string; chunk_ids: string[] }[];
  warnings?: string[];
};

const fields: Record<string, string> = {
  industry: "所属行业", business: "主营业务", revenue_model: "收入模式",
  customers_channels: "客户与渠道", suppliers: "供应链", key_assets: "核心资产与授权",
  development_stage: "发展阶段", geography: "经营地区",
};
const roles: Record<string, string> = {
  finance: "财务 Agent", legal: "法律 Agent", business: "业务 Agent", market: "市场 Agent",
};

export function parseCompanyReviewPlan(events: AnalysisEvent[]): ReviewPlan | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index].stage !== "company_review_plan") continue;
    try {
      const plan = JSON.parse(events[index].detail) as ReviewPlan;
      if (Array.isArray(plan.profile?.facts) && Array.isArray(plan.checks)
        && Array.isArray(plan.base_checks) && Array.isArray(plan.unknown_fields)) return plan;
    } catch { /* Interrupted or older events are not rendered as completed plans. */ }
  }
  return null;
}

export function CompanyReviewPlanView({ plan }: { plan: ReviewPlan }) {
  return (
    <div className="flex flex-col gap-4">
      <Alert>
        <AlertTitle>公司画像与审查规划</AlertTitle>
        <AlertDescription>
          这里展示原文支持的公司事实和待查问题，不代表风险已成立。专业 Agent 将独立检索全文并核验。
        </AlertDescription>
      </Alert>
      <div className="flex flex-wrap gap-2">
        <Badge variant="secondary">画像事实 {plan.profile.facts.length} 条</Badge>
        <Badge variant="outline">基础检查 {plan.base_checks.length} 项</Badge>
        <Badge variant="outline">专项问题 {plan.checks.length} 项</Badge>
        <Badge variant="outline">
          {plan.retrieval_rounds?.some((round) => round.mode === "hybrid") ? "混合检索" : "BM25 检索"}
        </Badge>
      </div>
      {plan.warnings?.map((warning, index) => (
        <Alert key={index}><AlertTitle>部分信息待补充</AlertTitle><AlertDescription>{warning}</AlertDescription></Alert>
      ))}
      <div className="grid gap-3 xl:grid-cols-2">
        {plan.profile.facts.map((fact, index) => (
          <Card key={`${fact.field}-${index}`}>
            <CardHeader>
              <CardTitle>{fields[fact.field] ?? fact.field}</CardTitle>
              <CardDescription>事实 {index + 1} · 原文引用</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <p>{fact.value}</p>
              {fact.sources.map((source, sourceIndex) => (
                <blockquote key={`${source.chunk_id}-${sourceIndex}`} className="border-l-2 pl-3 text-sm text-muted-foreground">
                  <p>{source.quote}</p><p className="mt-1 break-all">来源：{source.chunk_id}</p>
                </blockquote>
              ))}
            </CardContent>
          </Card>
        ))}
      </div>
      {plan.unknown_fields.length > 0 && (
        <p className="text-sm text-muted-foreground">尚未获取：{plan.unknown_fields.map((field) => fields[field] ?? field).join("、")}。未找到不等于不存在。</p>
      )}
      <Card>
        <CardHeader><CardTitle>基础检查目录</CardTitle><CardDescription>始终保留，不因画像提取结果而删除。</CardDescription></CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {plan.base_checks.map((check) => (
            <Badge key={check.id} variant="outline">{roles[check.owner] ?? check.owner} · {check.title}</Badge>
          ))}
        </CardContent>
      </Card>
      <div className="grid gap-3 xl:grid-cols-2">
        {plan.checks.map((check, index) => (
          <Card key={`${check.owner}-${index}`}>
            <CardHeader>
              <CardTitle>{check.title}</CardTitle>
              <CardDescription>{roles[check.owner] ?? check.owner} · 待查证</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <p>{check.question}</p>
              <p className="text-sm text-muted-foreground">提出依据：画像事实 {check.fact_indices.map((i) => i + 1).join("、")}</p>
              <details><summary className="cursor-pointer text-sm">查看检索问题</summary>
                <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm text-muted-foreground">
                  {check.queries.map((query, i) => <li key={i}>{query}</li>)}
                </ul>
              </details>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
