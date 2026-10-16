import { CirclePlay, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { submitRiskScope } from "@/api";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import type { BatchCompany, BatchSnapshot, RiskCandidate } from "@/types";

const ownerOptions = [
  { label: "财务 Agent", value: "finance" },
  { label: "法律 Agent", value: "legal" },
  { label: "业务 Agent", value: "business" },
  { label: "市场 Agent", value: "market" },
];

const severityLabels: Record<RiskCandidate["severity"], string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
  critical: "极高风险",
};

function severityVariant(severity: RiskCandidate["severity"]) {
  return severity === "critical" || severity === "high"
    ? "destructive"
    : severity === "medium"
      ? "secondary"
      : "outline";
}

type RiskScopePanelProps = {
  batch: BatchSnapshot;
  company: BatchCompany;
  serviceAvailable: boolean;
  onSnapshot: (snapshot: BatchSnapshot) => void;
  setError: (value: string | null) => void;
};

export function RiskScopePanel({ batch, company, serviceAvailable, onSnapshot, setError }: RiskScopePanelProps) {
  const review = company.risk_scope_review;
  const [reviewer, setReviewer] = useState("");
  const [globalRequirement, setGlobalRequirement] = useState("");
  const [candidates, setCandidates] = useState<RiskCandidate[]>(review?.candidates ?? []);
  const [included, setIncluded] = useState<Record<string, boolean>>(
    Object.fromEntries((review?.candidates ?? []).map((item) => [item.risk_type, item.ai_recommended])),
  );
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!review) return;
    setCandidates(review.candidates);
    setIncluded(Object.fromEntries(
      review.candidates.map((item) => [item.risk_type, item.ai_recommended]),
    ));
    setReviewer("");
    setGlobalRequirement("");
  }, [review?.run_id]);

  if (!review) return null;

  const recommendedCount = candidates.filter((candidate) => candidate.ai_recommended).length;
  const isReviewPlan = candidates.some((candidate) =>
    candidate.source === "mandatory_review" || candidate.source === "profile_review_question");
  const selectedCount = candidates.filter((candidate) => included[candidate.risk_type]).length;

  const updateOwner = (index: number, owner: string | null) => {
    if (!owner) return;
    setCandidates((current) => current.map((item, itemIndex) =>
      itemIndex === index ? { ...item, owner: owner as RiskCandidate["owner"] } : item,
    ));
  };

  const submit = async () => {
    if (!serviceAvailable) {
      setError("分析服务当前未连接，重新连接后才能确认风险范围。");
      return;
    }
    if (!reviewer.trim()) {
      setError("请填写风险范围确认人。");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const snapshot = await submitRiskScope(batch.batch_id, company.company_key, {
        run_id: review.run_id,
        reviewer: reviewer.trim(),
        selections: candidates.map((candidate) => ({
          risk_type: candidate.risk_type,
          included: included[candidate.risk_type] ?? false,
          owner: candidate.owner,
          severity: candidate.severity,
          analysis_requirement: "",
        })),
        added_risks: [],
        global_requirement: globalRequirement.trim(),
      });
      onSnapshot(snapshot);
    } catch (error) {
      setError(error instanceof Error ? error.message : "风险范围提交失败。");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card className="border-amber-500/20 bg-amber-500/[0.035]">
      <CardHeader className="border-b border-amber-500/10">
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="size-4 text-amber-400" />
          确认本次分析范围
        </CardTitle>
        <CardDescription>系统分析用时已暂停；确认后专业 Agent 将继续运行。</CardDescription>
        <CardAction><Badge variant="outline">等待人工确认</Badge></CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {[
            ["股票代码", company.company_id],
            ["公司名称", company.company_name],
            ["招股书日期", company.prospectus_date],
            ["所属行业", review.industry],
            ["业务模式", review.business_model],
            ["收入模式", review.revenue_model],
          ].map(([label, value]) => (
            <div key={label} className="min-w-0 rounded-lg border border-border/60 bg-background/30 p-3">
              <span className="text-xs text-muted-foreground">{label}</span>
              <p className="mt-1 line-clamp-2 text-sm font-medium">{value || "等待识别"}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">{isReviewPlan ? "待查事项" : "候选风险"} {candidates.length} 条</Badge>
          <Badge variant="outline">AI 建议纳入 {recommendedCount} 条</Badge>
          <Badge variant="outline">当前选中 {selectedCount} 条</Badge>
          {!isReviewPlan && review.funnel?.raw_candidates != null && (
            <Badge variant="outline">原始发现 {review.funnel.raw_candidates} 条</Badge>
          )}
          {!isReviewPlan && review.funnel?.consolidated_candidates != null && (
            <Badge variant="outline">语义合并 {review.funnel.consolidated_candidates} 条</Badge>
          )}
        </div>

        <ScrollArea className="h-[min(360px,45vh)] rounded-lg border border-border/60 bg-background/20">
          <FieldGroup className="gap-0">
            {candidates.map((candidate, index) => (
              <Field
                key={candidate.risk_type}
                orientation="horizontal"
                className="items-start border-b border-border/50 p-3 last:border-b-0"
              >
                <Checkbox
                  id={`${company.company_key}-${candidate.risk_type}`}
                  checked={included[candidate.risk_type] ?? false}
                  disabled={candidate.source === "mandatory_review"}
                  onCheckedChange={(checked) => setIncluded((current) => ({
                    ...current,
                    [candidate.risk_type]: checked === true,
                  }))}
                />
                <div className="flex min-w-0 flex-1 flex-col gap-2">
                  <FieldLabel htmlFor={`${company.company_key}-${candidate.risk_type}`}>
                    {candidate.title}
                  </FieldLabel>
                  <div className="flex flex-wrap gap-1">
                    {isReviewPlan ? <Badge variant="outline">{candidate.source === "mandatory_review" ? "基础必查" : "专项查证"}</Badge> : <Badge variant={severityVariant(candidate.severity)}>{severityLabels[candidate.severity]}</Badge>}
                    {!isReviewPlan && candidate.confidence != null && (
                      <Badge variant="outline">置信度 {Math.round(candidate.confidence * 100)}%</Badge>
                    )}
                    {!isReviewPlan && <Badge variant="outline">已有证据 {candidate.evidence_count ?? 0} 条</Badge>}
                    <Badge variant="outline">{candidate.ai_recommended ? "AI 建议纳入" : "AI 建议排除"}</Badge>
                  </div>
                  <p className="text-xs leading-relaxed text-muted-foreground">{candidate.rationale}</p>
                  <p className="text-xs leading-relaxed">
                    <span className="text-muted-foreground">核验重点：</span>{candidate.review_question}
                  </p>
                </div>
                <Select
                  items={ownerOptions}
                  value={candidate.owner}
                  onValueChange={(value) => updateOwner(index, value)}
                >
                  <SelectTrigger size="sm" className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {ownerOptions.map((option) => (
                        <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </Field>
            ))}
          </FieldGroup>
        </ScrollArea>

        <FieldGroup className="grid gap-3 lg:grid-cols-[220px_minmax(280px,1fr)_auto] lg:items-end">
          <Field>
            <FieldLabel htmlFor={`reviewer-${company.company_key}`}>确认人</FieldLabel>
            <Input
              id={`reviewer-${company.company_key}`}
              value={reviewer}
              onChange={(event) => setReviewer(event.target.value)}
              placeholder="姓名或工号"
            />
          </Field>
          <Field>
            <FieldLabel htmlFor={`requirement-${company.company_key}`}>统一补充分析要求</FieldLabel>
            <Input
              id={`requirement-${company.company_key}`}
              value={globalRequirement}
              onChange={(event) => setGlobalRequirement(event.target.value)}
              placeholder="可选"
            />
          </Field>
          <Button disabled={submitting || !serviceAvailable} onClick={() => void submit()}>
            {submitting ? <Spinner data-icon="inline-start" /> : <CirclePlay data-icon="inline-start" />}
            确认并启动专业 Agent
          </Button>
        </FieldGroup>
      </CardContent>
    </Card>
  );
}
