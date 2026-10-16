import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { buildSync } from "esbuild";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

const root = fileURLToPath(new URL("../", import.meta.url));
const compiled = buildSync({
  absWorkingDir: root,
  entryPoints: ["src/features/workspace/company-review-plan.tsx"],
  bundle: true, platform: "node", format: "cjs", packages: "external", write: false,
  jsx: "automatic",
});
const loaded = { exports: {} };
new Function("require", "module", "exports", compiled.outputFiles[0].text)(
  createRequire(import.meta.url), loaded, loaded.exports,
);
const { parseCompanyReviewPlan, CompanyReviewPlanView } = loaded.exports;
const plan = {
  status: "completed",
  profile: { facts: [{ field: "business", value: "化妆品销售",
    sources: [{ chunk_id: "c1", quote: "本公司主营化妆品销售" }] }] },
  unknown_fields: ["revenue_model"],
  checks: [{ title: "渠道核查", owner: "business", question: "线上销售占比如何变化？",
    fact_indices: [0], queries: ["线上渠道 收入占比"] }],
  base_checks: [{ id: "cash_flow", title: "现金流", owner: "finance", question: "核查现金流" }],
  retrieval_rounds: [{ mode: "bm25", chunk_ids: ["c1"] }],
};

test("renders profile, citations, unknown fields and questions without risk judgment", () => {
  const html = renderToStaticMarkup(createElement(CompanyReviewPlanView, { plan }));
  for (const expected of ["化妆品销售", "本公司主营化妆品销售", "c1", "收入模式",
    "渠道核查", "现金流", "BM25", "待查证", "不代表风险已成立"]) {
    assert.ok(html.includes(expected), expected);
  }
  assert.doesNotMatch(html, /高风险|中风险|低风险|置信度|概率/);
});

test("uses latest complete plan and tolerates interrupted payloads", () => {
  const events = [{ stage: "company_review_plan", detail: JSON.stringify(plan) },
    { stage: "company_review_plan", detail: "{incomplete" }];
  assert.equal(parseCompanyReviewPlan(events).profile.facts[0].value, "化妆品销售");
  assert.equal(parseCompanyReviewPlan([{ stage: "company_review_plan", detail: "{}" }]), null);
});

test("degraded plan still displays baseline and an explicit warning", () => {
  const html = renderToStaticMarkup(createElement(CompanyReviewPlanView, {
    plan: { ...plan, status: "degraded", checks: [], warnings: ["专项问题生成失败，保留基础检查。"] },
  }));
  assert.ok(html.includes("专项问题生成失败"));
  assert.ok(html.includes("现金流"));
});
