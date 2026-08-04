import { eventRole } from "@/lib/presentation";
import type { AnalysisEvent } from "@/types";

export type AgentOutputKind = "discovery" | "consolidated" | "analysis" | "decision";

export type AgentOutputCard = {
  kind: AgentOutputKind;
  title: string;
  riskType: string | null;
  claim: string;
  severity: string;
  confidence: number | null;
  evidenceIds: string[];
  owner: string | null;
  supported: boolean | null;
  accepted: boolean | null;
  uncertainty: string;
  rationale: string;
};

export type AgentOutputPresentation = {
  cards: AgentOutputCard[];
  incomplete: boolean;
};

const arrayKinds = [
  ["consolidated_risks", "consolidated"],
  ["candidates", "discovery"],
  ["assessments", "analysis"],
  ["decisions", "decision"],
] as const;

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function ids(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String).filter(Boolean) : [];
}

function confidence(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(value, 1))
    : null;
}

function stripEnvelope(value: string): string {
  let stripped = value.trim().replace(/^\[\[(?:batch|consolidation):\d+\/\d+]]/, "").trim();
  if (stripped.startsWith("```")) {
    stripped = stripped.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  }
  return stripped;
}

function completeObjects(value: string, start: number): string[] {
  const result: string[] = [];
  let objectStart = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < value.length; index += 1) {
    const character = value[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') inString = false;
      continue;
    }
    if (character === '"') {
      inString = true;
    } else if (character === "{") {
      if (depth === 0) objectStart = index;
      depth += 1;
    } else if (character === "}" && depth > 0) {
      depth -= 1;
      if (depth === 0 && objectStart >= 0) {
        result.push(value.slice(objectStart, index + 1));
        objectStart = -1;
      }
    }
  }
  return result;
}

function cardFromItem(
  role: string,
  item: Record<string, unknown>,
  kind: AgentOutputKind,
  riskTitles: Record<string, string>,
): AgentOutputCard {
  const riskType = text(item.risk_type) || null;
  const supported = typeof item.supported === "boolean" ? item.supported : null;
  const accepted = typeof item.accepted === "boolean" ? item.accepted : null;
  const fallback = riskType ? riskTitles[riskType] : "";
  const roleLabel = role === "decision" ? "决策裁决" : role === "discovery" ? "风险发现" : "专业核验";
  return {
    kind,
    title: text(item.title) || fallback || (riskType ? `${roleLabel} · ${riskType}` : `${roleLabel} · 结构化结论`),
    riskType,
    claim: text(item.claim),
    severity: text(item.severity) || "unknown",
    confidence: confidence(item.confidence),
    evidenceIds: ids(item.evidence_chunk_ids),
    owner: text(item.owner) || null,
    supported,
    accepted,
    uncertainty: text(item.uncertainty),
    rationale: text(item.rationale),
  };
}

function cardsFromPayload(
  role: string,
  payload: Record<string, unknown>,
  riskTitles: Record<string, string>,
): AgentOutputCard[] {
  for (const [key, kind] of arrayKinds) {
    const values = payload[key];
    if (Array.isArray(values)) {
      return values
        .filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item))
        .map((item) => cardFromItem(role, item, kind, riskTitles));
    }
  }
  if (["risk_type", "title", "claim", "supported", "accepted"].some((key) => key in payload)) {
    return [cardFromItem(role, payload, role === "decision" ? "decision" : "analysis", riskTitles)];
  }
  return [];
}

function partialCards(
  role: string,
  segment: string,
  riskTitles: Record<string, string>,
): AgentOutputCard[] {
  for (const [key, kind] of arrayKinds) {
    const match = new RegExp(`"${key}"\\s*:\\s*\\[`).exec(segment);
    if (!match) continue;
    return completeObjects(segment, match.index + match[0].length).flatMap((raw) => {
      try {
        const item = JSON.parse(raw) as Record<string, unknown>;
        return [cardFromItem(role, item, kind, riskTitles)];
      } catch {
        return [];
      }
    });
  }
  return [];
}

export function parseAgentOutput(
  events: AnalysisEvent[],
  role: string,
  suppliedRiskTitles: Record<string, string> = {},
): AgentOutputPresentation {
  const riskTitles = { ...suppliedRiskTitles };
  const streams = new Map<string, { value: string; structured: boolean }>();
  let incomplete = false;

  for (const event of events) {
    if (eventRole(event.stage) !== role) continue;
    if (event.stage.endsWith(":risk_catalog")) {
      try {
        Object.assign(riskTitles, JSON.parse(event.detail) as Record<string, string>);
      } catch {
        // 风险标题只是增强信息，损坏时不影响结论卡片。
      }
      continue;
    }
    if (!event.stage.endsWith(":structured") && !event.stage.endsWith(":delta")) continue;
    const marker = /^\[\[(?:batch|consolidation):\d+\/\d+]]/.exec(event.detail)?.[0];
    const key = marker ?? "__default__";
    const existing = streams.get(key);
    const structured = event.stage.endsWith(":structured");
    if (structured) {
      streams.set(key, { value: event.detail, structured });
    } else if (!existing?.structured) {
      const delta = marker ? event.detail.slice(marker.length) : event.detail;
      const prefix = existing?.value ?? marker ?? "";
      streams.set(key, { value: `${prefix}${delta}`, structured: false });
    }
  }

  const cards: AgentOutputCard[] = [];
  const hasStructured = [...streams.values()].some((stream) => stream.structured);
  for (const [key, stream] of streams) {
    if (hasStructured && key === "__default__" && !stream.structured) continue;
    const segment = stripEnvelope(stream.value);
    if (!segment) continue;
    try {
      const payload = JSON.parse(segment) as unknown;
      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        cards.push(...cardsFromPayload(role, payload as Record<string, unknown>, riskTitles));
      }
    } catch {
      incomplete = true;
      cards.push(...partialCards(role, segment, riskTitles));
    }
  }

  const deduplicated = new Map<string, AgentOutputCard>();
  cards.forEach((card, index) => {
    const identity = `${card.kind}:${card.riskType ?? `${card.title}:${index}`}`;
    deduplicated.set(identity, card);
  });
  return { cards: [...deduplicated.values()], incomplete };
}

export function parseAgentCards(
  events: AnalysisEvent[],
  role: string,
  riskTitles: Record<string, string> = {},
): AgentOutputCard[] {
  return parseAgentOutput(events, role, riskTitles).cards;
}
