const terminalAgentStates = new Set(["complete", "failed", "degraded", "skipped"]);

export function shouldShowStreamingPlaceholder(
  agentState: string | null | undefined,
  hasIncompleteStream: boolean,
): boolean {
  if (agentState === "running") return true;
  if (agentState && terminalAgentStates.has(agentState)) return false;
  return hasIncompleteStream;
}
