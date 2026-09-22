import type { EvaluationResponse, ExperimentResult } from "./types";

export function experimentData(
  evaluation: EvaluationResponse | null,
  name: string
): Record<string, unknown> | null {
  if (!evaluation) return null;
  const exp = evaluation.experiments.find((e) => e.experiment === name);
  return exp ? exp.data : null;
}

export function deepGet(
  data: Record<string, unknown> | null | undefined,
  path: string[]
): unknown {
  let node: unknown = data;
  for (const key of path) {
    if (node === null || node === undefined) return undefined;
    if (typeof node === "object") node = (node as Record<string, unknown>)[key];
    else return undefined;
  }
  return node;
}

export function metric(
  evaluation: EvaluationResponse | null,
  path: string[]
): number | string | null {
  const val = deepGet(experimentData(evaluation, path[0]), path.slice(1));
  if (typeof val === "number") return val;
  if (typeof val === "string") return val;
  return null;
}

export function fmtPct(val: unknown, digits = 1): string {
  if (typeof val !== "number") return "—";
  return `${(val * 100).toFixed(digits)}%`;
}

export function fmtNum(val: unknown, digits = 3): string {
  if (typeof val !== "number") return "—";
  if (val < 1) return val.toFixed(digits);
  return val.toFixed(2);
}

export interface RankingMetrics {
  recall_at_10: number | null;
  recall_at_50: number | null;
  hit_rate_at_10: number | null;
  hit_rate_at_50: number | null;
  n_sources: number | null;
}

export function getRankingMetrics(
  evaluation: EvaluationResponse | null
): RankingMetrics {
  const data = experimentData(evaluation, "experiment_3");
  const ranking = data ? (data["3a_per_source_ranking"] as Record<string, unknown> | undefined) : undefined;
  return {
    recall_at_10: typeof ranking?.["mean_recall@10"] === "number" ? (ranking["mean_recall@10"] as number) : null,
    recall_at_50: typeof ranking?.["mean_recall@50"] === "number" ? (ranking["mean_recall@50"] as number) : null,
    hit_rate_at_10: typeof ranking?.["hit_rate@10"] === "number" ? (ranking["hit_rate@10"] as number) : null,
    hit_rate_at_50: typeof ranking?.["hit_rate@50"] === "number" ? (ranking["hit_rate@50"] as number) : null,
    n_sources: typeof ranking?.["n_sources_evaluated"] === "number" ? (ranking["n_sources_evaluated"] as number) : null,
  };
}

export interface FeatureImportanceRow {
  name: string;
  value: number;
}

export function getFeatureImportance(
  evaluation: EvaluationResponse | null
): FeatureImportanceRow[] {
  const data = experimentData(evaluation, "experiment_2d");
  const top = deepGet(data, ["feature_importance_top10"]);
  if (!Array.isArray(top)) return [];
  return top
    .map((row) => {
      if (Array.isArray(row) && row.length === 2) {
        return { name: String(row[0]), value: Number(row[1]) };
      }
      return null;
    })
    .filter((r): r is FeatureImportanceRow => r !== null);
}