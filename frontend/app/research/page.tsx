"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  FlaskConical,
  BookOpen,
  BarChart3,
  ListOrdered,
  AlertTriangle,
  FileJson,
  GitBranch,
  Brain,
} from "lucide-react";
import PageShell from "@/components/PageShell";
import StatCard from "@/components/StatCard";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import { getEvaluation, getResearchSummary } from "@/lib/api";
import type { EvaluationResponse, ResearchSummary } from "@/lib/types";
import {
  metric,
  fmtPct,
  fmtNum,
  getFeatureImportance,
  getRankingMetrics,
} from "@/lib/metrics";

export default function ResearchPage() {
  const [summary, setSummary] = useState<ResearchSummary | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([getResearchSummary(), getEvaluation()])
      .then(([s, e]) => {
        if (s.status === "fulfilled") setSummary(s.value);
        if (e.status === "fulfilled") setEvaluation(e.value);
      })
      .finally(() => setLoading(false));
  }, []);

  const featureImportance = getFeatureImportance(evaluation);
  const ranking = getRankingMetrics(evaluation);

  // Model performance (cold-start)
  const csRoc = metric(evaluation, ["cold_start", "rf_node_only", "roc_auc"]);
  const csPr = metric(evaluation, ["cold_start", "rf_node_only", "pr_auc"]);

  // Experiment list for the evaluation section
  const experimentRows = evaluation?.experiments ?? [];

  return (
    <PageShell>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-warning/10 border border-warning/20 flex items-center justify-center">
            <FlaskConical className="w-4 h-4 text-warning" />
          </div>
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Research Results
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Evaluation metrics sourced from validated research artifacts
        </p>
      </header>

      {loading ? (
        <>
          <LoadingSkeleton variant="cards" count={4} label="Loading research metrics..." />
          <div className="mt-6">
            <LoadingSkeleton variant="chart" count={1} />
          </div>
        </>
      ) : (
        <div className="space-y-10">
          {/* Summary */}
          {summary && (
            <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen className="w-4 h-4 text-warning" />
                <h2 className="text-base font-semibold text-text-primary font-[family-name:var(--font-display)]">
                  {summary.title}
                </h2>
              </div>
              <p className="text-sm text-text-secondary leading-relaxed">{summary.description}</p>
              {summary.dataset && Object.keys(summary.dataset).length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
                  {Object.entries(summary.dataset).map(([k, v]) => (
                    <div key={k} className="p-3 rounded-[var(--radius-sm)] bg-elevated">
                      <span className="text-caption block">{k.replace(/_/g, " ")}</span>
                      <span className="text-sm text-text-secondary font-[family-name:var(--font-mono)]">
                        {typeof v === "number" ? v.toLocaleString() : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Model performance */}
          <section>
            <h2 className="text-lg font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)] flex items-center gap-2">
              <Brain className="w-4 h-4 text-accent" /> Model Performance
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard label="ROC-AUC (cold-start)" value={fmtNum(csRoc, 4)} sublabel="Random Forest · generalization to unseen neurons" accent="success" />
              <StatCard label="PR-AUC (cold-start)" value={fmtNum(csPr, 4)} sublabel="Precision–recall under class imbalance" accent="warning" />
              <StatCard label="ROC-AUC (holdout)" value={metric(evaluation, ["link_prediction", "rf_node_only", "roc_auc"]) != null ? fmtNum(metric(evaluation, ["link_prediction", "rf_node_only", "roc_auc"]), 4) : "—"} sublabel="Random edge holdout split" accent="accent" />
              <StatCard label="PR-AUC (holdout)" value={metric(evaluation, ["link_prediction", "rf_node_only", "pr_auc"]) != null ? fmtNum(metric(evaluation, ["link_prediction", "rf_node_only", "pr_auc"]), 4) : "—"} sublabel="Edge-level evaluation" accent="violet" />
            </div>
          </section>

          {/* Candidate ranking */}
          <section>
            <h2 className="text-lg font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)] flex items-center gap-2">
              <ListOrdered className="w-4 h-4 text-accent" /> Candidate Ranking Quality
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard label="Recall@10" value={ranking.recall_at_10 != null ? fmtPct(ranking.recall_at_10) : "—"} sublabel={ranking.n_sources != null ? `across ${ranking.n_sources} sources` : undefined} accent="accent" />
              <StatCard label="Recall@50" value={ranking.recall_at_50 != null ? fmtPct(ranking.recall_at_50) : "—"} sublabel="per-source ranking" accent="violet" />
              <StatCard label="Hit Rate@10" value={ranking.hit_rate_at_10 != null ? fmtPct(ranking.hit_rate_at_10) : "—"} sublabel="≥1 positive in top 10" accent="success" />
              <StatCard label="Hit Rate@50" value={ranking.hit_rate_at_50 != null ? fmtPct(ranking.hit_rate_at_50) : "—"} sublabel="≥1 positive in top 50" accent="warning" />
            </div>
          </section>

          {/* Feature importance */}
          {featureImportance.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)] flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-accent" /> Feature Importance
              </h2>
              <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
                <p className="text-xs text-text-muted mb-5">
                  Top contributing node-level features to the Random Forest pair-scoring model (from experiment 2d).
                </p>
                <FeatureImportanceChart items={featureImportance} />
              </div>
            </section>
          )}

          {/* Evaluation experiments */}
          <section>
            <h2 className="text-lg font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)] flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-accent" /> Evaluation Experiments
            </h2>
            {experimentRows.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {experimentRows.map((exp) => (
                  <motion.div
                    key={exp.experiment}
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-40px" }}
                    transition={{ duration: 0.3 }}
                    className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-5"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <GitBranch className="w-4 h-4 text-text-muted" />
                      <span className="text-sm font-semibold text-text-primary font-[family-name:var(--font-mono)]">
                        {exp.experiment}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      {Object.entries(exp.data).slice(0, 6).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-caption block truncate" title={k}>{k.replace(/_/g, " ")}</span>
                          <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary truncate block">
                            {typeof v === "number"
                              ? v < 1 && v > 0
                                ? fmtPct(v as number, 1)
                                : v.toFixed(3)
                              : typeof v === "boolean"
                              ? String(v)
                              : String(v ?? "—").slice(0, 28)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-text-muted">No evaluation artifacts available.</div>
            )}
          </section>

          {/* Key findings */}
          {summary?.key_findings && summary.key_findings.length > 0 && (
            <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
              <div className="flex items-center gap-2 mb-4">
                <FileJson className="w-4 h-4 text-accent" />
                <h2 className="text-base font-semibold text-text-primary font-[family-name:var(--font-display)]">
                  Key Findings
                </h2>
              </div>
              <div className="space-y-3">
                {summary.key_findings.map((finding, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-[var(--radius-sm)] bg-elevated">
                    <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-accent/10 text-accent text-xs font-bold shrink-0">
                      {i + 1}
                    </span>
                    <p className="text-sm text-text-secondary text-left">{finding}</p>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Limitations */}
          <section className="bg-warning/5 border border-warning/20 rounded-[var(--radius-lg)] p-6">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4 text-warning" />
              <h2 className="text-base font-semibold text-text-primary font-[family-name:var(--font-display)]">
                Scientific Limitations
              </h2>
            </div>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li className="flex items-start gap-2"><span className="text-warning mt-0.5">—</span>Candidate connections are model-suggested hypotheses, not confirmed biological discoveries.</li>
              <li className="flex items-start gap-2"><span className="text-warning mt-0.5">—</span>Neurotransmitter annotations may be incomplete or uncertain.</li>
              <li className="flex items-start gap-2"><span className="text-warning mt-0.5">—</span>No biological experimental validation has been performed on predicted connections.</li>
              <li className="flex items-start gap-2"><span className="text-warning mt-0.5">—</span>A high ranking score does not equal biological probability of a synapse.</li>
            </ul>
          </section>
        </div>
      )}
    </PageShell>
  );
}