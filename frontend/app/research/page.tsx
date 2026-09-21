"use client";

import { useEffect, useState } from "react";
import { FlaskConical, BookOpen, TrendingUp, AlertTriangle } from "lucide-react";
import MetricCard from "@/components/MetricCard";
import { getResearchSummary, getEvaluation } from "@/lib/api";
import type { ResearchSummary, EvaluationResponse, ExperimentResult } from "@/lib/types";

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

  const lpExperiment = evaluation?.experiments.find((e) => e.experiment === "link_prediction");
  const coldStartExperiment = evaluation?.experiments.find((e) => e.experiment === "cold_start");

  const getMetric = (exp: ExperimentResult | undefined, key: string): string => {
    if (!exp) return "—";
    const val = exp.data[key];
    if (typeof val === "number") return `${(val * 100).toFixed(1)}%`;
    if (typeof val === "object" && val !== null) return JSON.stringify(val).slice(0, 40);
    return String(val ?? "—");
  };

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <FlaskConical className="w-5 h-5 text-success" />
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Research Results
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Evaluation metrics, key findings, and research methodology
        </p>
      </header>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 skeleton rounded-[var(--radius-md)]" />
            ))}
          </div>
        </div>
      ) : (
        <>
          {summary && (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-8">
              <div className="flex items-center gap-2 mb-4">
                <BookOpen className="w-4.5 h-4.5 text-success" />
                <h2 className="text-lg font-semibold text-text-primary font-[family-name:var(--font-display)]">
                  {summary.title}
                </h2>
              </div>
              <p className="text-sm text-text-secondary leading-relaxed mb-4">{summary.description}</p>
              {summary.dataset && Object.keys(summary.dataset).length > 0 && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
                  {Object.entries(summary.dataset).map(([k, v]) => (
                    <div key={k} className="p-3 rounded-[var(--radius-sm)] bg-elevated">
                      <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium block">{k.replace(/_/g, " ")}</span>
                      <span className="text-sm text-text-secondary font-[family-name:var(--font-mono)]">{String(v)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <MetricCard label="ROC-AUC" value={getMetric(lpExperiment, "roc_auc")} icon={TrendingUp} accent="success" subtitle="Link Prediction" />
            <MetricCard label="PR-AUC" value={getMetric(lpExperiment, "pr_auc")} icon={FlaskConical} accent="violet" subtitle="Link Prediction" />
            <MetricCard label="Cold-Start ROC-AUC" value={getMetric(coldStartExperiment, "roc_auc")} icon={TrendingUp} accent="accent" subtitle="Generalization" />
            <MetricCard label="Experiments" value={String(evaluation?.experiments.length ?? 0)} icon={BookOpen} accent="warning" subtitle="completed" />
          </div>

          {evaluation?.experiments && evaluation.experiments.length > 0 && (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-8">
              <h2 className="text-base font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)]">
                Experiment Results
              </h2>
              <div className="space-y-4">
                {evaluation.experiments.map((exp) => (
                  <div key={exp.experiment} className="p-4 rounded-[var(--radius-sm)] bg-elevated">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-text-primary">{exp.experiment}</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {Object.entries(exp.data).slice(0, 8).map(([k, v]) => (
                        <div key={k}>
                          <span className="text-[10px] uppercase text-text-muted block">{k.replace(/_/g, " ")}</span>
                          <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary">
                            {typeof v === "number" ? (v < 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)) : String(v ?? "—").slice(0, 30)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {summary?.key_findings && summary.key_findings.length > 0 && (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-8">
              <h2 className="text-base font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)]">
                Key Findings
              </h2>
              <div className="space-y-3">
                {summary.key_findings.map((finding, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-[var(--radius-sm)] bg-elevated">
                    <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-success/12 text-success text-xs font-bold shrink-0">
                      {i + 1}
                    </span>
                    <p className="text-sm text-text-secondary">{finding}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="bg-warning/5 border border-warning/20 rounded-[var(--radius-lg)] p-6">
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
          </div>
        </>
      )}
    </div>
  );
}
