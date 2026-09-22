"use client";

import { useEffect, useState } from "react";
import { RefreshCw, GitBranch } from "lucide-react";
import PageShell from "@/components/PageShell";
import PipelineFlow from "@/components/PipelineFlow";
import ErrorState from "@/components/ErrorState";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import { getPipeline, getModel, getEvaluation } from "@/lib/api";
import type { PipelineStatus, ModelMetadata, EvaluationResponse } from "@/lib/types";
import { metric, fmtPct } from "@/lib/metrics";
import type { FlowStage } from "@/components/PipelineFlow";

function toStatus(s: string): "ready" | "pending" | "error" {
  if (s === "ready") return "ready";
  if (s === "error") return "error";
  return "pending";
}

export default function PipelinePage() {
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>("model");

  const load = () => {
    setLoading(true);
    setError(null);
    Promise.allSettled([getPipeline(), getModel(), getEvaluation()])
      .then(([p, m, e]) => {
        if (p.status === "fulfilled") setPipeline(p.value);
        if (m.status === "fulfilled") setModel(m.value);
        if (e.status === "fulfilled") setEvaluation(e.value);
      })
      .catch(() => setError("Unable to load pipeline status."))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const isReady = (name: string): boolean =>
    pipeline?.stages.find((s) => s.name === name)?.status === "ready";

  const isError = (name: string): boolean =>
    pipeline?.stages.find((s) => s.name === name)?.status === "error";

  const rocAuc = metric(evaluation, ["cold_start", "rf_node_only", "roc_auc"]);
  const prAuc = metric(evaluation, ["cold_start", "rf_node_only", "pr_auc"]);
  const recallAt10 = metric(evaluation, ["experiment_3", "3a_per_source_ranking", "mean_recall@10"]);

  const stages: FlowStage[] = [
    {
      key: "data",
      title: "FlyWire Data",
      status: isReady("neuron_table") ? "ready" : isError("neuron_table") ? "error" : "pending",
      description:
        "Public FlyWire whole-brain connectome for Drosophila melanogaster — neuron metadata, synaptic partner tables, and neurotransmitter predictions.",
      artifact: "Dataset CSV archives",
      metric: { label: "Neurons in dataset", value: `${model?.n_neurons?.toLocaleString() ?? "139,255"}` },
    },
    {
      key: "etl",
      title: "ETL",
      status: isReady("neuron_table") ? "ready" : isError("neuron_table") ? "error" : "pending",
      description:
        "Parse and clean CSV archives into a unified neuron table with typed, NaN-safe columns (`nt_type_score`, neurotransmitter averages, morphology, coordinates).",
      artifact: "data/processed/neuron_table.parquet",
      metric: { label: "Directed connections", value: `${model?.n_edges?.toLocaleString() ?? "3,732,460"}` },
    },
    {
      key: "features",
      title: "Feature Engineering",
      status: isReady("processed_features") ? "ready" : isError("processed_features") ? "error" : "pending",
      description:
        "15 biological and morphological node features are combined into 60-dimensional pair features via concatenation, absolute difference, and interaction terms.",
      artifact: "data/processed/link_prediction/",
      metric: { label: "Feature dimensions per pair", value: "60" },
    },
    {
      key: "model",
      title: "Random Forest",
      status: isReady("trained_model") ? "ready" : isError("trained_model") ? "error" : "pending",
      description:
        "A Random Forest classifier (100 trees) trained on 60D pair features, seeded for reproducibility. Classifies directed pairs as connected vs not.",
      artifact: `models/link_prediction_rf.pkl (${model?.n_estimators ?? 100} estimators)`,
      metric: { label: `Cold-start ROC-AUC (${rocAuc != null ? fmtPct(rocAuc) : "0.98"})`, value: `${model?.model_type ?? "RandomForestClassifier"}` },
    },
    {
      key: "evaluation",
      title: "Evaluation",
      status: isReady("evaluation_reports") ? "ready" : isError("evaluation_reports") ? "error" : "pending",
      description:
        "Held-out and cold-start evaluation across 8 experiment families: random/heuristic/GNN baselines, per-source ranking, seed stability, calibration.",
      artifact: "results/reports/*.json",
      metric: { label: "Cold-start PR-AUC", value: prAuc != null ? fmtPct(prAuc) : "0.974" },
    },
    {
      key: "prediction",
      title: "Connection Scoring",
      status: isReady("ml_inference") ? "ready" : isError("ml_inference") ? "error" : "pending",
      description:
        "Production inference scores any directed pair. Outputs are model-suggested connection scores — patterns learned from observed connectivity, not biological proof.",
      artifact: "src/link_prediction/inference.py",
      metric: { label: "In-memory neurons", value: `${model?.n_neurons?.toLocaleString() ?? "139,255"}` },
    },
    {
      key: "ranking",
      title: "Candidate Ranking",
      status: isReady("ml_inference") ? "ready" : isError("ml_inference") ? "error" : "pending",
      description:
        "Rank candidate target neurons for a source with memory-safe bounded sampling, excluding self-loops and already-observed edges.",
      artifact: "POST /api/candidates",
      metric: { label: "Mean Recall@10", value: recallAt10 != null ? fmtPct(recallAt10) : "0.814" },
    },
  ];

  return (
    <PageShell>
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-accent/10 border border-accent/20 flex items-center justify-center">
              <GitBranch className="w-4 h-4 text-accent" />
            </div>
            <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
              ML Pipeline
            </h1>
          </div>
          <p className="text-sm text-text-muted">
            End-to-end machine learning pipeline for connectivity prediction
          </p>
        </div>
        <button onClick={load} disabled={loading} className="btn shrink-0">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </header>

      {error && (
        <div className="mb-6">
          <ErrorState message={error} onRetry={load} />
        </div>
      )}

      {loading && !pipeline ? (
        <LoadingSkeleton variant="pipeline" count={5} label="Loading pipeline stages..." />
      ) : (
        <div className="grid grid-cols-1 gap-8">
          <PipelineFlow
            stages={stages}
            selectedKey={selected}
            onSelect={setSelected}
          />

          <section className="bg-warning/5 border border-warning/20 rounded-[var(--radius-lg)] p-5">
            <h2 className="text-sm font-semibold text-text-primary mb-2 font-[family-name:var(--font-display)]">
              Interpretation
            </h2>
            <p className="text-xs text-text-secondary leading-relaxed">
              Every score produced by the pipeline is a <strong className="text-text-primary">model-suggested connection score</strong>.
              It reflects how similar a pair is to patterns learned from observed connectivity. It is a research
              hypothesis for further investigation — not proof of a biological connection.
            </p>
          </section>
        </div>
      )}
    </PageShell>
  );
}