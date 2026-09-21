"use client";

import { useEffect, useState } from "react";
import { GitBranch, RefreshCw } from "lucide-react";
import PipelineStage from "@/components/PipelineStage";
import { getPipeline } from "@/lib/api";
import type { PipelineStatus, PipelineStage as PipelineStageType } from "@/lib/types";

const STAGE_DESCRIPTIONS: Record<string, string> = {
  trained_model: "Random Forest classifier trained on 60D pair features",
  processed_features: "Feature matrix, ID mapping, and aggregated edge data",
  neuron_table: "Neuron metadata including coordinates, NT type, morphology",
  evaluation_reports: "Model evaluation metrics and experiment results",
  ml_inference: "Production inference engine for live predictions",
};

const STAGE_ORDER = ["trained_model", "processed_features", "neuron_table", "ml_inference", "evaluation_reports"];

export default function PipelinePage() {
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<PipelineStageType | null>(null);

  const load = () => {
    setLoading(true);
    getPipeline()
      .then(setPipeline)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const stages = [...(pipeline?.stages ?? [])].sort(
    (a, b) => (STAGE_ORDER.indexOf(a.name) ?? 99) - (STAGE_ORDER.indexOf(b.name) ?? 99)
  );
  const readyCount = stages.filter((s) => s.status === "ready").length;
  const total = stages.length;
  const progressPct = total > 0 ? Math.round((readyCount / total) * 100) : 0;

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="flex items-center justify-between mb-8">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <GitBranch className="w-5 h-5 text-accent" />
            <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
              ML Pipeline
            </h1>
          </div>
          <p className="text-sm text-text-muted">
            End-to-end machine learning pipeline for connectivity prediction
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-secondary hover:text-text-primary hover:border-border-strong transition-colors cursor-pointer disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </header>

      <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-8">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-text-primary">Overall Progress</span>
          <span className="text-sm font-[family-name:var(--font-mono)] text-text-secondary">
            {progressPct}%
          </span>
        </div>
        <div className="h-2 bg-elevated rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-accent to-violet rounded-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        <div className="mt-2">
          <span className="text-xs text-text-muted">
            {readyCount} of {total} stages ready
          </span>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 skeleton rounded-[var(--radius-md)]" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stages.map((stage) => (
            <PipelineStage
              key={stage.name}
              name={stage.name}
              status={stage.status === "ready" ? "completed" : stage.status === "error" ? "error" : "pending"}
              description={STAGE_DESCRIPTIONS[stage.name] || stage.detail}
              onClick={() => setSelected(stage)}
            />
          ))}
        </div>
      )}

      {selected && (
        <div
          className="fixed inset-0 bg-base/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="bg-surface border border-border-strong rounded-[var(--radius-lg)] p-6 max-w-md w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-text-primary font-[family-name:var(--font-display)]">
                {selected.name}
              </h3>
              <span
                className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                  selected.status === "ready"
                    ? "bg-success/12 text-success"
                    : selected.status === "error"
                    ? "bg-error/12 text-error"
                    : "bg-elevated text-text-muted"
                }`}
              >
                {selected.status}
              </span>
            </div>
            {STAGE_DESCRIPTIONS[selected.name] && (
              <p className="text-sm text-text-secondary mb-3">
                {STAGE_DESCRIPTIONS[selected.name]}
              </p>
            )}
            {selected.detail && (
              <div className="p-3 rounded-[var(--radius-sm)] bg-elevated mb-3">
                <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium block mb-1">Detail</span>
                <span className="text-sm text-text-secondary font-[family-name:var(--font-mono)]">{selected.detail}</span>
              </div>
            )}
            <button
              onClick={() => setSelected(null)}
              className="mt-4 w-full py-2 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
