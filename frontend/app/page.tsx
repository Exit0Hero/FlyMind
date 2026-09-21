"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Brain,
  GitBranch,
  ArrowRight,
  Activity,
  Cpu,
  Zap,
  FlaskConical,
} from "lucide-react";
import MetricCard from "@/components/MetricCard";
import Badge from "@/components/Badge";
import { getHealth, getModel, getPipeline } from "@/lib/api";
import type { HealthStatus, ModelMetadata, PipelineStatus } from "@/lib/types";

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [pipeline, setPipeline] = useState<PipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([getHealth(), getModel(), getPipeline()])
      .then(([h, m, p]) => {
        if (h.status === "fulfilled") setHealth(h.value);
        if (m.status === "fulfilled") setModel(m.value);
        if (p.status === "fulfilled") setPipeline(p.value);
      })
      .finally(() => setLoading(false));
  }, []);

  const readyStages = pipeline?.stages.filter((s) => s.status === "ready").length ?? 0;
  const totalStages = pipeline?.stages.length ?? 0;

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-[var(--radius-md)] bg-accent/10 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
              FlyMind
            </h1>
            <p className="text-sm text-text-muted">
              Learning relationships between neuron properties and directed connectivity in the fruit-fly brain
            </p>
          </div>
        </div>
      </header>

      {loading ? (
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-32 skeleton rounded-[var(--radius-md)]" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <MetricCard
              label="Neurons"
              value={model?.n_neurons?.toLocaleString() ?? "—"}
              icon={Brain}
              accent="accent"
              subtitle={model?.model_type}
            />
            <MetricCard
              label="Directed Edges"
              value={model?.n_edges?.toLocaleString() ?? "—"}
              icon={GitBranch}
              accent="accent"
              subtitle="synaptic connections"
            />
            <MetricCard
              label="Feature Dimensions"
              value={model?.feature_dim ?? "—"}
              icon={Activity}
              accent="violet"
              subtitle={`${model?.n_features ?? 0} node-level features → ${model?.feature_dim ?? 0}D pair features`}
            />
            <MetricCard
              label="System Status"
              value={health?.status === "ok" ? "Ready" : health?.status ?? "Unknown"}
              icon={Cpu}
              accent={health?.status === "ok" ? "success" : "warning"}
              subtitle={health?.version ? `v${health.version}` : undefined}
            />
          </div>

          <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-text-primary font-[family-name:var(--font-display)]">
                ML Pipeline
              </h2>
              <Link
                href="/pipeline"
                className="text-xs text-accent hover:text-accent/80 flex items-center gap-1 no-underline transition-colors"
              >
                View All <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {(pipeline?.stages ?? []).map((stage) => (
                <div
                  key={stage.name}
                  className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle min-w-[140px]"
                >
                  <Badge
                    variant={
                      stage.status === "ready"
                        ? "success"
                        : stage.status === "error"
                        ? "error"
                        : "muted"
                    }
                    dot
                  >
                    {stage.name}
                  </Badge>
                </div>
              ))}
              {totalStages === 0 && (
                <span className="text-sm text-text-muted">No pipeline data</span>
              )}
            </div>
            {totalStages > 0 && (
              <div className="text-xs text-text-muted mt-3">
                {readyStages}/{totalStages} stages ready
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Link
              href="/neurons"
              className="group bg-surface border border-border-subtle rounded-[var(--radius-md)] p-5 card-interactive no-underline"
            >
              <Brain className="w-5 h-5 text-accent mb-3" />
              <h3 className="text-sm font-semibold text-text-primary mb-1">
                Neuron Explorer
              </h3>
              <p className="text-xs text-text-muted">
                Search and inspect individual neurons and their properties
              </p>
              <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-accent mt-3 transition-colors" />
            </Link>
            <Link
              href="/predictor"
              className="group bg-surface border border-border-subtle rounded-[var(--radius-md)] p-5 card-interactive no-underline"
            >
              <Zap className="w-5 h-5 text-violet mb-3" />
              <h3 className="text-sm font-semibold text-text-primary mb-1">
                Connection Predictor
              </h3>
              <p className="text-xs text-text-muted">
                Predict connection score between any two neurons
              </p>
              <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-violet mt-3 transition-colors" />
            </Link>
            <Link
              href="/candidates"
              className="group bg-surface border border-border-subtle rounded-[var(--radius-md)] p-5 card-interactive no-underline"
            >
              <FlaskConical className="w-5 h-5 text-success mb-3" />
              <h3 className="text-sm font-semibold text-text-primary mb-1">
                Candidate Ranking
              </h3>
              <p className="text-xs text-text-muted">
                Rank candidate target neurons for a given source
              </p>
              <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-success mt-3 transition-colors" />
            </Link>
            <Link
              href="/research"
              className="group bg-surface border border-border-subtle rounded-[var(--radius-md)] p-5 card-interactive no-underline"
            >
              <Activity className="w-5 h-5 text-warning mb-3" />
              <h3 className="text-sm font-semibold text-text-primary mb-1">
                Research Results
              </h3>
              <p className="text-xs text-text-muted">
                Explore evaluation metrics and findings
              </p>
              <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-warning mt-3 transition-colors" />
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
