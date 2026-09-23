"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Brain,
  ArrowRight,
  Cpu,
  Database,
  Workflow,
  Boxes,
  TestTubes,
  Zap,
  ListOrdered,
  Gauge,
  Network,
  Radar,
} from "lucide-react";
import PageShell from "@/components/PageShell";
import StatCard from "@/components/StatCard";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import {
  getHealth,
  getModel,
  getEvaluation,
} from "@/lib/api";
import type { HealthStatus, ModelMetadata, EvaluationResponse } from "@/lib/types";
import { metric, fmtNum } from "@/lib/metrics";
import ErrorState from "@/components/ErrorState";

const HERO_PIPELINE = [
  { label: "FlyWire Data", icon: Database },
  { label: "ETL", icon: Workflow },
  { label: "Features", icon: Boxes },
  { label: "Random Forest", icon: Brain },
  { label: "Evaluation", icon: TestTubes },
  { label: "Scoring", icon: Zap },
  { label: "Ranking", icon: ListOrdered },
] as const;

const PAGE_ACTIONS = [
  {
    href: "/neurons",
    title: "Explore Neurons",
    desc: "Search and inspect neuron properties",
    icon: Brain,
    accent: "text-accent",
    border: "hover:border-accent/40",
    iconBg: "bg-accent/10",
  },
  {
    href: "/predictor",
    title: "Predict Connection",
    desc: "Score a directed neuron pair",
    icon: Zap,
    accent: "text-violet",
    border: "hover:border-violet/40",
    iconBg: "bg-violet/10",
  },
  {
    href: "/candidates",
    title: "Rank Candidates",
    desc: "Find model-suggested targets",
    icon: ListOrdered,
    accent: "text-success",
    border: "hover:border-success/40",
    iconBg: "bg-success/10",
  },
  {
    href: "/research",
    title: "View Research",
    desc: "Evaluation metrics and findings",
    icon: Radar,
    accent: "text-warning",
    border: "hover:border-warning/40",
    iconBg: "bg-warning/10",
  },
] as const;

export default function OverviewPage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    Promise.allSettled([getHealth(), getModel(), getEvaluation()])
      .then(([h, m, e]) => {
        if (h.status === "fulfilled") setHealth(h.value);
        if (m.status === "fulfilled") setModel(m.value);
        if (e.status === "fulfilled") setEvaluation(e.value);
        const ok = [h, m, e].filter((r) => r.status === "fulfilled").length;
        if (ok === 0) setError("Unable to load metrics from the API.");
        else if (ok < 3) setError("Some metrics could not be loaded.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  // Live values only — never hardcode dataset or evaluation numbers.
  const nNeurons = model?.n_neurons ?? health?.n_neurons ?? null;
  const nEdges = model?.n_edges ?? health?.n_edges ?? null;
  const rocAuc = metric(evaluation, ["cold_start", "rf_node_only", "roc_auc"]);
  const prAuc = metric(evaluation, ["cold_start", "rf_node_only", "pr_auc"]);

  return (
    <PageShell>
      {/* ---------- Hero ---------- */}
      <section className="relative overflow-hidden rounded-[var(--radius-lg)] border border-border-subtle bg-surface mb-10">
        <div
          className="absolute inset-0 opacity-40 pointer-events-none"
          style={{
            background:
              "radial-gradient(ellipse 60% 60% at 20% 0%, rgba(79,209,197,0.08), transparent 60%), radial-gradient(ellipse 50% 50% at 90% 100%, rgba(139,124,246,0.08), transparent 60%)",
          }}
        />
        <div className="relative px-6 sm:px-10 py-12 sm:py-16">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-accent/10 border border-accent/20 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-accent" />
              </div>
              <span className="text-xs font-medium tracking-[0.16em] uppercase text-text-muted">
                Computational Neuroscience Platform
              </span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-text-primary font-[family-name:var(--font-display)] leading-tight">
              <span className="text-gradient-accent">FlyMind</span>
            </h1>

            <p className="mt-5 text-base sm:text-lg text-text-secondary max-w-2xl leading-relaxed">
              Machine learning for exploring patterns in neuron connectivity.
              We learn relationships between neuron properties and observed
              synaptic connections in the fruit-fly brain.
            </p>

            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/predictor" className="btn btn-accent no-underline">
                Predict a Connection <ArrowRight className="w-3.5 h-3.5" />
              </Link>
              <Link href="/research" className="btn no-underline">
                Explore Research Results
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ---------- Key metrics ---------- */}
      {loading ? (
        <LoadingSkeleton variant="cards" count={4} label="Loading model metrics..." />
      ) : error && !model && !health && !evaluation ? (
        <div className="mb-10">
          <ErrorState message={error} onRetry={load} />
        </div>
      ) : (
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10" aria-label="Key metrics">
          <StatCard
            label="Neurons"
            value={nNeurons != null ? nNeurons.toLocaleString() : "—"}
            sublabel={
              model?.n_features != null
                ? `${model.n_features} biological & morphological features each`
                : "Awaiting model metadata"
            }
            accent="accent"
          />
          <StatCard
            label="Directed Connections"
            value={nEdges != null ? `${(nEdges / 1e6).toFixed(2)}M` : "—"}
            sublabel={nEdges != null ? `${nEdges.toLocaleString()} in evaluated dataset` : "—"}
            accent="violet"
          />
          <StatCard
            label="ROC-AUC"
            value={fmtNum(rocAuc)}
            sublabel="Cold-start generalization"
            accent="success"
          />
          <StatCard
            label="PR-AUC"
            value={fmtNum(prAuc)}
            sublabel="Precision–recall balance"
            accent="warning"
          />
        </section>
      )}

      {/* ---------- How FlyMind Works ---------- */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 sm:p-8 mb-10">
        <div className="flex items-center gap-2 mb-2">
          <Gauge className="w-4 h-4 text-accent" />
          <h2 className="text-lg font-semibold text-text-primary font-[family-name:var(--font-display)]">
            How FlyMind Works
          </h2>
        </div>
        <p className="text-sm text-text-muted mb-6">
          A supervised machine-learning pipeline maps neuron properties to observed connectivity.
        </p>

        <div className="flex flex-wrap items-center gap-2">
          {HERO_PIPELINE.map((stage, i) => (
            <div key={stage.label} className="flex items-center gap-2">
              <div className="group flex items-center gap-2 px-3 py-2 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle hover:border-border-strong transition-colors">
                <stage.icon className="w-3.5 h-3.5 text-accent" />
                <span className="text-xs font-medium text-text-secondary">{stage.label}</span>
              </div>
              {i < HERO_PIPELINE.length - 1 && (
                <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
              )}
            </div>
          ))}
        </div>

        <Link
          href="/pipeline"
          className="inline-flex items-center gap-1.5 text-sm text-accent hover:text-accent/80 hover:underline underline-offset-4 transition-colors no-underline mt-5"
        >
          Explore the full pipeline <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </section>

      {/* ---------- Navigation actions ---------- */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" aria-label="Quick navigation">
        {PAGE_ACTIONS.map((action) => (
          <motion.div
            key={action.href}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.35 }}
          >
            <Link
              href={action.href}
              className={`group bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-5 card-interactive no-underline h-full flex flex-col ${action.border}`}
            >
              <div className={`w-9 h-9 rounded-[var(--radius-sm)] ${action.iconBg} flex items-center justify-center mb-3`}>
                <action.icon className={`w-4 h-4 ${action.accent}`} />
              </div>
              <h3 className="text-sm font-semibold text-text-primary mb-1 font-[family-name:var(--font-display)]">
                {action.title}
              </h3>
              <p className="text-xs text-text-muted flex-1 leading-relaxed">{action.desc}</p>
              <ArrowRight className={`w-4 h-4 text-text-muted group-hover:${action.accent} mt-3 transition-colors`} />
            </Link>
          </motion.div>
        ))}
      </section>

      {/* ---------- Footer strip ---------- */}
      <section className="mt-10 flex flex-col sm:flex-row items-start sm:items-center gap-3 rounded-[var(--radius-lg)] border border-warning/20 bg-warning/5 p-5">
        <Network className="w-4 h-4 text-warning shrink-0 mt-0.5" />
        <p className="text-xs text-text-secondary leading-relaxed">
          Scores shown are <strong className="text-text-primary">model-suggested connection scores</strong> learned
          from patterns in observed connectivity. They do not constitute biological confirmation of a connection.
        </p>
      </section>
    </PageShell>
  );
}