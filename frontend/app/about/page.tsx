"use client";

import { motion } from "framer-motion";
import {
  BookOpen,
  Brain,
  GitBranch,
  Zap,
  ExternalLink,
  Boxes,
  ArrowDown,
  Sigma,
  Code2,
  Database,
  Network,
} from "lucide-react";
import PageShell from "@/components/PageShell";

const FLOW = [
  { label: "Neuron A", icon: Brain },
  { label: "Neuron B", icon: Brain },
];

const PROCESS = [
  "Biological + Morphological Features",
  "Pairwise Feature Engineering",
  "Random Forest",
  "Connection Score",
  "Candidate Ranking",
];

const TECHS = [
  { label: "Frontend", value: "Next.js 15, React 19, Tailwind CSS 4" },
  { label: "Backend", value: "Python, FastAPI" },
  { label: "ML", value: "scikit-learn (Random Forest)" },
  { label: "Data", value: "FlyWire whole-brain connectome" },
  { label: "Visualization", value: "Framer Motion, Lucide Icons" },
  { label: "Language", value: "TypeScript, Python" },
];

export default function AboutPage() {
  return (
    <PageShell className="max-w-[1000px]">
      <header className="mb-10">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-accent/10 border border-accent/20 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Methodology
          </h1>
        </div>
        <p className="text-sm text-text-muted mt-1">
          How FlyMind turns neuron properties into connection scores
        </p>
      </header>

      {/* Overview */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-accent" />
          Overview
        </h2>
        <div className="text-sm text-text-secondary leading-relaxed space-y-3">
          <p>
            FlyMind is a machine-learning platform for analyzing neural connectivity in the{" "}
            <em>Drosophila melanogaster</em> brain. It learns patterns from a large whole-brain connectome
            and scores the likelihood that pairs of neurons connect — producing{" "}
            <strong className="text-text-primary">model-suggested connection scores</strong> for researchers to
            investigate further.
          </p>
          <p>
            The platform combines biological properties (neurotransmitter profiles), morphological measures
            (size, surface area, length), and spatial coordinates of neurons into features used by a
            supervised model trained on observed connectivity.
          </p>
        </div>
      </section>

      {/* ML Process */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <h2 className="text-base font-semibold text-text-primary mb-5 font-[family-name:var(--font-display)] flex items-center gap-2">
          <Sigma className="w-4 h-4 text-accent" />
          The Machine-Learning Process
        </h2>

        {/* A + B */}
        <div className="flex items-center justify-center gap-4 mb-4">
          {FLOW.map((f) => (
            <div key={f.label} className="flex items-center gap-2 px-4 py-2 rounded-[var(--radius-md)] bg-elevated border border-border-subtle">
              <f.icon className="w-4 h-4 text-accent" />
              <span className="text-sm font-medium text-text-primary">{f.label}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 px-4 py-2 rounded-[var(--radius-md)] bg-violet/10 border border-violet/20">
            <Code2 className="w-4 h-4 text-violet" />
            <span className="text-sm font-medium text-violet">Directed Pair</span>
          </div>
        </div>

        {/* Process flow */}
        <div className="space-y-2">
          {PROCESS.map((step, i) => {
            const icons = [Boxes, Boxes, Brain, Zap, Network];
            const Icon = icons[i];
            return (
              <motion.div
                key={step}
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{ duration: 0.3, delay: i * 0.06 }}
                className="relative"
              >
                <div className="flex items-center gap-3 px-4 py-3 rounded-[var(--radius-md)] bg-elevated border border-border-subtle">
                  <Icon className="w-4 h-4 text-accent shrink-0" />
                  <span className="text-sm text-text-secondary">{step}</span>
                </div>
                {i < PROCESS.length - 1 && (
                  <div className="flex justify-center py-1">
                    <ArrowDown className="w-4 h-4 text-text-muted" />
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* Random Forest explained */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
          <Brain className="w-4 h-4 text-violet" />
          Why a Random Forest?
        </h2>
        <p className="text-sm text-text-secondary leading-relaxed">
          FlyMind uses many decision trees together. Each tree independently evaluates patterns in the
          neuron-pair features, and the forest combines their decisions into a final score. This ensemble
          approach is robust, reproducible, and produces a ranking signal —
          ideal for generating candidate connections at connectome scale.
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          {[
            { n: "100", label: "decision trees" },
            { n: "60", label: "pair features" },
            { n: "139,255", label: "neurons" },
            { n: "3.73M", label: "observed edges" },
          ].map((s) => (
            <div key={s.label} className="p-3 rounded-[var(--radius-sm)] bg-elevated text-center">
              <div className="font-[family-name:var(--font-display)] text-xl font-bold text-accent">{s.n}</div>
              <div className="text-[11px] text-text-muted mt-0.5">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Tech stack */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-accent" />
          Tech Stack
        </h2>
        <div className="grid grid-cols-2 gap-3">
          {TECHS.map((item) => (
            <div key={item.label} className="p-3 rounded-[var(--radius-sm)] bg-elevated">
              <span className="text-caption">{item.label}</span>
              <p className="text-xs text-text-secondary mt-0.5">{item.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* External */}
      <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
        <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
          <Database className="w-4 h-4 text-accent" />
          Data Source
        </h2>
        <p className="text-sm text-text-secondary leading-relaxed mb-3">
          FlyMind uses the publicly available FlyWire whole-brain connectome of{" "}
          <em>Drosophila melanogaster</em>, which includes neuron metadata, synapse partners, and
          neurotransmitter predictions.
        </p>
        <a
          href="https://www.flywire.ai/"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm text-accent hover:text-accent/80 hover:underline underline-offset-4 transition-colors no-underline"
        >
          FlyWire connectome <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </section>
    </PageShell>
  );
}