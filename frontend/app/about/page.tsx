"use client";

import {
  Info,
  Brain,
  GitBranch,
  Zap,
  BookOpen,
  ExternalLink,
  Code2,
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="p-8 max-w-[900px] mx-auto">
      <header className="mb-10">
        <div className="flex items-center gap-3 mb-1">
          <Info className="w-5 h-5 text-accent" />
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            About FlyMind
          </h1>
        </div>
        <p className="text-sm text-text-muted mt-1">
          Methodology and platform overview
        </p>
      </header>

      <div className="space-y-8">
        <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
          <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-accent" />
            Overview
          </h2>
          <div className="text-sm text-text-secondary leading-relaxed space-y-3">
            <p>
              FlyMind is a machine learning platform for analyzing neural connectivity
              in the <em>Drosophila melanogaster</em> brain. It provides predictive models
              for synaptic connections between neurons, enabling researchers to explore
              the connectome at scale.
            </p>
            <p>
              The platform combines structural features of neurons—such as morphology,
              spatial overlap, and connectivity patterns—with graph-based representations
              to predict the existence and strength of synaptic connections.
            </p>
          </div>
        </section>

        <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
          <h2 className="text-base font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)] flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-accent" />
            Pipeline Architecture
          </h2>
          <div className="space-y-3">
            {[
              {
                step: "Data Ingestion",
                desc: "Load connectome graph from the FlyWire dataset, including neuron metadata and synaptic partners.",
                icon: Code2,
              },
              {
                step: "Feature Engineering",
                desc: "Extract structural features: morphological descriptors, spatial proximity metrics, and topological graph features.",
                icon: Brain,
              },
              {
                step: "Model Training",
                desc: "Train gradient-boosted classifiers on labeled positive/negative connection pairs with cross-validation.",
                icon: Zap,
              },
              {
                step: "Evaluation",
                desc: "Assess model performance using accuracy, precision, recall, F1 score, and AUC-ROC on held-out test data.",
                icon: GitBranch,
              },
            ].map((item, i) => (
              <div
                key={i}
                className="flex items-start gap-3 p-3 rounded-[var(--radius-sm)] bg-elevated"
              >
                <div className="w-7 h-7 rounded-full bg-accent/10 flex items-center justify-center shrink-0">
                  <span className="text-xs font-bold text-accent">{i + 1}</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-text-primary">
                    {item.step}
                  </span>
                  <p className="text-xs text-text-muted mt-0.5">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
          <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
            <Brain className="w-4 h-4 text-violet" />
            Methodology
          </h2>
          <div className="text-sm text-text-secondary leading-relaxed space-y-3">
            <p>
              The prediction model operates on pairs of neurons (source, target) and
              classifies whether a directed synaptic connection exists. Features are
              computed per-pair and include:
            </p>
            <ul className="list-none space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-accent mt-0.5">—</span>
                <span>Spatial overlap: number of synapses between the neuron arbors</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent mt-0.5">—</span>
                <span>Morphological similarity: shared branching patterns and arbor volumes</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent mt-0.5">—</span>
                <span>Graph features: shared neighbors, Jaccard index, Adamic-Adar score</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-accent mt-0.5">—</span>
                <span>Neuron type encoding: categorical embeddings for cell type annotations</span>
              </li>
            </ul>
            <p>
              Training uses stratified 5-fold cross-validation with early stopping.
              The final model is an ensemble of gradient-boosted decision trees, optimized
              for AUC-ROC while maintaining high precision at the operating threshold.
            </p>
          </div>
        </section>

        <section className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
          <h2 className="text-base font-semibold text-text-primary mb-3 font-[family-name:var(--font-display)] flex items-center gap-2">
            <Zap className="w-4 h-4 text-violet" />
            Tech Stack
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: "Frontend", value: "Next.js 15, React 19, Tailwind CSS 4" },
              { label: "Backend", value: "Python, FastAPI" },
              { label: "ML", value: "scikit-learn, XGBoost, PyTorch" },
              { label: "Data", value: "FlyWire connectome dataset" },
              { label: "Visualization", value: "Framer Motion, Lucide Icons" },
              { label: "Language", value: "TypeScript, Python" },
            ].map((item) => (
              <div key={item.label} className="p-3 rounded-[var(--radius-sm)] bg-elevated">
                <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium">
                  {item.label}
                </span>
                <p className="text-xs text-text-secondary mt-0.5">{item.value}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
