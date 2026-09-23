"use client";

import { motion } from "framer-motion";
import { Check, ChevronRight, X, Loader2, Layers, Database, Workflow, Boxes, Brain, TestTubes, Zap, Crown } from "lucide-react";

export interface FlowStage {
  key: string;
  title: string;
  status: "ready" | "pending" | "error";
  description: string;
  artifact?: string;
  metric?: { label: string; value: string };
}

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  data: Database,
  etl: Workflow,
  features: Boxes,
  model: Brain,
  evaluation: TestTubes,
  prediction: Zap,
  ranking: Crown,
};

interface PipelineFlowProps {
  stages: FlowStage[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}

export default function PipelineFlow({ stages, selectedKey, onSelect }: PipelineFlowProps) {
  const readyCount = stages.filter((s) => s.status === "ready").length;

  return (
    <div className="relative">
      {/* vertical connector line */}
      <div className="absolute left-[19px] top-6 bottom-6 w-px bg-border-subtle hidden sm:block" />

      <div className="space-y-3">
        {stages.map((stage, idx) => {
          const Icon = iconMap[stage.key] ?? Layers;
          const isSelected = selectedKey === stage.key;
          const isReady = stage.status === "ready";

          const statusDot = isReady ? (
            <Check className="w-3.5 h-3.5 text-success" />
          ) : stage.status === "error" ? (
            <X className="w-3.5 h-3.5 text-error" />
          ) : (
            <Loader2 className="w-3.5 h-3.5 text-text-muted animate-spin" />
          );

          const dotBg = isReady
            ? "bg-success/15 text-success border-success/30"
            : stage.status === "error"
            ? "bg-error/15 text-error border-error/30"
            : "bg-elevated text-text-muted border-border-subtle";

          return (
            <motion.div
              key={stage.key}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.35, delay: idx * 0.08 }}
              className="relative pl-0 sm:pl-14"
            >
              <div className="hidden sm:flex items-center justify-center absolute left-0 top-1/2 -translate-y-1/2 w-6 h-6">
                <span className={`connector-dot ${dotBg} flex items-center justify-center`}>
                  {statusDot}
                </span>
              </div>

              <button
                onClick={() => onSelect(stage.key)}
                aria-pressed={isSelected}
                className={`w-full text-left rounded-[var(--radius-md)] border p-4 transition-all duration-200 cursor-pointer ${
                  isSelected
                    ? "border-accent/40 bg-accent/5 shadow-[0_0_24px_rgba(79,209,197,0.08)]"
                    : "bg-surface border-border-subtle hover:border-border-strong hover:bg-elevated/50"
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-9 h-9 rounded-[var(--radius-sm)] flex items-center justify-center shrink-0 ${
                    isReady ? "bg-accent/10 text-accent" : "bg-elevated text-text-muted"
                  }`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-text-primary">
                        {stage.title}
                      </span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded-full border font-medium uppercase tracking-wide ${
                          isReady
                            ? "bg-success/10 text-success border-success/20"
                            : stage.status === "error"
                            ? "bg-error/10 text-error border-error/20"
                            : "bg-elevated text-text-muted border-border-subtle"
                        }`}
                      >
                        {stage.status}
                      </span>
                    </div>
                    <p className="text-xs text-text-muted mt-0.5 line-clamp-1">
                      {stage.description}
                    </p>
                  </div>
                  <ChevronRight
                    className={`w-4 h-4 shrink-0 transition-transform ${
                      isSelected ? "rotate-90 text-accent" : "text-text-muted"
                    }`}
                  />
                </div>
              </button>

              {isSelected && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="overflow-hidden"
                >
                  <div className="ml-0 sm:ml-14 mt-2 rounded-[var(--radius-md)] bg-elevated border border-border-subtle p-4">
                    <p className="text-sm text-text-secondary leading-relaxed">
                      {stage.description}
                    </p>
                    <div className="flex flex-wrap gap-4 mt-3">
                      {stage.artifact && (
                        <div>
                          <span className="text-caption">Artifact</span>
                          <p className="text-xs font-[family-name:var(--font-mono)] text-text-secondary mt-0.5">
                            {stage.artifact}
                          </p>
                        </div>
                      )}
                      {stage.metric && (
                        <div>
                          <span className="text-caption">Key Result</span>
                          <div className="flex items-center gap-2 mt-0.5">
                            <span className="text-sm font-[family-name:var(--font-display)] font-bold text-accent">
                              {stage.metric.value}
                            </span>
                            <span className="text-xs text-text-muted">
                              {stage.metric.label}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>

      <div className="mt-5 flex items-center gap-3 text-xs text-text-muted">
        <div className="h-1.5 flex-1 bg-elevated rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-success to-accent rounded-full transition-all duration-700"
            style={{ width: `${(readyCount / stages.length) * 100}%` }}
          />
        </div>
        <span>
          {readyCount}/{stages.length} ready
        </span>
      </div>
    </div>
  );
}