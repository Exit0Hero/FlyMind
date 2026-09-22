"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Zap,
  ArrowDown,
  Loader2,
  AlertCircle,
  Info,
  ShieldCheck,
  Cpu,
} from "lucide-react";
import PageShell from "@/components/PageShell";
import NeuronSelect from "@/components/NeuronSelect";
import ScoreBar from "@/components/ScoreBar";
import Badge from "@/components/Badge";
import ErrorState from "@/components/ErrorState";
import { predictConnection } from "@/lib/api";
import type { PredictionResult } from "@/lib/types";

export default function PredictorPage() {
  const [source, setSource] = useState<number | null>(null);
  const [target, setTarget] = useState<number | null>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [targetName, setTargetName] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canPredict = source !== null && target !== null && !loading;

  const handlePredict = async () => {
    if (source === null || target === null) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictConnection(source, target);
      setResult(data);
    } catch {
      setError("Prediction could not be completed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-violet/10 border border-violet/20 flex items-center justify-center">
            <Zap className="w-4 h-4 text-violet" />
          </div>
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Connection Predictor
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Score the model-suggested connection between two neurons
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] gap-8">
        {/* Input column */}
        <div>
          <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
            <h2 className="text-base font-semibold text-text-primary mb-5 font-[family-name:var(--font-display)]">
              Select Source & Target
            </h2>

            <NeuronSelect
              label="Source Neuron"
              value={source}
              onSelect={(id, name) => {
                setSource(id === -1 ? null : id);
                setSourceName(id === -1 ? null : name);
              }}
              placeholder="Search source neuron..."
              accent="violet"
            />

            <div className="flex items-center justify-center py-3">
              <div className="w-8 h-8 rounded-full bg-elevated border border-border-subtle flex items-center justify-center">
                <ArrowDown className="w-4 h-4 text-text-muted" />
              </div>
            </div>

            <NeuronSelect
              label="Target Neuron"
              value={target}
              onSelect={(id, name) => {
                setTarget(id === -1 ? null : id);
                setTargetName(id === -1 ? null : name);
              }}
              placeholder="Search target neuron..."
              accent="violet"
            />

            <button
              onClick={handlePredict}
              disabled={!canPredict}
              className="btn btn-violet w-full mt-6"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running prediction...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Run Prediction
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mt-4">
              <ErrorState message={error} onRetry={handlePredict} />
            </div>
          )}

          <div className="mt-4 p-4 rounded-[var(--radius-md)] border border-border-subtle bg-elevated flex items-start gap-2.5">
            <Info className="w-4 h-4 text-text-muted shrink-0 mt-0.5" />
            <p className="text-[11px] text-text-muted leading-relaxed">
              A higher score indicates stronger similarity to patterns learned from observed connectivity.
              It does not prove that a biological connection exists.
            </p>
          </div>
        </div>

        {/* Result column */}
        <div className="lg:sticky lg:top-6 self-start">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-10 text-center"
                role="status"
                aria-live="polite"
              >
                <Loader2 className="w-6 h-6 text-violet animate-spin mx-auto mb-4" />
                <p className="text-sm text-text-muted">Scoring the pair with the random forest…</p>
              </motion.div>
            ) : result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 glow-violet"
              >
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-base font-semibold text-text-primary font-[family-name:var(--font-display)]">
                    Model-Suggested Connection Score
                  </h2>
                  <Badge variant="info">RF · {Math.round(result.score * 100)}%</Badge>
                </div>

                <div className="text-center mb-6">
                  <motion.div
                    initial={{ scale: 0.7, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", stiffness: 260, damping: 20, delay: 0.15 }}
                    className="font-[family-name:var(--font-display)] text-6xl sm:text-7xl font-bold text-violet tracking-tight"
                    aria-label={`Connection score ${Math.round(result.score * 100)} percent`}
                  >
                    {Math.round(result.score * 100)}
                  </motion.div>
                  <div className="text-sm text-text-muted mt-1">out of 100</div>
                  <div className="max-w-xs mx-auto mt-6">
                    <ScoreBar score={result.score} variant="violet" label="Score" />
                  </div>
                </div>

                <div className="border-t border-border-subtle pt-5 space-y-2.5">
                  <InfoRow
                    label="Source Neuron"
                    value={sourceName ? `${sourceName} · ${result.source_root_id}` : String(result.source_root_id)}
                  />
                  <InfoRow
                    label="Target Neuron"
                    value={targetName ? `${targetName} · ${result.target_root_id}` : String(result.target_root_id)}
                  />
                  <InfoRow
                    label="Observed Connection"
                    value={
                      <Badge variant={result.is_known_edge ? "success" : "muted"}>
                        {result.is_known_edge ? "Observed in dataset" : "Not observed"}
                      </Badge>
                    }
                  />
                  <InfoRow
                    label="Self Loop"
                    value={<Badge variant={result.is_self_loop ? "warning" : "muted"}>{result.is_self_loop ? "Yes" : "No"}</Badge>}
                  />
                  <InfoRow
                    label="Model"
                    value={
                      <span className="flex items-center gap-1.5 text-xs font-[family-name:var(--font-mono)] text-text-secondary">
                        <Cpu className="w-3 h-3 text-accent" /> Random Forest · 100 trees
                      </span>
                    }
                  />
                </div>

                <div className="mt-5 p-3.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle flex items-start gap-2">
                  <ShieldCheck className="w-4 h-4 text-text-muted shrink-0 mt-0.5" />
                  <p className="text-[11px] text-text-muted leading-relaxed">
                    This score is derived from patterns in observed connectivity. It is a research hypothesis,
                    not a biological confirmation of a synaptic connection.
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-10 text-center h-full"
              >
                <Zap className="w-8 h-8 text-text-muted mx-auto mb-3" />
                <p className="text-sm text-text-muted">
                  Select a source and target neuron to run a prediction
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {error && <AlertCircle className="hidden" />}
    </PageShell>
  );
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-xs text-text-muted shrink-0">{label}</span>
      <span className="text-xs text-text-secondary text-right min-w-0 truncate">{value}</span>
    </div>
  );
}