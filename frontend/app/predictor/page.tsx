"use client";

import { useState } from "react";
import { Zap, ArrowRight, Loader2, AlertCircle } from "lucide-react";
import ScoreBar from "@/components/ScoreBar";
import Badge from "@/components/Badge";
import { predictConnection } from "@/lib/api";
import type { PredictionResult } from "@/lib/types";

export default function PredictorPage() {
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    const srcId = parseInt(source.trim(), 10);
    const tgtId = parseInt(target.trim(), 10);
    if (isNaN(srcId) || isNaN(tgtId)) {
      setError("Please enter valid numeric neuron root IDs");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictConnection(srcId, tgtId);
      setResult(data);
    } catch {
      setError("Prediction failed. Check neuron IDs and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <Zap className="w-5 h-5 text-violet" />
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Connection Predictor
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Predict the model-suggested connection score between two neurons
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_420px] gap-8">
        <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6">
          <h2 className="text-base font-semibold text-text-primary mb-4 font-[family-name:var(--font-display)]">
            Input Neurons
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-[11px] uppercase tracking-wider text-text-muted font-medium mb-1.5">
                Source Neuron Root ID
              </label>
              <input
                type="text"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="e.g., 720575940597856265"
                className="w-full px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary font-[family-name:var(--font-mono)] placeholder:text-text-muted outline-none focus:border-violet transition-colors"
              />
            </div>

            <div className="flex items-center justify-center">
              <div className="w-8 h-8 rounded-full bg-elevated border border-border-subtle flex items-center justify-center">
                <ArrowRight className="w-4 h-4 text-text-muted" />
              </div>
            </div>

            <div>
              <label className="block text-[11px] uppercase tracking-wider text-text-muted font-medium mb-1.5">
                Target Neuron Root ID
              </label>
              <input
                type="text"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="e.g., 720575940602380768"
                className="w-full px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary font-[family-name:var(--font-mono)] placeholder:text-text-muted outline-none focus:border-violet transition-colors"
              />
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !source.trim() || !target.trim()}
              className="w-full py-2.5 rounded-[var(--radius-sm)] bg-violet/15 text-violet text-sm font-medium hover:bg-violet/25 transition-colors cursor-pointer disabled:opacity-50 border border-violet/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Predict Connection
                </>
              )}
            </button>
          </div>
        </div>

        <div>
          {error && (
            <div className="bg-error/10 border border-error/20 rounded-[var(--radius-md)] p-4 text-sm text-error mb-4 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}

          {result ? (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 glow-violet">
              <h2 className="text-base font-semibold text-text-primary mb-5 font-[family-name:var(--font-display)]">
                Prediction Result
              </h2>

              <div className="text-center mb-6">
                <div className="font-[family-name:var(--font-display)] text-6xl font-bold text-violet mb-1">
                  {Math.round(result.score * 100)}%
                </div>
                <div className="text-sm text-text-muted">Model-Suggested Connection Score</div>
              </div>

              <div className="space-y-4 mb-6">
                <ScoreBar score={result.score} variant="violet" label="Score" />
              </div>

              <div className="border-t border-border-subtle pt-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Known Connection</span>
                    <Badge variant={result.is_known_edge ? "success" : "muted"}>
                      {result.is_known_edge ? "Yes" : "No"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Self Loop</span>
                    <Badge variant={result.is_self_loop ? "warning" : "muted"}>
                      {result.is_self_loop ? "Yes" : "No"}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Source</span>
                    <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary">
                      {result.source_root_id}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">Target</span>
                    <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary">
                      {result.target_root_id}
                    </span>
                  </div>
                </div>
              </div>

              <div className="mt-4 p-3 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle">
                <p className="text-[11px] text-text-muted leading-relaxed">
                  This score is a model-suggested hypothesis. It does NOT constitute biological confirmation of a connection.
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 text-center">
              <Zap className="w-8 h-8 text-text-muted mx-auto mb-2" />
              <p className="text-sm text-text-muted">Enter neuron IDs and run prediction to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
