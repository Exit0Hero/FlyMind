"use client";

import { useState } from "react";
import { Trophy, Loader2, AlertCircle, ArrowUpDown } from "lucide-react";
import ScoreBar from "@/components/ScoreBar";
import Badge from "@/components/Badge";
import { getCandidates } from "@/lib/api";
import type { CandidateItem } from "@/lib/types";

export default function CandidatesPage() {
  const [source, setSource] = useState("");
  const [numCandidates, setNumCandidates] = useState(10);
  const [poolSize, setPoolSize] = useState(1000);
  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [totalSampled, setTotalSampled] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    const srcId = parseInt(source.trim(), 10);
    if (isNaN(srcId)) {
      setError("Please enter a valid numeric neuron root ID");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await getCandidates(srcId, numCandidates, poolSize);
      setCandidates(data.candidates);
      setTotalSampled(data.total_sampled);
    } catch {
      setError("Failed to fetch candidates. Check the neuron ID and try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <Trophy className="w-5 h-5 text-violet" />
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Candidate Ranking
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Find the most likely connected neurons for a given source
        </p>
      </header>

      <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <div className="flex items-end gap-4">
          <div className="flex-1">
            <label className="block text-[11px] uppercase tracking-wider text-text-muted font-medium mb-1.5">
              Source Neuron Root ID
            </label>
            <input
              type="text"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g., 720575940597856265"
              className="w-full px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary font-[family-name:var(--font-mono)] placeholder:text-text-muted outline-none focus:border-violet transition-colors"
            />
          </div>
          <div className="w-28">
            <label className="block text-[11px] uppercase tracking-wider text-text-muted font-medium mb-1.5">
              Top N
            </label>
            <input
              type="number"
              min={1}
              max={100}
              value={numCandidates}
              onChange={(e) => setNumCandidates(Number(e.target.value))}
              className="w-full px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary font-[family-name:var(--font-mono)] outline-none focus:border-violet transition-colors"
            />
          </div>
          <div className="w-32">
            <label className="block text-[11px] uppercase tracking-wider text-text-muted font-medium mb-1.5">
              Pool Size
            </label>
            <input
              type="number"
              min={100}
              max={10000}
              value={poolSize}
              onChange={(e) => setPoolSize(Number(e.target.value))}
              className="w-full px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary font-[family-name:var(--font-mono)] outline-none focus:border-violet transition-colors"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !source.trim()}
            className="px-5 py-2.5 rounded-[var(--radius-sm)] bg-violet/15 text-violet text-sm font-medium hover:bg-violet/25 transition-colors cursor-pointer disabled:opacity-50 border border-violet/20 flex items-center gap-2"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowUpDown className="w-4 h-4" />}
            Rank
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-error/10 border border-error/20 rounded-[var(--radius-md)] p-4 text-sm text-error mb-6 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {candidates.length > 0 ? (
        <>
          <div className="text-xs text-text-muted mb-4">
            Sampled {totalSampled.toLocaleString()} candidates, showing top {candidates.length}
          </div>
          <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border-subtle">
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3">Rank</th>
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3">Primary Type</th>
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3">Root ID</th>
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3">NT Type</th>
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3">Super Class</th>
                  <th className="text-left text-[11px] uppercase tracking-wider text-text-muted font-medium px-5 py-3 min-w-[200px]">Score</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => (
                  <tr
                    key={c.target_root_id}
                    className="border-b border-border-subtle last:border-0 hover:bg-elevated/50 transition-colors"
                  >
                    <td className="px-5 py-3.5">
                      <span
                        className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${
                          c.rank <= 3 ? "bg-violet/15 text-violet" : "bg-elevated text-text-muted"
                        }`}
                      >
                        {c.rank}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-sm text-text-primary font-medium">
                      {c.target_primary_type || "—"}
                    </td>
                    <td className="px-5 py-3.5 text-xs font-[family-name:var(--font-mono)] text-text-muted">
                      {c.target_root_id}
                    </td>
                    <td className="px-5 py-3.5">
                      {c.target_nt_type ? <Badge variant="info">{c.target_nt_type}</Badge> : <span className="text-xs text-text-muted">—</span>}
                    </td>
                    <td className="px-5 py-3.5">
                      {c.target_super_class ? <Badge variant="muted">{c.target_super_class}</Badge> : <span className="text-xs text-text-muted">—</span>}
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-3">
                        <div className="flex-1">
                          <ScoreBar score={c.score} variant="violet" showValue={false} height={5} />
                        </div>
                        <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary min-w-[40px] text-right">
                          {(c.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        !loading && !error && (
          <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-12 text-center">
            <Trophy className="w-8 h-8 text-text-muted mx-auto mb-2" />
            <p className="text-sm text-text-muted">Enter a source neuron to find candidate connections</p>
          </div>
        )
      )}
    </div>
  );
}
