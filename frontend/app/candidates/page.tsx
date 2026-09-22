"use client";

import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ListOrdered,
  Loader2,
  AlertCircle,
  ChevronDown,
  Search,
  Trophy,
} from "lucide-react";
import PageShell from "@/components/PageShell";
import NeuronSelect from "@/components/NeuronSelect";
import ScoreBar from "@/components/ScoreBar";
import Badge from "@/components/Badge";
import ErrorState from "@/components/ErrorState";
import { getCandidates } from "@/lib/api";
import type { CandidateItem } from "@/lib/types";

const K_OPTIONS = [10, 25, 50, 100];

type SortKey = "rank" | "score" | "type";

export default function CandidatesPage() {
  const [source, setSource] = useState<number | null>(null);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [k, setK] = useState(10);
  const [poolSize] = useState(1000);
  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [totalSampled, setTotalSampled] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);

  const handleRank = async () => {
    if (source === null) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getCandidates(source, k, poolSize);
      setCandidates(data.candidates);
      setTotalSampled(data.total_sampled);
      setExpanded(null);
    } catch {
      setError("Unable to load candidates. Please try again.");
      setCandidates([]);
    } finally {
      setLoading(false);
    }
  };

  const filtered = useMemo(() => {
    let rows = [...candidates];
    if (filter.trim()) {
      const q = filter.trim().toLowerCase();
      rows = rows.filter(
        (c) =>
          c.target_primary_type?.toLowerCase().includes(q) ||
          c.target_root_id.toString().includes(q) ||
          c.target_nt_type?.toLowerCase().includes(q)
      );
    }
    rows.sort((a, b) => {
      let cmp = 0;
      if (sort === "rank") cmp = a.rank - b.rank;
      else if (sort === "score") cmp = a.score - b.score;
      else cmp = (a.target_primary_type || "").localeCompare(b.target_primary_type || "");
      return sortAsc ? cmp : -cmp;
    });
    return rows;
  }, [candidates, filter, sort, sortAsc]);

  const toggleSort = (key: SortKey) => {
    if (sort === key) setSortAsc((a) => !a);
    else {
      setSort(key);
      setSortAsc(key !== "rank");
    }
    setExpanded(null);
  };

  const sortArrow = (key: SortKey) =>
    sort === key ? (sortAsc ? "↑" : "↓") : "";

  return (
    <PageShell>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-success/10 border border-success/20 flex items-center justify-center">
            <ListOrdered className="w-4 h-4 text-success" />
          </div>
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Candidate Ranking
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Rank model-suggested candidate connections for a source neuron
        </p>
      </header>

      {/* Controls */}
      <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-end gap-4 flex-wrap">
          <div className="flex-1 min-w-[240px]">
            <NeuronSelect
              label="Source Neuron"
              value={source}
              onSelect={(id, name) => {
                setSource(id === -1 ? null : id);
                setSourceName(id === -1 ? null : name);
                setError(null);
              }}
              placeholder="Search source neuron..."
              accent="violet"
            />
          </div>

          <div>
            <span className="text-caption block mb-1.5">Number of Candidates</span>
            <div className="flex gap-1.5 flex-wrap">
              {K_OPTIONS.map((opt) => (
                <button
                  key={opt}
                  onClick={() => setK(opt)}
                  aria-pressed={k === opt}
                  className={`px-3 py-2 rounded-[var(--radius-sm)] border text-sm font-medium transition-colors cursor-pointer ${
                    k === opt
                      ? "bg-violet/15 text-violet border-violet/30"
                      : "bg-elevated text-text-secondary border-border-subtle hover:border-border-strong"
                  }`}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleRank}
            disabled={source === null || loading}
            className="btn btn-violet"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trophy className="w-4 h-4" />}
            Rank Candidates
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6">
          <ErrorState message={error} onRetry={handleRank} />
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="space-y-2" role="status" aria-label="Generating candidates">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-14 skeleton rounded-[var(--radius-md)]" />
          ))}
        </div>
      )}

      {!loading && candidates.length > 0 && (
        <>
          {/* Summary + tools */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <p className="text-xs text-text-muted">
              {sourceName ? `Source: ${sourceName}` : `Source: ${source}`} — sampled{" "}
              {totalSampled.toLocaleString()} candidates, showing{" "}
              {filtered.length === candidates.length ? `top ${candidates.length}` : `${filtered.length} of ${candidates.length}`}
            </p>
            <div className="relative w-full sm:w-56">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
              <input
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Filter by type, NT, or ID"
                aria-label="Filter candidates"
                className="field pl-8 py-2 text-xs"
              />
            </div>
          </div>

          <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border-subtle bg-elevated/50">
                    <th className="text-left px-4 py-3">
                      <button onClick={() => toggleSort("rank")} className="text-[11px] uppercase tracking-wider text-text-muted hover:text-text-secondary font-medium cursor-pointer">
                        Rank {sortArrow("rank")}
                      </button>
                    </th>
                    <th className="text-left px-4 py-3">
                      <button onClick={() => toggleSort("type")} className="text-[11px] uppercase tracking-wider text-text-muted hover:text-text-secondary font-medium cursor-pointer">
                        Target {sortArrow("type")}
                      </button>
                    </th>
                    <th className="text-left px-4 py-3 text-[11px] uppercase tracking-wider text-text-muted font-medium">
                      Root ID
                    </th>
                    <th className="text-left px-4 py-3 text-[11px] uppercase tracking-wider text-text-muted font-medium">
                      NT
                    </th>
                    <th className="text-left px-4 py-3 min-w-[180px]">
                      <button onClick={() => toggleSort("score")} className="text-[11px] uppercase tracking-wider text-text-muted hover:text-text-secondary font-medium cursor-pointer">
                        Score {sortArrow("score")}
                      </button>
                    </th>
                    <th className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody>
                  <AnimatePresence initial={false}>
                    {filtered.map((c) => {
                      const isOpen = expanded === c.target_root_id;
                      return (
                        <motion.tr
                          key={c.target_root_id}
                          layout
                          initial={{ opacity: 0, y: 6 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="border-b border-border-subtle last:border-0 hover:bg-elevated/40 transition-colors"
                        >
                          <td className="px-4 py-3">
                            <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold ${
                              c.rank <= 3 ? "bg-violet/15 text-violet" : "bg-elevated text-text-muted"
                            }`}>
                              {c.rank}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm text-text-primary font-medium">
                            {c.target_primary_type || "—"}
                          </td>
                          <td className="px-4 py-3 text-xs font-[family-name:var(--font-mono)] text-text-muted">
                            {c.target_root_id}
                          </td>
                          <td className="px-4 py-3">
                            {c.target_nt_type ? <Badge variant="info">{c.target_nt_type}</Badge> : <span className="text-xs text-text-muted">—</span>}
                          </td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <div className="flex-1 min-w-[100px]">
                                <ScoreBar score={c.score} variant="violet" showValue={false} height={5} />
                              </div>
                              <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary min-w-[40px] text-right">
                                {(c.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <button
                              onClick={() => setExpanded(isOpen ? null : c.target_root_id)}
                              aria-label={isOpen ? "Collapse row" : "Expand row"}
                              aria-expanded={isOpen}
                              className="p-1.5 rounded-[var(--radius-sm)] text-text-muted hover:text-accent hover:bg-elevated transition-colors cursor-pointer"
                            >
                              <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${isOpen ? "rotate-180 text-accent" : ""}`} />
                            </button>
                          </td>
                          {isOpen && (
                            <motion.tr
                              key={`${c.target_root_id}-detail`}
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className="bg-elevated/40"
                            >
                              <td colSpan={6} className="px-4 py-3">
                                <div className="flex flex-wrap items-center gap-3 text-xs text-text-secondary">
                                  <span>
                                    <span className="text-caption mr-1.5">Target</span>
                                    {String(c.target_root_id)}
                                  </span>
                                  {c.target_super_class && (
                                    <span>
                                      <span className="text-caption mr-1.5">Super class</span>
                                      {c.target_super_class}
                                    </span>
                                  )}
                                  <span>
                                    <span className="text-caption mr-1.5">Score</span>
                                    <span className="font-[family-name:var(--font-mono)] text-violet">
                                      {(c.score * 100).toFixed(1)}%
                                    </span>
                                  </span>
                                  <span className="text-text-muted italic">
                                    Model-suggested candidate connection
                                  </span>
                                </div>
                              </td>
                            </motion.tr>
                          )}
                        </motion.tr>
                      );
                    })}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-4 p-4 rounded-[var(--radius-md)] border border-border-subtle bg-elevated flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-text-muted shrink-0 mt-0.5" />
            <p className="text-[11px] text-text-muted leading-relaxed">
              Rankings reflect the model's learned scoring of candidate targets. These are{" "}
              <strong className="text-text-primary">model-suggested candidate connections</strong> for further
              investigation — not confirmed biological connections.
            </p>
          </div>
        </>
      )}

      {!loading && !error && candidates.length === 0 && (
        <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-12 text-center">
          <ListOrdered className="w-8 h-8 text-text-muted mx-auto mb-2" />
          <p className="text-sm text-text-muted">
            Select a source neuron and choose the number of candidates to run ranking
          </p>
        </div>
      )}
    </PageShell>
  );
}