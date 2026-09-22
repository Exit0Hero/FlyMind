"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Search,
  X,
  Tag,
  Dna,
  Ruler,
  MapPin,
  GitFork,
  Loader2,
  ArrowUpRight,
  Database,
} from "lucide-react";
import PageShell from "@/components/PageShell";
import Badge from "@/components/Badge";
import ErrorState from "@/components/ErrorState";
import { searchNeurons, getNeuron } from "@/lib/api";
import type { Neuron, NeuronSearchItem } from "@/lib/types";

function fmtLen(v: number | null): string {
  if (v == null) return "—";
  return `${(v / 1000).toFixed(1)} μm`;
}
function fmtArea(v: number | null): string {
  if (v == null) return "—";
  return `${(v / 1e6).toFixed(1)} μm²`;
}
function fmtVol(v: number | null): string {
  if (v == null) return "—";
  return `${(v / 1e9).toFixed(1)} μm³`;
}
function fmtDeg(v: number | null): string {
  return v == null ? "—" : String(v);
}

function KV({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <span className="text-caption">{label}</span>
      <p className={`text-sm text-text-secondary mt-0.5 truncate ${mono ? "font-[family-name:var(--font-mono)]" : ""}`}>
        {value}
      </p>
    </div>
  );
}

function GroupCard({
  icon: Icon,
  title,
  children,
  delay = 0,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  children: React.ReactNode;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
      className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-5"
    >
      <div className="flex items-center gap-2 mb-4">
        <Icon className="w-4 h-4 text-accent" />
        <h3 className="text-sm font-semibold text-text-primary font-[family-name:var(--font-display)]">
          {title}
        </h3>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">{children}</div>
    </motion.div>
  );
}

export default function NeuronsPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<NeuronSearchItem[]>([]);
  const [selected, setSelected] = useState<Neuron | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setSearched(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await searchNeurons(q.trim(), 20);
      setResults(data.results);
      setSearched(true);
    } catch {
      setError("Unable to load neuron data. Please check your search and try again.");
      setResults([]);
      setSearched(true);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleQueryChange = (q: string) => {
    setQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(q), 350);
  };

  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  const handleSelect = async (item: NeuronSearchItem) => {
    setDetailLoading(true);
    setError(null);
    try {
      const full = await getNeuron(item.root_id);
      setSelected(full);
    } catch {
      setError("Unable to load neuron data. Please try again.");
    } finally {
      setDetailLoading(false);
    }
  };

  const clear = () => {
    setQuery("");
    setResults([]);
    setSelected(null);
    setSearched(false);
    setError(null);
  };

  const ntList = selected
    ? [
        { label: "NT Type", value: selected.nt_type || "—" },
        { label: "NT Score", value: selected.nt_type_score != null ? `${(selected.nt_type_score * 100).toFixed(0)}%` : "—" },
      ]
    : [];

  return (
    <PageShell>
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-9 h-9 rounded-[var(--radius-sm)] bg-accent/10 border border-accent/20 flex items-center justify-center">
            <Brain className="w-4 h-4 text-accent" />
          </div>
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Neuron Explorer
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Search by name, root ID, or cell type — then inspect biological, morphological, and spatial properties
        </p>
      </header>

      {/* Search */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
        <input
          type="text"
          value={query}
          onChange={(e) => handleQueryChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && doSearch(query)}
          placeholder="Search neurons — e.g., Tm16, MBON, or 720575940597856265"
          aria-label="Search neurons"
          className="field pl-9 pr-9"
        />
        {query && (
          <button
            onClick={clear}
            aria-label="Clear search"
            className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {error && (
        <div className="mb-6">
          <ErrorState message={error} onRetry={() => doSearch(query)} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] gap-6">
        {/* Results list */}
        <div>
          {loading ? (
            <div className="space-y-2" role="status" aria-label="Searching neurons">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-16 skeleton rounded-[var(--radius-md)]" />
              ))}
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-text-muted mb-3">
                {results.length} match{results.length === 1 ? "" : "es"}
              </p>
              {results.map((n) => (
                <button
                  key={n.root_id}
                  onClick={() => handleSelect(n)}
                  className={`w-full text-left p-4 rounded-[var(--radius-md)] border transition-all duration-150 cursor-pointer ${
                    selected?.root_id === n.root_id
                      ? "bg-accent/5 border-accent/30"
                      : "bg-surface border-border-subtle hover:border-border-strong"
                  } ${detailLoading ? "cursor-wait" : ""}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-accent/10 flex items-center justify-center shrink-0">
                        <Brain className="w-4 h-4 text-accent" />
                      </div>
                      <div className="min-w-0">
                        <span className="text-sm font-semibold text-text-primary block truncate">
                          {n.name || `Neuron ${n.root_id}`}
                        </span>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-[11px] font-[family-name:var(--font-mono)] text-text-muted">
                            {n.root_id}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex-shrink-0 flex items-center gap-1.5">
                      {n.nt_type && <Badge variant="info">{n.nt_type}</Badge>}
                      {n.primary_type && <Badge variant="muted">{n.primary_type}</Badge>}
                      <ArrowUpRight className="w-4 h-4 text-text-muted" />
                    </div>
                  </div>
                </button>
              ))}
            </div>
          ) : !searched ? (
            <div className="text-center py-16 px-4">
              <Database className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">Search to find individual neurons in the connectome</p>
            </div>
          ) : (
            <div className="text-center py-16">
              <Search className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">No neurons found for that search</p>
            </div>
          )}
        </div>

        {/* Detail panel */}
        <div className="lg:sticky lg:top-6 self-start">
          {detailLoading ? (
            <div className="space-y-4" role="status" aria-label="Loading neuron detail">
              <div className="h-8 skeleton rounded-[var(--radius-md)] w-2/3" />
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-40 skeleton rounded-[var(--radius-lg)]" />
              ))}
            </div>
          ) : selected ? (
            <div className="space-y-4">
              <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-5">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <h2 className="text-xl font-bold text-text-primary font-[family-name:var(--font-display)]">
                    {selected.name || `Neuron ${selected.root_id}`}
                  </h2>
                  {selected.nt_type && <Badge variant="info" dot>{selected.nt_type}</Badge>}
                </div>
                <p className="text-xs font-[family-name:var(--font-mono)] text-text-muted mt-1">
                  Root ID: {selected.root_id}
                </p>
              </div>

              <GroupCard icon={Tag} title="Identity" delay={0.05}>
                <KV label="Name" value={selected.name || "—"} />
                <KV label="Primary Type" value={selected.primary_type || "—"} />
                <KV label="Super Class" value={selected.super_class || "—"} />
                <KV label="Flow" value={selected.flow || "—"} />
                <KV label="Side" value={selected.side || "—"} />
              </GroupCard>

              <GroupCard icon={Dna} title="Biology" delay={0.1}>
                {ntList.length ? (
                  ntList.map((item) => (
                    <KV key={item.label} label={item.label} value={item.value} />
                  ))
                ) : (
                  <p className="text-xs text-text-muted col-span-2">No neurotransmitter data available</p>
                )}
                <KV label="Outgoing Degree" value={fmtDeg(selected.outgoing_count)} />
                <KV label="Incoming Degree" value={fmtDeg(selected.incoming_count)} />
              </GroupCard>

              <GroupCard icon={Ruler} title="Morphology" delay={0.15}>
                <KV label="Length" value={fmtLen(selected.length_nm)} />
                <KV label="Surface Area" value={fmtArea(selected.area_nm)} />
                <KV label="Volume" value={fmtVol(selected.size_nm)} />
              </GroupCard>

              <GroupCard icon={MapPin} title="Spatial" delay={0.2}>
                <div className="col-span-2">
                  <span className="text-caption">Coordinates (nm)</span>
                  <p className="text-sm font-[family-name:var(--font-mono)] text-text-secondary mt-0.5">
                    ({selected.coord_x?.toFixed(0)} — {selected.coord_y?.toFixed(0)} — {selected.coord_z?.toFixed(0)})
                  </p>
                </div>
              </GroupCard>

              <GroupCard icon={GitFork} title="Connectivity" delay={0.25}>
                <p className="text-xs text-text-muted col-span-2 leading-relaxed">
                  Per-neuron degree totals for this neuron. Fine-grained connectivity is available through the
                  Connection Predictor and Candidate Ranking tools.
                </p>
              </GroupCard>
            </div>
          ) : (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-8 text-center">
              <Brain className="w-8 h-8 text-text-muted mx-auto mb-2" />
              <p className="text-sm text-text-muted">Select a neuron to inspect its properties</p>
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}