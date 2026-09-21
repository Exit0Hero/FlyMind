"use client";

import { useState, useCallback } from "react";
import { Brain, Search, X, MapPin, Hash, ChevronRight } from "lucide-react";
import Badge from "@/components/Badge";
import { searchNeurons, getNeuron } from "@/lib/api";
import type { Neuron, NeuronSearchItem } from "@/lib/types";

export default function NeuronsPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<NeuronSearchItem[]>([]);
  const [selected, setSelected] = useState<Neuron | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const doSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await searchNeurons(query.trim());
      setResults(data.results);
    } catch {
      setError("Failed to search neurons");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query]);

  const handleSelect = async (item: NeuronSearchItem) => {
    try {
      const full = await getNeuron(item.root_id);
      setSelected(full);
    } catch {
      setSelected(null);
    }
  };

  return (
    <div className="p-8 max-w-[1400px] mx-auto">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-1">
          <Brain className="w-5 h-5 text-accent" />
          <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] tracking-tight">
            Neuron Explorer
          </h1>
        </div>
        <p className="text-sm text-text-muted">
          Search, inspect, and analyze individual neurons in the connectome
        </p>
      </header>

      <div className="flex gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
            placeholder="Search by name, root ID, or type (e.g. Tm16, 720575940597856265)..."
            className="w-full pl-9 pr-9 py-2.5 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary placeholder:text-text-muted outline-none focus:border-accent transition-colors"
          />
          {query && (
            <button
              onClick={() => { setQuery(""); setResults([]); setError(null); setSelected(null); }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        <button
          onClick={doSearch}
          disabled={loading || !query.trim()}
          className="px-4 py-2.5 rounded-[var(--radius-sm)] bg-accent/15 text-accent text-sm font-medium hover:bg-accent/25 transition-colors cursor-pointer disabled:opacity-50 border border-accent/20"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-6">
        <div>
          {error && (
            <div className="bg-error/10 border border-error/20 rounded-[var(--radius-md)] p-4 text-sm text-error mb-4">
              {error}
            </div>
          )}

          {results.length > 0 ? (
            <div className="space-y-2">
              {results.map((neuron) => (
                <button
                  key={neuron.root_id}
                  onClick={() => handleSelect(neuron)}
                  className={`
                    w-full text-left p-4 rounded-[var(--radius-md)] border transition-all duration-150 cursor-pointer
                    ${
                      selected?.root_id === neuron.root_id
                        ? "bg-accent/5 border-accent/30"
                        : "bg-surface border-border-subtle hover:border-border-strong"
                    }
                  `}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-8 h-8 rounded-[var(--radius-sm)] bg-accent/10 flex items-center justify-center shrink-0">
                        <Brain className="w-4 h-4 text-accent" />
                      </div>
                      <div className="min-w-0">
                        <span className="text-sm font-semibold text-text-primary block truncate">
                          {neuron.name || `Neuron ${neuron.root_id}`}
                        </span>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-[11px] font-[family-name:var(--font-mono)] text-text-muted">
                            {neuron.root_id}
                          </span>
                          {neuron.nt_type && <Badge variant="info">{neuron.nt_type}</Badge>}
                          {neuron.super_class && <Badge variant="muted">{neuron.super_class}</Badge>}
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-text-muted shrink-0" />
                  </div>
                </button>
              ))}
            </div>
          ) : !loading && query ? (
            <div className="text-center py-12">
              <Brain className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">No neurons found</p>
            </div>
          ) : (
            <div className="text-center py-12">
              <Search className="w-8 h-8 text-text-muted mx-auto mb-3" />
              <p className="text-sm text-text-muted">Enter a search query to find neurons</p>
            </div>
          )}
        </div>

        <div className="lg:sticky lg:top-8 self-start">
          {selected ? (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-5">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-4.5 h-4.5 text-accent" />
                <h3 className="text-base font-semibold text-text-primary font-[family-name:var(--font-display)]">
                  Neuron Detail
                </h3>
              </div>

              <div className="space-y-3">
                <DetailRow label="Name" value={selected.name || "—"} />
                <DetailRow label="Root ID" value={String(selected.root_id)} mono />
                {selected.primary_type && <DetailRow label="Primary Type" value={selected.primary_type} />}
                {selected.nt_type && (
                  <div>
                    <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium">NT Type</span>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <Badge variant="info" dot>{selected.nt_type}</Badge>
                      {selected.nt_type_score != null && (
                        <span className="text-xs font-[family-name:var(--font-mono)] text-text-muted">
                          ({(selected.nt_type_score * 100).toFixed(0)}%)
                        </span>
                      )}
                    </div>
                  </div>
                )}
                {selected.super_class && <DetailRow label="Super Class" value={selected.super_class} />}
                {selected.flow && <DetailRow label="Flow" value={selected.flow} />}
                {selected.side && <DetailRow label="Side" value={selected.side} />}
                {(selected.coord_x != null) && (
                  <div>
                    <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium">Coordinates</span>
                    <p className="text-sm font-[family-name:var(--font-mono)] text-text-secondary mt-0.5">
                      ({selected.coord_x?.toFixed(0)}, {selected.coord_y?.toFixed(0)}, {selected.coord_z?.toFixed(0)})
                    </p>
                  </div>
                )}
                {selected.length_nm != null && (
                  <DetailRow label="Length" value={`${(selected.length_nm / 1000).toFixed(1)} μm`} mono />
                )}
                {selected.area_nm != null && (
                  <DetailRow label="Area" value={`${(selected.area_nm / 1e6).toFixed(1)} μm²`} mono />
                )}
              </div>
            </div>
          ) : (
            <div className="bg-surface border border-border-subtle rounded-[var(--radius-lg)] p-6 text-center">
              <Brain className="w-8 h-8 text-text-muted mx-auto mb-2" />
              <p className="text-sm text-text-muted">Select a neuron to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <span className="text-[11px] uppercase tracking-wider text-text-muted font-medium">{label}</span>
      <p className={`text-sm text-text-secondary mt-0.5 ${mono ? "font-[family-name:var(--font-mono)]" : ""}`}>
        {value}
      </p>
    </div>
  );
}
