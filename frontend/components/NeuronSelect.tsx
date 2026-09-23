"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Loader2, Check, ChevronDown, X } from "lucide-react";
import { searchNeurons } from "@/lib/api";
import type { NeuronSearchItem } from "@/lib/types";

interface NeuronSelectProps {
  label: string;
  value: string | null;
  onSelect: (rootId: string | null, name: string | null) => void;
  placeholder?: string;
  accent?: "accent" | "violet";
  disabled?: boolean;
}

export default function NeuronSelect({
  label,
  value,
  onSelect,
  placeholder = "Search by name, root ID, or type",
  accent = "violet",
  disabled = false,
}: NeuronSelectProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState<NeuronSearchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const isSelected = value !== null;

  useEffect(() => {
    if (value === null) setSelectedName(null);
  }, [value]);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, []);

  const doSearch = (q: string) => {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    searchNeurons(q.trim(), 12)
      .then((d) => {
        setResults(d.results);
        if (d.results.length === 0) {
          setError(`No neurons found for "${q}"`);
        }
      })
      .catch(() => {
        setError("Unable to search neurons right now.");
        setResults([]);
      })
      .finally(() => setLoading(false));
  };

  const handleQueryChange = (q: string) => {
    setQuery(q);
    setOpen(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(q), 350);
  };

  const handleSelect = (item: NeuronSearchItem) => {
    onSelect(item.root_id, item.name);
    setSelectedName(item.name);
    setQuery("");
    setOpen(false);
  };

  const clear = () => {
    onSelect(null, null);
    setSelectedName(null);
    setQuery("");
    setOpen(false);
  };

  const borderAccent = accent === "violet" ? "focus:border-violet" : "focus:border-accent";
  const textAccent = accent === "violet" ? "text-violet" : "text-accent";

  return (
    <div ref={containerRef} className="relative">
      <div className="flex items-center justify-between mb-1.5">
        <label className="text-caption">{label}</label>
        {isSelected && (
          <button
            onClick={clear}
            className="text-[11px] text-text-muted hover:text-error transition-colors cursor-pointer flex items-center gap-1"
            aria-label="Clear selection"
          >
            <X className="w-3 h-3" /> Clear
          </button>
        )}
      </div>

      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          if (isSelected) return;
          setOpen((o) => !o);
          if (!query && !results.length) doSearch("Tm");
        }}
        className={`w-full text-left px-3 py-2.5 rounded-[var(--radius-sm)] bg-elevated border text-sm transition-colors cursor-pointer disabled:opacity-50 ${
          isSelected
            ? "border-success/40 bg-success/5"
            : "border-border-subtle hover:border-border-strong"
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          {isSelected ? (
            <>
              <Check className={`w-3.5 h-3.5 ${textAccent}`} />
              <span className="text-sm text-text-primary truncate">
                {selectedName || `Neuron ${value}`}
              </span>
              <span className="text-xs font-[family-name:var(--font-mono)] text-text-muted ml-auto">
                {value}
              </span>
            </>
          ) : (
            <>
              <Search className="w-3.5 h-3.5 text-text-muted" />
              <span className="text-sm text-text-muted truncate">{placeholder}</span>
              <ChevronDown className="w-3.5 h-3.5 text-text-muted ml-auto" />
            </>
          )}
        </div>
      </button>

      {open && !isSelected && (
        <div className="absolute z-30 mt-2 w-full bg-overlay border border-border-strong rounded-[var(--radius-md)] shadow-2xl overflow-hidden">
          <div className="p-2 border-b border-border-subtle">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted" />
              <input
                autoFocus
                type="text"
                value={query}
                onChange={(e) => handleQueryChange(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") doSearch(query);
                }}
                placeholder="Type to search neurons..."
                aria-label={`Search ${label.toLowerCase()}`}
                className={`w-full pl-8 pr-3 py-2 rounded-[var(--radius-sm)] bg-elevated border border-border-subtle text-sm text-text-primary placeholder:text-text-muted outline-none ${borderAccent}`}
              />
              {loading && (
                <Loader2 className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted animate-spin" />
              )}
            </div>
          </div>

          <div
            role="listbox"
            className="max-h-72 overflow-y-auto"
            aria-label={`${label} results`}
          >
            {results.length > 0 ? (
              results.map((item) => (
                <button
                  key={item.root_id}
                  role="option"
                  aria-selected="false"
                  onClick={() => handleSelect(item)}
                  className="w-full text-left px-3 py-2.5 hover:bg-elevated transition-colors cursor-pointer flex items-center gap-3"
                >
                  <div className="min-w-0 flex-1">
                    <span className="block text-sm text-text-primary truncate">
                      {item.name || `Neuron ${item.root_id}`}
                    </span>
                    <span className="block text-[11px] font-[family-name:var(--font-mono)] text-text-muted">
                      {item.root_id}
                    </span>
                  </div>
                  {item.nt_type && (
                    <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-accent/10 text-accent border border-accent/20 font-medium">
                      {item.nt_type}
                    </span>
                  )}
                  {item.primary_type && (
                    <span className="shrink-0 text-[10px] px-1.5 py-0.5 rounded bg-violet/10 text-violet border border-violet/20 font-medium">
                      {item.primary_type}
                    </span>
                  )}
                </button>
              ))
            ) : (
              <div className="px-3 py-6 text-center">
                <p className="text-xs text-text-muted">
                  {error || "Type at least 2 characters to search"}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}