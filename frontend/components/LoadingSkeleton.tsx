"use client";

import { Loader2 } from "lucide-react";

interface LoadingSkeletonProps {
  variant?: "cards" | "list" | "text" | "chart" | "pipeline";
  count?: number;
  label?: string;
}

export default function LoadingSkeleton({
  variant = "cards",
  count = 4,
  label,
}: LoadingSkeletonProps) {
  const sizes =
    variant === "list"
      ? "h-16"
      : variant === "chart"
      ? "h-40"
      : variant === "pipeline"
      ? "h-24"
      : variant === "text"
      ? "h-4"
      : "h-32";

  return (
    <div role="status" aria-live="polite" aria-label={label ?? "Loading data"}>
      {label && (
        <div className="flex items-center gap-2 text-sm text-text-muted mb-4">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          {label}
        </div>
      )}
      <div
        className={`grid gap-4 ${
          variant === "list" || variant === "pipeline"
            ? "grid-cols-1"
            : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
        }`}
      >
        {[...Array(count)].map((_, i) => (
          <div key={i} className={`${sizes} skeleton rounded-[var(--radius-md)]`} />
        ))}
      </div>
    </div>
  );
}