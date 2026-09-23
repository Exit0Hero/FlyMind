"use client";

import { RefreshCw, AlertTriangle } from "lucide-react";

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export default function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="rounded-[var(--radius-md)] border border-error/20 bg-error/5 p-5">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-error shrink-0 mt-0.5" />
        <div className="flex-1">
          <p className="text-sm text-text-secondary">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 flex items-center gap-1.5 text-xs text-accent hover:text-accent/80 transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" /> Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}