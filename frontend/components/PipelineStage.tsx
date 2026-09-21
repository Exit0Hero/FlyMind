"use client";

import { CheckCircle2, Loader2, Clock, AlertCircle } from "lucide-react";

interface PipelineStageProps {
  name: string;
  status: "completed" | "running" | "pending" | "error";
  description?: string;
  duration_ms?: number;
  progress?: number;
  onClick?: () => void;
}

const statusConfig = {
  completed: {
    icon: CheckCircle2,
    color: "text-success",
    bg: "bg-success/10",
    border: "border-success/30",
    label: "Completed",
  },
  running: {
    icon: Loader2,
    color: "text-accent",
    bg: "bg-accent/10",
    border: "border-accent/30",
    label: "Running",
  },
  pending: {
    icon: Clock,
    color: "text-text-muted",
    bg: "bg-elevated",
    border: "border-border-subtle",
    label: "Pending",
  },
  error: {
    icon: AlertCircle,
    color: "text-error",
    bg: "bg-error/10",
    border: "border-error/30",
    label: "Error",
  },
};

export default function PipelineStage({
  name,
  status,
  description,
  duration_ms,
  progress,
  onClick,
}: PipelineStageProps) {
  const cfg = statusConfig[status];
  const Icon = cfg.icon;

  return (
    <button
      onClick={onClick}
      className={`
        w-full text-left p-4 rounded-[var(--radius-md)] border ${cfg.border}
        ${cfg.bg} transition-all duration-200 cursor-pointer
        hover:border-opacity-60 card-interactive
      `}
    >
      <div className="flex items-start gap-3">
        <div
          className={`w-8 h-8 rounded-[var(--radius-sm)] flex items-center justify-center shrink-0 ${cfg.bg}`}
        >
          <Icon
            className={`w-4 h-4 ${cfg.color} ${
              status === "running" ? "animate-spin" : ""
            }`}
          />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-text-primary">
              {name}
            </span>
            <span className={`text-[10px] font-medium uppercase ${cfg.color}`}>
              {cfg.label}
            </span>
          </div>
          {description && (
            <p className="text-xs text-text-muted mt-1 line-clamp-2">
              {description}
            </p>
          )}
          <div className="flex items-center gap-3 mt-2">
            {duration_ms !== undefined && (
              <span className="text-[11px] font-[family-name:var(--font-mono)] text-text-muted">
                {duration_ms >= 1000
                  ? `${(duration_ms / 1000).toFixed(1)}s`
                  : `${duration_ms}ms`}
              </span>
            )}
            {progress !== undefined && status === "running" && (
              <div className="flex-1 max-w-[120px] h-1.5 bg-base rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </button>
  );
}
