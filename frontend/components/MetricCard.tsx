"use client";

import { type LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  accent?: "accent" | "violet" | "success" | "warning" | "error";
  subtitle?: string;
}

const accentMap = {
  accent: {
    iconBg: "bg-accent/10",
    iconColor: "text-accent",
    glow: "glow-accent",
  },
  violet: {
    iconBg: "bg-violet/10",
    iconColor: "text-violet",
    glow: "glow-violet",
  },
  success: {
    iconBg: "bg-success/10",
    iconColor: "text-success",
    glow: "",
  },
  warning: {
    iconBg: "bg-warning/10",
    iconColor: "text-warning",
    glow: "",
  },
  error: {
    iconBg: "bg-error/10",
    iconColor: "text-error",
    glow: "",
  },
};

export default function MetricCard({
  label,
  value,
  icon: Icon,
  accent = "accent",
  subtitle,
}: MetricCardProps) {
  const a = accentMap[accent];

  return (
    <div
      className={`bg-surface border border-border-subtle rounded-[var(--radius-md)] p-5 card-interactive ${a.glow}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className={`w-9 h-9 rounded-[var(--radius-sm)] ${a.iconBg} flex items-center justify-center`}
        >
          <Icon className={`w-4.5 h-4.5 ${a.iconColor}`} />
        </div>
      </div>
      <div className="font-[family-name:var(--font-display)] text-2xl font-bold text-text-primary tracking-tight">
        {value}
      </div>
      <div className="text-sm text-text-secondary mt-1">{label}</div>
      {subtitle && (
        <div className="text-xs text-text-muted mt-1.5">{subtitle}</div>
      )}
    </div>
  );
}
