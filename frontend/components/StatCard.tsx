"use client";

interface StatCardProps {
  label: string;
  value: string;
  sublabel?: string;
  accent?: "accent" | "violet" | "success" | "warning";
  delay?: number;
}

const accentMap = {
  accent: "text-accent border-accent/20 bg-accent/5",
  violet: "text-violet border-violet/20 bg-violet/5",
  success: "text-success border-success/20 bg-success/5",
  warning: "text-warning border-warning/20 bg-warning/5",
};

export default function StatCard({
  label,
  value,
  sublabel,
  accent = "accent",
}: StatCardProps) {
  return (
    <div
      className={`rounded-[var(--radius-lg)] border p-5 ${accentMap[accent]} backdrop-blur-sm`}
      style={{ animation: `fade-in-up 0.5s ease-out forwards` }}
    >
      <div className="font-[family-name:var(--font-display)] text-3xl font-bold tracking-tight text-text-primary">
        {value}
      </div>
      <div className="text-sm text-text-secondary mt-1">{label}</div>
      {sublabel && (
        <div className="text-xs text-text-muted mt-0.5">{sublabel}</div>
      )}
    </div>
  );
}