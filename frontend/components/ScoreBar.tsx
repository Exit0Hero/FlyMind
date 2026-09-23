"use client";

interface ScoreBarProps {
  score: number;
  variant?: "accent" | "violet";
  label?: string;
  showValue?: boolean;
  height?: number;
}

export default function ScoreBar({
  score,
  variant = "violet",
  label,
  showValue = true,
  height = 6,
}: ScoreBarProps) {
  const clamped = Math.min(1, Math.max(0, score));
  const pct = Math.round(clamped * 100);

  const color =
    variant === "violet"
      ? "bg-violet"
      : "bg-accent";

  return (
    <div className="w-full">
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-xs text-text-muted">{label}</span>}
          {showValue && (
            <span className="text-xs font-medium font-[family-name:var(--font-mono)] text-text-secondary">
              {pct}%
            </span>
          )}
        </div>
      )}
      <div
        className="w-full bg-elevated rounded-full overflow-hidden"
        style={{ height }}
      >
        <div
          className={`${color} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${pct}%`, height: "100%" }}
        />
      </div>
    </div>
  );
}
