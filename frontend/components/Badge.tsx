"use client";

interface BadgeProps {
  variant: "success" | "warning" | "error" | "info" | "muted";
  children: React.ReactNode;
  dot?: boolean;
}

const variantMap = {
  success: "bg-success/12 text-success border-success/20",
  warning: "bg-warning/12 text-warning border-warning/20",
  error: "bg-error/12 text-error border-error/20",
  info: "bg-accent/12 text-accent border-accent/20",
  muted: "bg-elevated text-text-muted border-border-subtle",
};

const dotColorMap = {
  success: "bg-success",
  warning: "bg-warning",
  error: "bg-error",
  info: "bg-accent",
  muted: "bg-text-muted",
};

export default function Badge({ variant, children, dot = false }: BadgeProps) {
  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full
        text-[11px] font-medium border ${variantMap[variant]}
      `}
    >
      {dot && (
        <span className={`w-1.5 h-1.5 rounded-full ${dotColorMap[variant]}`} />
      )}
      {children}
    </span>
  );
}
