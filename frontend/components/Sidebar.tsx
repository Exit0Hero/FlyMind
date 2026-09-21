"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  GitBranch,
  Brain,
  Zap,
  BarChart3,
  FlaskConical,
  Info,
  Cpu,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/pipeline", label: "Pipeline", icon: GitBranch },
  { href: "/neurons", label: "Neurons", icon: Brain },
  { href: "/predictor", label: "Predictor", icon: Zap },
  { href: "/candidates", label: "Candidates", icon: BarChart3 },
  { href: "/research", label: "Research", icon: FlaskConical },
  { href: "/about", label: "About", icon: Info },
] as const;

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-60 bg-surface border-r border-border-subtle flex flex-col z-40">
      <div className="px-5 py-6 border-b border-border-subtle">
        <Link href="/" className="flex items-center gap-2.5 no-underline">
          <div className="w-8 h-8 rounded-md bg-accent/15 flex items-center justify-center">
            <Cpu className="w-4.5 h-4.5 text-accent" />
          </div>
          <div>
            <span className="text-sm font-semibold text-text-primary tracking-tight font-[family-name:var(--font-display)]">
              FlyMind
            </span>
            <span className="block text-[10px] text-text-muted font-medium tracking-wide uppercase">
              Connectome ML
            </span>
          </div>
        </Link>
      </div>

      <nav className="flex-1 py-3 px-3 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium
                transition-colors duration-150 no-underline
                ${
                  active
                    ? "bg-accent/10 text-accent"
                    : "text-text-secondary hover:text-text-primary hover:bg-elevated"
                }
              `}
            >
              <item.icon className="w-4 h-4 shrink-0" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse-slow" />
          <span className="text-xs text-text-muted">System Online</span>
        </div>
      </div>
    </aside>
  );
}
