"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  GitBranch,
  Brain,
  Zap,
  ListOrdered,
  FlaskConical,
  BookOpen,
  Cpu,
  Menu,
  X,
} from "lucide-react";
import { getHealth } from "@/lib/api";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/pipeline", label: "ML Pipeline", icon: GitBranch },
  { href: "/neurons", label: "Neuron Explorer", icon: Brain },
  { href: "/predictor", label: "Connection Predictor", icon: Zap },
  { href: "/candidates", label: "Candidate Ranking", icon: ListOrdered },
  { href: "/research", label: "Research Results", icon: FlaskConical },
  { href: "/about", label: "Methodology", icon: BookOpen },
] as const;

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [healthState, setHealthState] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    let cancelled = false;
    const check = () =>
      getHealth()
        .then(() => {
          if (!cancelled) setHealthState("online");
        })
        .catch(() => {
          if (!cancelled) setHealthState("offline");
        });
    check();
    const id = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const closeMobile = () => setMobileOpen(false);

  const statusDot =
    healthState === "online"
      ? "bg-success animate-pulse-slow"
      : healthState === "offline"
        ? "bg-error"
        : "bg-warning animate-pulse-slow";
  const statusLabel =
    healthState === "online"
      ? "System Online"
      : healthState === "offline"
        ? "System Offline"
        : "Checking…";

  const content = (
    <>
      <div className="px-5 py-5 border-b border-border-subtle">
        <Link href="/" onClick={closeMobile} className="flex items-center gap-2.5 no-underline group">
          <div className="w-9 h-9 rounded-md bg-gradient-to-br from-accent/20 to-violet/20 border border-accent/20 flex items-center justify-center group-hover:border-accent/40 transition-colors">
            <Cpu className="w-5 h-5 text-accent" />
          </div>
          <div>
            <span className="text-[15px] font-bold text-text-primary tracking-tight font-[family-name:var(--font-display)]">
              FlyMind
            </span>
            <span className="block text-[9px] text-text-muted font-medium tracking-[0.14em] uppercase">
              Connectome Analysis
            </span>
          </div>
        </Link>
      </div>

      <nav className="flex-1 py-3 px-3 space-y-0.5 overflow-y-auto" aria-label="Main navigation">
        {NAV_ITEMS.map((item) => {
          const active =
            item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={closeMobile}
              aria-current={active ? "page" : undefined}
              className={`
                relative flex items-center gap-2.5 px-3 py-2.5 rounded-[var(--radius-sm)] text-sm font-medium
                transition-colors duration-150 no-underline
                ${
                  active
                    ? "text-accent"
                    : "text-text-secondary hover:text-text-primary hover:bg-elevated"
                }
              `}
            >
              {active && (
                <motion.span
                  layoutId="sidebar-active"
                  className="absolute inset-0 rounded-[var(--radius-sm)] bg-accent/10 border border-accent/20"
                  transition={{ type: "spring", stiffness: 500, damping: 40 }}
                />
              )}
              <item.icon className="w-4 h-4 shrink-0 relative z-10" />
              <span className="relative z-10">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-border-subtle">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusDot}`} />
          <span className="text-xs text-text-muted">{statusLabel}</span>
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile top bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-40 bg-base/90 backdrop-blur-md border-b border-border-subtle px-4 py-3 flex items-center justify-between">
        <Link href="/" onClick={closeMobile} className="flex items-center gap-2 no-underline">
          <div className="w-7 h-7 rounded-md bg-accent/15 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-accent" />
          </div>
          <span className="text-sm font-bold text-text-primary font-[family-name:var(--font-display)]">
            FlyMind
          </span>
        </Link>
        <button
          onClick={() => setMobileOpen((o) => !o)}
          aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={mobileOpen}
          className="p-2 rounded-[var(--radius-sm)] text-text-secondary hover:text-text-primary hover:bg-elevated transition-colors cursor-pointer"
        >
          {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-base/60 backdrop-blur-sm"
          onClick={closeMobile}
          aria-hidden
        />
      )}

      {/* Desktop sidebar / mobile drawer */}
      <aside
        className={`
          fixed top-0 bottom-0 left-0 w-64 bg-surface border-r border-border-subtle flex flex-col z-50
          transition-transform duration-300 ease-in-out
          ${mobileOpen ? "translate-x-0" : "-translate-x-full"}
          md:translate-x-0 md:top-0
        `}
      >
        {content}
      </aside>
    </>
  );
}