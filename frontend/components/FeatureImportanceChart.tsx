"use client";

import { motion } from "framer-motion";
import type { FeatureImportanceRow } from "@/lib/metrics";

export default function FeatureImportanceChart({
  items,
}: {
  items: FeatureImportanceRow[];
}) {
  if (!items.length) return null;

  const max = Math.max(...items.map((i) => i.value));

  return (
    <div className="space-y-2.5">
      {items.map((item, i) => {
        const pct = max > 0 ? (item.value / max) * 100 : 0;
        return (
          <div key={item.name} className="flex items-center gap-3">
            <div className="w-28 shrink-0 text-right">
              <span className="text-xs font-[family-name:var(--font-mono)] text-text-secondary truncate">
                {item.name}
              </span>
            </div>
            <div className="flex-1 h-4 bg-elevated rounded-[4px] overflow-hidden min-w-0">
              <motion.div
                initial={{ width: 0 }}
                whileInView={{ width: `${pct}%` }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, ease: "easeOut", delay: i * 0.04 }}
                className="h-full rounded-[4px] bg-gradient-to-r from-accent to-violet"
              />
            </div>
            <div className="w-14 shrink-0 text-xs font-[family-name:var(--font-mono)] text-text-muted">
              {item.value.toFixed(4)}
            </div>
          </div>
        );
      })}
    </div>
  );
}