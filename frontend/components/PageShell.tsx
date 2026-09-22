"use client";

import { motion } from "framer-motion";

export default function PageShell({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`p-4 sm:p-6 lg:p-8 max-w-[1400px] mx-auto ${className}`}
    >
      {children}
    </motion.div>
  );
}