"use client";

import Link from "next/link";
import { useEffect } from "react";
import ErrorState from "@/components/ErrorState";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surface the digest for operators without leaking stack frames to the UI.
    console.error("FlyMind page error", error?.digest ?? error?.message);
  }, [error]);

  return (
    <div className="p-6 max-w-xl mx-auto mt-16">
      <ErrorState
        message="Something went wrong while rendering this page."
        onRetry={reset}
      />
      <p className="mt-4 text-center">
        <Link href="/" className="text-sm text-accent hover:underline underline-offset-4">
          Back to overview
        </Link>
      </p>
    </div>
  );
}
