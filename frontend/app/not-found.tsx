import Link from "next/link";

export default function NotFound() {
  return (
    <div className="p-6 max-w-xl mx-auto mt-24 text-center">
      <p className="text-sm text-text-muted uppercase tracking-[0.16em] mb-3">404</p>
      <h1 className="text-2xl font-bold text-text-primary font-[family-name:var(--font-display)] mb-3">
        Page not found
      </h1>
      <p className="text-sm text-text-secondary mb-6">
        The page you are looking for does not exist in FlyMind.
      </p>
      <Link href="/" className="btn btn-accent no-underline">
        Back to overview
      </Link>
    </div>
  );
}
