import type { NextConfig } from "next";

/**
 * Frontend proxy configuration.
 *
 * The browser talks to same-origin `/api/*`; Next.js rewrites that to the
 * FastAPI backend at runtime so no API host/key is ever sent to the client.
 * Point `BACKEND_URL` at the API process (defaults to the local dev server).
 */
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8001";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
