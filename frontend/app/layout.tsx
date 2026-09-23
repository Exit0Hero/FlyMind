import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "FlyMind — Connectome Analysis Platform",
  description: "ML-powered neural connectivity analysis for Drosophila connectomics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Sidebar />
        <main className="md:ml-64 pt-14 md:pt-0 min-h-screen">{children}</main>
      </body>
    </html>
  );
}
