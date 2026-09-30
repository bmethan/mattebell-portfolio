import type { Metadata } from "next";
import "./globals.css";
import { SITE, TITLE, OG_SHARED } from "@/lib/site";

// The page sets its own description and canonical (app/page.tsx), so a future route never inherits them.
export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: TITLE,
  openGraph: OG_SHARED,
  twitter: { card: "summary_large_image" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
