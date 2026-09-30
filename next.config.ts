import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      // One address for search engines: www redirects to the bare domain.
      {
        source: "/:path*",
        has: [{ type: "host", value: "www.mattebell.xyz" }],
        destination: "https://mattebell.xyz/:path*",
        permanent: true,
      },
      // The old resume file name, in case it was shared.
      { source: "/mbresume_2026.pdf", destination: "/matthew_bell_resume_2026.pdf", permanent: true },
    ];
  },
  async headers() {
    // The resume stays downloadable but out of search results, since it carries a phone number.
    return [{ source: "/matthew_bell_resume_2026.pdf", headers: [{ key: "X-Robots-Tag", value: "noindex" }] }];
  },
};

export default nextConfig;
