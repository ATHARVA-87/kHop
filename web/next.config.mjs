/** Static export: no Next.js server at runtime. FastAPI serves web/out on the same port as the
 * API, which removes the server-side Next.js attack surface (research 2026-10-29). */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  poweredByHeader: false,
  trailingSlash: true,
  // the recorded demo (make demo-site) is served from a sub-path such as /kHop/ on GitHub Pages,
  // and is written to its own folder so it never replaces the app's web/out
  ...(process.env.NEXT_PUBLIC_BASE_PATH ? { basePath: process.env.NEXT_PUBLIC_BASE_PATH } : {}),
  ...(process.env.NEXT_PUBLIC_REPLAY === "1" ? { distDir: "out-demo" } : {}),
};
export default nextConfig;
