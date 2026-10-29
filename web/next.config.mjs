/** Static export: no Next.js server at runtime. FastAPI serves web/out on the same port as the
 * API, which removes the server-side Next.js attack surface (research 2026-10-29). */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  poweredByHeader: false,
  trailingSlash: true,
};
export default nextConfig;
