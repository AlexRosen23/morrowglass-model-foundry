import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> }; // Node global (no @types/node needed)
// API_PORT lets a second copy (e.g. a test port) run beside the live one.
const target = `http://127.0.0.1:${process.env.API_PORT ?? "5100"}`;
const api = { "/api": { target, changeOrigin: false }, "/v1": { target, changeOrigin: false } };

export default defineConfig({
  plugins: [react()],
  server: { port: Number(process.env.WEB_PORT ?? 5101), strictPort: true, host: "127.0.0.1", proxy: api },
  preview: { port: Number(process.env.WEB_PORT ?? 5101), strictPort: true, host: "127.0.0.1", proxy: api },
});
