import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = { "/api": { target: "http://127.0.0.1:5100", changeOrigin: false } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5101, strictPort: true, host: "127.0.0.1", proxy: api },
  preview: { port: 5101, strictPort: true, host: "127.0.0.1", proxy: api },
});
