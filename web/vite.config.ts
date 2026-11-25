import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The browser calls /api/... ; Vite forwards it to the FastAPI service with the /api prefix removed.
// Same-origin requests mean no CORS setup is needed. Point OVASCOPE_API_URL elsewhere if the API
// runs on another host or port (e.g. the Docker gateway on http://localhost:8000).
const target = process.env.OVASCOPE_API_URL ?? "http://localhost:8000";
const proxy = {
  "/api": { target, changeOrigin: true, rewrite: (path: string) => path.replace(/^\/api/, "") },
  "/openapi.json": { target, changeOrigin: true },   // loaded by the API docs page (/api/docs)
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: { manualChunks: { charts: ["recharts"], vendor: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"] } },
    },
  },
});
