import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/health": "http://localhost:8001",
      "/models": "http://localhost:8001",
      "/sessions": "http://localhost:8001",
      "/query": "http://localhost:8001",
    },
  },
});
