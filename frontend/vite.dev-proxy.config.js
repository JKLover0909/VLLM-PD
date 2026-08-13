import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5555,
    proxy: {
      "/health": "http://localhost:8002",
      "/models": "http://localhost:8002",
      "/knowledge": "http://localhost:8002",
      "/auth": "http://localhost:8002",
      "/sessions": "http://localhost:8002",
      "/query": "http://localhost:8002",
      "/agent": "http://localhost:8002",
      "/mes": "http://localhost:8002",
      "/wms": "http://localhost:8002",
      "/research": "http://localhost:8002",
      "/quick-answers": "http://localhost:8002",
      "/reports": "http://localhost:8002",
      "/sources": "http://localhost:8002",
    },
  },
});
