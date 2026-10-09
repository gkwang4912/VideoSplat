import { defineConfig } from "vite";

export default defineConfig({
  root: "frontend",
  base: "/static/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    target: "es2022",
  },
});
