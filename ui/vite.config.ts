import path from "node:path"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  // The Flask server only serves /assets/app.js and /assets/style.css, so emit exactly those.
  base: "/assets/",
  build: {
    outDir: "../src/workbench_web",
    assetsDir: "",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: "app.js",
        chunkFileNames: "app.js",
        assetFileNames: "style.css",
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (proxyReq, req) => {
            if (req.headers.origin === "http://127.0.0.1:5173") {
              proxyReq.setHeader("Origin", "http://127.0.0.1:8765")
            }
          })
        },
      },
    },
  },
})
