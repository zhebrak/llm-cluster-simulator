/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    allowedHosts: true,
  },
  build: {
    target: ['es2022', 'chrome105', 'firefox121', 'safari15.4', 'edge105'],
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  // @ts-expect-error - vitest extends vite config
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
  },
})
