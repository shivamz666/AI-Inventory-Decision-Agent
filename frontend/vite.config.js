import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/products': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/decision': 'http://localhost:8000',
      '/analyze': 'http://localhost:8000',
      '/alternative-products': 'http://localhost:8000',
      '/approval': 'http://localhost:8000',
      '/approvals': 'http://localhost:8000',
      '/demo': 'http://localhost:8000',
    }
  }
})
