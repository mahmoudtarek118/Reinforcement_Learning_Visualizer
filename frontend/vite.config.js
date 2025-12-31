import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Base path for GitHub Pages - update 'RLbouns' if your repo name is different
  base: process.env.NODE_ENV === 'production' ? '/RLbouns/' : '/',
})
