/// <reference types="node" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

const port = Number(process.env['PORT']) || 7431;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port,
    strictPort: true,
    host: '127.0.0.1',
    hmr: { protocol: 'wss', host: 'water4.localhost', clientPort: 443 },
  },
});
