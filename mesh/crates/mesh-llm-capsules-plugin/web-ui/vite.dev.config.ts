// Dev server for the Evidence page, outside the console.
//
//   VITE_EVIDENCE_FIXTURES=/abs/path/to/fixture-set pnpm dev
//     answers the page's routes from a captured run (dev/plugin-route-fixtures.ts);
//   MESH_UI_API_ORIGIN=http://127.0.0.1:3131 pnpm dev
//     proxies /api to a running node instead.
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import { pluginRouteFixtures } from './dev/plugin-route-fixtures.ts'
import { scopeToMountRoot } from './scope-css.ts'

const dirname = path.dirname(fileURLToPath(import.meta.url))
const fixtures = process.env.VITE_EVIDENCE_FIXTURES
const apiTarget = process.env.MESH_UI_API_ORIGIN ?? 'http://127.0.0.1:3131'

export default defineConfig({
  root: path.resolve(dirname, 'dev'),
  plugins: [pluginRouteFixtures(fixtures), react(), tailwindcss(), scopeToMountRoot()],
  resolve: { alias: { '@': path.resolve(dirname, './src') } },
  server: {
    host: '127.0.0.1',
    port: 5180,
    strictPort: false,
    proxy: fixtures ? undefined : { '/api': { target: apiTarget, changeOrigin: true } }
  }
})
