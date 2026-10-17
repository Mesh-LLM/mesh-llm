// Builds the plugin web UI bundle: ONE browser-importable ES module,
// `../bundle/register-mesh-plugin-ui.js`, with React and every dependency
// inlined. The console imports it as-is (it does not transpile TypeScript or
// JSX, and resolves no bare npm imports), so nothing may stay external and
// there is one file: the manifest's bundle root `bundle/` holds exactly what
// this build writes.
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { PluginOption } from 'vite'
import { defineConfig } from 'vitest/config'
import { scopeToMountRoot } from './scope-css.ts'

const dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * The twin-bracket diff viewer (react-diff-viewer-continued) lists a lazy
 * loader for ~35 refractor syntax grammars. The page never sets
 * `highlightLanguage`, so none is ever requested, but with every dynamic import
 * folded into the one module they would all ship. Each `refractor/<language>`
 * resolves to a stub that fails to load. The library already treats a failed
 * grammar load as "no highlighting" (`ensureLanguage` catches the rejection
 * and renders plain text), so behaviour is unchanged. `refractor/core` (the
 * tokenizer the viewer imports eagerly) is left alone.
 */
function noSyntaxGrammars(): PluginOption {
  const STUB = '\0capsules:no-syntax-grammar'
  return {
    name: 'capsules:no-syntax-grammars',
    enforce: 'pre',
    resolveId(source) {
      return /^refractor\/(?!core$)[a-z0-9-]+$/.test(source) ? STUB : null
    },
    load(id) {
      return id === STUB ? "throw new Error('syntax grammars are not bundled with the Evidence page')" : null
    }
  }
}

export default defineConfig({
  // scopeToMountRoot runs after Tailwind's own transform: it needs the compiled CSS.
  plugins: [noSyntaxGrammars(), react(), tailwindcss(), scopeToMountRoot()],
  resolve: { alias: { '@': path.resolve(dirname, './src') } },
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    outDir: path.resolve(dirname, '../bundle'),
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: false,
    copyPublicDir: false,
    lib: {
      entry: path.resolve(dirname, 'src/register-mesh-plugin-ui.tsx'),
      formats: ['es'],
      fileName: () => 'register-mesh-plugin-ui.js'
    },
    // The diff viewer lazy-loads per-language highlighters; fold them into
    // the one module rather than shipping dozens of sibling chunks.
    rolldownOptions: { output: { inlineDynamicImports: true } }
  },
  test: {
    environment: 'jsdom',
    maxWorkers: 2,
    globals: true,
    setupFiles: ['./src/lib/test/setup.ts', './src/plugin-host/test-host-setup.ts'],
    css: true,
    include: ['src/**/*.test.{ts,tsx}', 'dev/**/*.test.ts']
  }
})
