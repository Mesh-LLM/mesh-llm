# capsules Evidence page (mesh-llm plugin web UI bundle)

The plugin's one console page, **Evidence**, at
`/plugins/capsules/evidence`. It uses mesh-llm's plugin web UI
projection (manifest `web_ui` block, `src/web_ui_manifest.rs`): install the
plugin and the page appears. No console fork is needed.

```bash
pnpm install
pnpm typecheck && pnpm test     # unit suite
pnpm build                      # -> ../bundle/register-mesh-plugin-ui.js (one ES module)
```

## Fixture mode (no node)

```bash
VITE_EVIDENCE_FIXTURES=/path/to/<captured-fixture-set> pnpm dev
# open http://127.0.0.1:5180/
```

The dev harness (`dev/`) mounts the page the way the console does:
`registerMeshPluginUi(host)` → `pages.evidence({ element, host })`. It uses a
standalone host, and a stand-in for the console's global stylesheet that
compiles **no** utilities, so a class the page's own sheet misses shows up
broken here instead of passing by accident. The fixture sets are the fork
console's captures (`scripts/capture-evidence-fixtures.mjs`). Each plugin
route is answered from the host route that captured the same data
(`dev/plugin-route-fixtures.ts`), and an uncaptured route is a 404, never
invented data. Without `VITE_EVIDENCE_FIXTURES`, `/api` is proxied to
`MESH_UI_API_ORIGIN` (default `http://127.0.0.1:3131`).

- `DATA-ROUTES.md`: the routes the page reads, and what the plugin must serve
- `VENDORED.md`: what came from the mesh-llm console, and what changed
