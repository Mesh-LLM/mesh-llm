# PR 991 live plugin web UI evidence

These screenshots were captured in Chromium by
`crates/mesh-llm-ui/e2e/plugins/web-ui-exemplar.live.spec.ts` against the real
console server at `http://127.0.0.1:13131`.

The test built `target/debug/mesh-llm` with `just build`, packaged and installed
the maintained `web-ui-exemplar`, and started the client-only host with the
documented `just mesh-client` recipe. The production Vite bundle was embedded
in and served by `mesh-llm`; no Vite development server, API fixture, request
interception, or mocked backend was used.

The run forced the application preference and browser color scheme to dark and
verified that the browser received successful same-origin plugin metadata,
configuration, JavaScript asset, runtime status, and config-control responses.
See `live-validation.json` for the recorded requests, status codes, persisted
setting, disabled projection behavior, and browser diagnostics.

The screenshots show:

1. `01-plugin-page-ready.png` — the installed plugin's deep-linked page and
   browser-loaded JavaScript contribution.
2. `02-plugin-settings-persisted.png` — the running plugin in Configuration →
   Plugins after `retention_days = 46` was saved and survived a reload.
3. `03-plugin-ui-disabled-capability-alive.png` — the host's disabled-projection
   state after the asset became unavailable while the plugin's non-UI tool
   continued to respond.
