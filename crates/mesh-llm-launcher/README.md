# Mesh Launcher

Mesh Launcher is the tray-only desktop shell for Mesh. It has no webview: native menu items,
notifications, and `mesh-llm://pair/...` links open the existing local web console.

The launcher prefers a `mesh-llm` executable beside the app bundle, then falls back to the Mesh
runtime embedded in the launcher binary. `MESH_LLM_BIN` can select an explicit installation.

The console defaults to port `3131`. Pass `--console <port>` to the launcher or set
`MESH_LLM_CONSOLE_PORT` to control the daemon, status polling, deep links, and console routes as a
single unit.
