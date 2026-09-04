# Mesh desktop app

Mesh is a native Tauri desktop application around the existing Mesh console. Launching it opens a
normal application window; the menu-bar/tray controls remain available for starting and stopping
the runtime, pairing another device, autostart, notifications, and diagnostics. It never sends the
main experience to an external browser.

The app prefers a `mesh-llm` executable beside the app bundle, then falls back to the Mesh runtime
embedded in the application binary. `MESH_LLM_BIN` can select an explicit installation during
development.

The console defaults to port `3131`. Pass `--console <port>` to the launcher or set
`MESH_LLM_CONSOLE_PORT` to control the daemon, status polling, deep links, and application routes as
a single unit.

From the repository root, build and run the desktop app:

```bash
just launcher-dev
```

On macOS the app opens as a normal Dock application and also installs a menu-bar item. Closing the
window keeps Mesh available from the menu bar; **Quit Mesh** exits the desktop app.
