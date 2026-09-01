---
name: deploy-macos
description: Use this skill when deploying, installing, launching, or serving mesh-llm on a macOS machine (local or remote over SSH), including installing a release, shipping a dev build bundle, codesign/quarantine fixes, choosing a model, and verifying it serves.
metadata:
  short-description: Deploy mesh-llm on a macOS node
---

# deploy-macos

Use this when standing up mesh-llm on a macOS machine — either installing a
release or shipping a locally built dev binary to a remote Mac for testing.

This is the macOS counterpart to `deploy-linux-gpu`. The current binary embeds
the staged llama.cpp runtime: the bundle is a **single `mesh-llm` binary**.
There is no `rpc-server`, no `llama-server`, and no `.dylib` set anymore — if
you see instructions mentioning those, they are outdated.

Related skills/docs:

- `deploy-linux-gpu` — remote Linux/CUDA nodes
- `deploy-windows` — Windows nodes
- `mesh-join` — creating/joining private and public meshes (tokens, NAT, multi-node)
- `connect-agents` — pointing Goose/Claude Code/OpenCode/Pi at a running mesh
- `docs/USAGE.md` — install details, service mode, model storage
- `docs/CLI.md` — full command and model-ref reference

## The one rule that matters most

**mesh-llm resolves and downloads the model itself.** Pass `--model <ref>` and
it fetches the GGUF into the standard Hugging Face cache on first use. Do NOT
pre-download with `hf`/`huggingface-cli`, do NOT scp GGUFs around. (Only
`--gguf` takes a local file path you manage yourself.)

## Install path A: official release (most cases)

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash
```

The binary lands at `~/.local/bin/mesh-llm` (may not be on a non-interactive
SSH `PATH` — use the full path or `bash -lc`). Metal is the macOS backend; the
installer picks it automatically.

To install as a per-user background service (launchd agent) in the same step:

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | bash -s -- --service
```

Service files: `~/Library/LaunchAgents/com.mesh-llm.mesh-llm.plist`, shared env
in `~/.config/mesh-llm/service.env`, startup models in `~/.mesh-llm/config.toml`.

## Install path B: dev build to a remote Mac

Build and bundle locally (from the repo):

```bash
just release-build   # serious testing must use the release binary
just bundle          # /tmp/mesh-llm-bundle.tar.gz (single mesh-llm binary)
```

Ship and unpack:

```bash
scp -P <SSH_PORT> /tmp/mesh-llm-bundle.tar.gz user@host:
ssh -p <SSH_PORT> user@host 'mkdir -p ~/bin && tar xzf mesh-llm-bundle.tar.gz -C ~/bin --strip-components=1'
```

### Clear quarantine and sign the final remote binary — ALWAYS after scp

Files transferred via scp can carry provenance/quarantine xattrs that make
macOS SIGKILL the binary on launch (exit 137). Clear those attributes before
signing, then sign the final remote binary once with the stable Apple-issued
development identity used for that build:

```bash
xattr -cr ~/bin/mesh-llm
codesign --force --sign "<APPLE_DEVELOPMENT_IDENTITY>" ~/bin/mesh-llm
codesign --verify --verbose=2 ~/bin/mesh-llm
codesign -dv --verbose=4 ~/bin/mesh-llm 2>&1 | tee /tmp/mesh-llm-codesign.txt
```

`xattr ~/bin/mesh-llm` should print nothing. Do not ad-hoc sign the
binary: that identity is not stable across builds and can invalidate the Local
Network privacy decision. Signing changes the file hash, so do not compare
local and remote hashes afterward.

Verify the version on the remote matches what you built:

```bash
~/bin/mesh-llm --version
```

### Authorize macOS Local Network access — BEFORE remote diagnosis

macOS Local Network privacy is keyed to the **responsible code** and its code
signature. Replacing or ad-hoc re-signing a development binary can therefore
change the identity whose decision macOS remembers. A GUI app or launch agent
may show a blocking Local Network alert on the logged-in desktop even though
the same network is reachable from an interactive shell.

For the first multi-node run of an exact app/binary identity:

1. Sign it once with the stable Apple-issued development identity used for that
   build. Do not repeatedly ad-hoc re-sign it between attempts. Verify with
   `codesign --verify --verbose=2 <path>` and record
   `codesign -dv --verbose=4 <path> 2>&1` in the lab evidence.
2. Launch it once from the same responsible app/service context intended for
   the test while a user is logged in at the Mac. Trigger a mesh join or mDNS
   operation, then accept the Local Network alert on that Mac. Repeat on every
   Mac participating in the test.
3. Check **System Settings > Privacy & Security > Local Network**. If the
   responsible app is listed, make sure access is enabled. macOS offers no
   supported command to query or reset an individual program from the
   undetermined state, so scripts must not claim this check passed merely
   because a raw UDP probe worked.
4. Authorize the exact launch context that will run the test. For a
   CLI-launched repro over SSH, use the held foreground TTY described by
   `remote-observable-process`. If a per-user launchd agent or GUI app will run
   the workload, launch that exact agent/app while a user is logged in and
   accept its prompt before diagnosis; an SSH TTY does not authorize a
   different responsible app.
5. Treat the network preflight as passed only after both nodes report an iroh
   **direct** path, `observed_via_relay=false`, and an
   `observed_direct_remote_addr` whose IP matches the intended LAN address of
   the peer. `direct_addr_available=true` alone only proves that some direct
   address was observed. An invitation containing a LAN candidate, a raw UDP
   probe, or a working relay path does not prove the deployed process has Local
   Network access.

If no alert appears, keep the process alive after its first failed LAN
operation: macOS can fail to display the alert for very short-lived processes.
Do not debug iroh candidate selection until this gate is complete. See Apple
TN3179, [Understanding local network privacy](https://developer.apple.com/documentation/technotes/tn3179-understanding-local-network-privacy).

## Launch

Serve a model and join the public mesh:

```bash
mesh-llm serve --model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL --auto
```

- `--auto` discovers and joins the community mesh; local serving and mesh
  joining happen together.
- Without `--auto` (and without `--join`/`--discover`) you create a private
  mesh and an invite token is emitted — see the `mesh-join` skill.
- `--model` accepts catalog names, `repo:QUANT`, `repo/file.gguf`, or a full HF
  URL. `--gguf /path/file.gguf` serves a local file directly.
- API on `:9337`, management console on `:3131` (override with `--port` /
  `--console`).

Notes / gotchas:

- **Do NOT use `--headless` to "go quiet"** — it only disables the embedded web
  UI and does nothing for backgrounding. For machine-readable output use
  `--log-format json`.
- **Model load takes time.** Poll `/v1/models` until your model appears before
  concluding anything is broken.
- For background test runs from an agent:
  `bash -c 'nohup mesh-llm serve --model <ref> --auto > /tmp/mesh.log 2>&1 & disown'`.
  For persistence across reboots, prefer the `--service` install.

## Verify it's actually serving

```bash
# Ports bound
lsof -nP -iTCP:9337 -iTCP:3131 -sTCP:LISTEN

# Models (union of local + mesh peers)
curl -s http://localhost:9337/v1/models | python3 -m json.tool

# Status / peers
curl -s http://localhost:3131/api/status | python3 -m json.tool

# Inference — the returned "model" field tells you which node/model answered
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

To force your local model specifically, pass its exact id from `/v1/models`
instead of `auto`.

## Logs and state

- `~/.mesh-llm/runtime/<pid>/logs/skippy-native.log` — embedded llama.cpp/skippy
  native logs. Check here first if a model fails to load.
- `~/.mesh-llm/key` — persistent node identity.
- `~/.mesh-llm/config.toml` — startup models and defaults for bare `mesh-llm serve`.
- HF cache (`~/.cache/huggingface/...`) — downloaded GGUFs; you generally never
  need to touch this.

## Stop / clean up

```bash
mesh-llm stop        # scoped stop of tracked instances (preferred)
# emergency only: use the PID for the specific instance being validated
kill -9 <mesh-llm-pid>
```

A clean stop removes the instance runtime dir under `~/.mesh-llm/runtime/`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Exit 137 immediately after scp | macOS quarantine/provenance xattr | Clear xattrs, sign once with the stable Apple-issued development identity, and verify as above |
| `mesh-llm: command not found` over SSH | `~/.local/bin` not on non-interactive PATH | Full path or `bash -lc` |
| Empty `/v1/models` | Model still downloading/loading | Wait; watch skippy-native.log |
| "No inference server available" | Election in progress or load failed | Check stderr + skippy-native.log |
| Stale runtime dir after crash | Unclean exit | `rm -rf ~/.mesh-llm/runtime/<stale_pid>/` (auto-GC'd after 1h too) |
