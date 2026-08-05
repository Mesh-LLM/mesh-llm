---
name: deploy-linux-gpu
description: Use this skill when deploying, installing, launching, or serving mesh-llm on a remote Linux GPU node (rented GPUs like Vast.ai or RunPod, or a self-managed CUDA server), including installing the CUDA build, choosing a model, keeping it alive under a supervisor, and verifying it serves.
metadata:
  short-description: Deploy mesh-llm on a remote Linux GPU node
---

# deploy-linux-gpu

Use this when standing up mesh-llm on a remote Linux GPU box (rented GPU like
Vast.ai / RunPod, or your own server) to serve a specific model and join the
mesh.

This is the Linux/CUDA counterpart to the `deploy-macos` skill. It does NOT use
the old `llama-server`/`rpc-server` lane — the current binary embeds the staged
runtime. There are no `.dylib`/`codesign`/quarantine steps on Linux.

note --auto flag tells it to join the public mesh. serve command with --model tells it to run a specific model.
Examples here are for solo serving — don't read this in isolation:

- `deploy-macos` / `deploy-windows` — other platforms
- `mesh-join` — creating/joining private and public meshes (tokens, NAT, multi-node)
- `connect-agents` — pointing Goose/Claude Code/OpenCode/Pi at a running mesh
- `docs/USAGE.md` — install details, service mode, model storage
- `docs/CLI.md` — full command and model-ref reference
- `docs/SKIPPY_SPLITS.md` — splitting big models across nodes

## The one rule that matters most

**mesh-llm resolves and downloads the model itself.** When you pass `--model`,
it fetches the GGUF into the standard Hugging Face cache on first use and serves
it. Do NOT pre-download with `hf` / `huggingface-cli`, do NOT scp a GGUF, do NOT
hunt for where the file lives. Just pass `--model <ref>` and let mesh-llm do it.

## Install

SSH in and run the official installer. It auto-detects the GPU and CUDA major
version and pulls the matching CUDA build (e.g. `...-cuda-13.tar.gz`). RTX 50xx
(Blackwell / sm_120) is detected and handled by the installer.

```bash
curl -fsSL https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh | sh
```

The binary lands at `~/.local/bin/mesh-llm` (not always on a non-interactive
SSH `PATH` — use the full path or a login shell). Verify:

```bash
~/.local/bin/mesh-llm --version
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
```

Confirm the driver/CUDA is new enough for your GPU (Blackwell/RTX 50xx needs
CUDA 13 + a recent driver, which the installer's `-cuda-13` asset targets).

## The launch command

all you need:

```bash
mesh-llm serve --model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL --auto
```

- mesh-llm downloads/resolves the model itself (HF cache, first use only).
- It serves the model **locally on the GPU**.
- `--auto` also discovers and joins the community mesh, so `/v1/models` shows
  the union of local + peer models and routing works across nodes.
- Both serving locally AND joining the mesh happen together — `--auto` does not
  suppress local serving when `--model` is set.

Notes / gotchas:

- **Do NOT use `--headless`** to "go quiet". It only disables the embedded web
  console; it does nothing useful for backgrounding and is a recurring mistake.
- **Model load takes time.** After it joins the mesh, the GPU load + server
  bring-up can take a few minutes. Do not conclude "it's not serving" from an
  early check — poll until the GPU shows VRAM used and ports are bound.
- `--model` accepts catalog names, `repo/file.gguf`, `repo:QUANT`, or a full
  HF URL. For models not in the bundled catalog (e.g. Qwen3.6), use the HF ref
  form `org/Repo-GGUF:QUANT`.

### Choosing a quant for a context target

Context length is auto-scaled to VRAM (KV cache defaults to Q8_0). With `--auto`
the planner targets up to 4 concurrent slots, which divides the KV budget. If
you need a guaranteed deep context per request, pin it:

```bash
mesh-llm serve --model <ref> --auto --ctx-size 65536
```

Pick the quant so `model_bytes + KV` fits VRAM at your target context. KV cost
scales with layers × kv_heads × head_dim. Heavy-KV models (e.g. 64 layers,
head_dim 256) cost ~130 KB/token at Q8_0 → ~8.5 GB at 64K per slot, so prefer a
smaller weight quant (e.g. UD-Q4_K_XL) on a 32 GB card to leave KV headroom.

## Keep it running (survives SSH disconnect)

A plain `cmd &` over SSH dies when the session closes, and `tmux` dies if the
tmux server is killed/reaped. On managed GPU images that ship **supervisor**
(common on Vast.ai), use supervisor — it restarts on crash and persists.

```bash
cat > /etc/supervisor/conf.d/mesh-llm.conf <<'EOF'
[program:mesh-llm]
command=/root/.local/bin/mesh-llm serve --model unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL --auto
autostart=true
autorestart=true
startsecs=10
stopwaitsecs=30
stdout_logfile=/var/log/mesh-llm.log
stderr_logfile=/var/log/mesh-llm.log
environment=HOME="/root"
EOF
supervisorctl reread && supervisorctl update && supervisorctl start mesh-llm
supervisorctl status mesh-llm
```

If there is no supervisor, the installer can set up a `systemd --user` service
for you (`curl -fsSL .../install.sh | sh -s -- --service` installs
`~/.config/systemd/user/mesh-llm.service`; startup models go in
`~/.mesh-llm/config.toml`, and `sudo loginctl enable-linger "$USER"` makes it
survive reboot before login). Otherwise use `tmux new -d`, or a foreground
process in a held SSH session for first-run debugging (allocate a TTY with
`ssh -tt host 'bash -lc "..."'`).

## Verify it's actually serving

Poll until VRAM is used and the OpenAI port is bound, then test inference.

```bash
# GPU should show real VRAM used (not ~1 MiB) once the model loads
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# OpenAI port 9337 and console 3131 should be bound
ss -lntp | grep -E '9337|3131'

# Models (union of local + mesh peers)
curl -s http://localhost:9337/v1/models | python3 -m json.tool

# Inference — confirm the returned "model" is YOUR model id
curl -s http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

The response `"model"` field tells you which node/model answered. To force your
local model specifically, pass its exact id from `/v1/models` instead of `auto`.

## Logs and state

- `/var/log/mesh-llm.log` — process output when run under the supervisor config
  above.
- `~/.mesh-llm/runtime/<pid>/logs/skippy-native.log` — embedded llama.cpp/skippy
  native logs (redirected away from the TUI). Check here if the model fails to
  load onto the GPU.
- `~/.mesh-llm/key` — persistent node identity.
- HF cache (`~/.cache/huggingface/...`) — where mesh-llm puts downloaded GGUFs.
  You generally never need to touch this.

## Stop / clean up

```bash
supervisorctl stop mesh-llm      # if under supervisor
# or, for a tracked foreground/background run:
mesh-llm stop
# emergency only:
pkill -9 -f mesh-llm
```

A clean stop removes the instance runtime dir under `~/.mesh-llm/runtime/`.

## Vast.ai specifics

- First SSH after boot can fail auth with a "try again after a few seconds"
  banner. Retry the connection (a short loop that re-attempts on exit code 255
  works).
- Vast images often run their own services (portal, jupyter, syncthing,
  supervisor) — a non-zero load average at idle is normal and not your process.
- Vast pre-allocates external ports and fronts web apps with Caddy; if you want
  the console/API reachable externally, map it through the box's documented
  external port + proxy rather than assuming 9337/3131 are public.
- Use the proxy or direct SSH connect string Vast gives you; `-L 8080:...`
  port-forwards are handy for poking at local services from your workstation.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| GPU shows ~1 MiB, no 9337/3131 bound | Model still loading after mesh join | Wait — load can take minutes; poll `nvidia-smi` + `ss` |
| `mesh-llm: command not found` over SSH | `~/.local/bin` not on non-interactive PATH | Use full path or `bash -lc` |
| Process dies on SSH disconnect | Backgrounded with `&` or bare tmux | Use supervisor / `systemd` / `tmux new -d` |
| Installer pulls CPU build | GPU/CUDA not detected | Check `nvidia-smi`; set `MESH_LLM_INSTALL_FLAVOR=cuda` |
| Wrong/old CUDA build for Blackwell | CUDA major mismatch | Ensure CUDA 13 + recent driver for RTX 50xx |
| Empty `/v1/models` | API up but model not loaded yet | Wait for load; re-check |
