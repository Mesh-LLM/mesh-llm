# mesh-llm Fly.io Console

Fly app running mesh-llm in `--client` mode — no GPU, just QUIC tunnels to mesh nodes.

| App | URL | Fly config |
|---|---|---|
| **console** | [mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev) | `fly/console/fly.toml` |

Also available at [public.meshllm.cloud](https://public.meshllm.cloud).

## Architecture

```
                              ┌─────────────────────────┐
Browser/curl ──HTTPS──→ Fly   │  mesh-llm --client      │
                              │  discovers mesh via      │──QUIC──→ GPU nodes
                              │  Nostr, tunnels requests │
                              └─────────────────────────┘
```

Exposes `:3131` (dashboard, chat, topology) and proxies inference to mesh GPU nodes.

## Deploy

### Preferred: GitHub Action (manual)

Deploy via the **Deploy Fly Console** workflow
(`.github/workflows/fly-deploy-console.yml`). It builds the image on Fly's
remote builders and deploys `mesh-llm-console` — no local Fly login needed.

From the Actions tab, run the workflow, or from the CLI:

```bash
# Deploys the default branch
gh workflow run "Deploy Fly Console"

# Deploy a specific branch/tag/SHA
gh workflow run "Deploy Fly Console" -f ref=v0.72.0
```

The workflow authenticates with the `FLY_API_TOKEN` repo secret, which holds an
app-scoped Fly deploy token:

```bash
fly tokens create deploy -a mesh-llm-console
gh secret set FLY_API_TOKEN   # paste the token when prompted
```

### Fallback: local deploy

From the **repo root** (requires `fly auth login`):

```bash
fly deploy --config fly/console/fly.toml --dockerfile fly/Dockerfile
```

## Run locally

```bash
# Same as what the Fly app runs — no Docker needed
mesh-llm --client --auto
```

## Docker (local)

```bash
docker build -f fly/Dockerfile -t mesh-llm-console .
docker run -p 3131:3131 -p 9337:9337 mesh-llm-console
```
