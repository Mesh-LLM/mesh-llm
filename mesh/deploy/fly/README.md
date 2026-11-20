# mesh-llm Fly.io Console

Fly app running mesh-llm in `--client` mode — no GPU, just QUIC tunnels to mesh nodes.

| App | URL | Fly config |
|---|---|---|
| **console** | [mesh-llm-console.fly.dev](https://mesh-llm-console.fly.dev) | `mesh/deploy/fly/console/fly.toml` |

Also available at [public.meshllm.cloud](https://public.meshllm.cloud).

## Architecture

```
                              ┌─────────────────────────┐
Browser ──────HTTPS──→ Fly    │  mesh-llm --client      │
                              │  discovers mesh via      │──QUIC──→ GPU nodes
                              │  Nostr, tunnels requests │
                              └─────────────────────────┘
```

Exposes `:3131` for the dashboard, browser Chat, and topology. Browser Chat uses
`/api/responses` (or `/api/chat`) to reach mesh GPU nodes. The Console does not
expose the native inference routes (`/v1/*`, `/models*`, `/mesh/v1/*`).
OpenAI-compatible clients should run their own `mesh-llm client --auto` and use
`http://127.0.0.1:9337/v1`, or use a separately exposed native inference listener.
The single-port Fly Console is not a public inference API.

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
fly deploy --config mesh/deploy/fly/console/fly.toml --dockerfile mesh/deploy/fly/Dockerfile
```

## Run locally

```bash
# Same as what the Fly app runs — no Docker needed
mesh-llm --client --auto
```

## Docker (local)

```bash
docker build -f mesh/deploy/fly/Dockerfile -t mesh-llm-console .
docker run -p 3131:3131 -p 9337:9337 mesh-llm-console
```
