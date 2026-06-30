---
title: Quickstart
---

# Quickstart

The easiest way to try Mesh is to create your own private mesh. Start one node on this machine, finish setup, send a chat message, then try an agent. Later, you can add more machines or invite other people to join the same private mesh.

## 1. Install the executable

Choose your platform for the fastest path:

**macOS**:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
```

Full guide: [Installing on macOS](/docs/pages/installing-macos/)

**Linux**:

```sh
curl -fsSL https://meshllm.cloud/install.sh | bash
```

Full guide: [Installing on Linux](/docs/pages/installing-linux/)

**Windows**:

```powershell
irm https://meshllm.cloud/install.ps1 | iex
```

Full guide: [Installing on Windows](/docs/pages/installing-windows/)

Open a new terminal after install if the installer added Mesh to your `PATH`.

## 2. Finish setup

Run the setup command after the executable is installed:

```sh
mesh-llm setup
```

On Windows PowerShell:

```powershell
mesh-llm.exe setup
```

On interactive macOS and Linux terminals, setup can offer to install and enable the background service. The GitHub star prompt only appears when interactive and eligible, and it defaults to Yes.

## 3. Start one private node

Use this model first on a 12GB+ machine:

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

On Windows PowerShell:

```powershell
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

Keep this terminal open. A ready node exposes:

| Surface | URL |
|---|---|
| Console | `http://localhost:3131` |
| OpenAI-compatible API | `http://localhost:9337/v1` |

If the model does not load, stop Mesh and use the [model picker](/docs/pages/choose-a-model/) to choose a smaller starting point.

```sh
mesh-llm stop
```

## 4. Chat in the console

Open:

```text
http://localhost:3131
```

Send a short prompt in the chat view:

```text
Say hello in one sentence.
```

This proves the node is running, the model loaded, and the local routing path works.

## 5. Check the API

List the models your local node can route to:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

You should see at least one model id. Use that id for direct API calls or agents.

## 6. Try an agent

After console chat works, run one agent launcher:

```sh
mesh-llm goose
```

Other launchers use the same local endpoint:

```sh
mesh-llm claude
```

```sh
mesh-llm opencode --host 127.0.0.1:9337
```

```sh
mesh-llm pi --host 127.0.0.1:9337
```

For tools without a Mesh launcher, configure an OpenAI-compatible provider with base URL `http://localhost:9337/v1` and API key `dummy`.

## 7. Add another machine

Install Mesh on another machine and run the same command with the same mesh name:

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

Mesh nodes using the same private mesh name find each other and advertise their models to the same local API.
