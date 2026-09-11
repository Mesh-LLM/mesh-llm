---
title: Share an existing model server
---

# Share an existing model server

Already running Ollama or LM Studio on your Mac Studio, desktop, or another
machine? `share` makes those existing models available to the users and
machines on your mesh through an OpenAI-compatible API.

**`share` does not start, stop, or restart your existing server.** It does not
manage that server's model lifecycle. You continue to operate it with your
existing tools; Mesh advertises its available models and forwards inference
requests to it.

Mesh forwards requests to the existing server; it does not split that server's
model across GPUs. No plugin installation, native inference runtime, model
downloads, or config edits are needed on the sharing node.

## 1. Identify the API you already run

Prerequisites: [Mesh installed](/docs/pages/installing-mesh/), an existing API
server already running, and at least one model reported by its `/v1/models`
endpoint. Server setup and model management remain outside `share`.

- **Ollama:** the default API address is `http://localhost:11434`.
- **LM Studio:** use the address and port of its already-running local API
  server; the examples below use `http://localhost:1234`.
- **Other servers:** use an HTTP OpenAI-compatible API exposing `/v1/models`
  and `/v1/chat/completions`, without authentication.

Run Mesh on the same machine as the upstream, so you can keep the upstream
bound to localhost. You do not need to expose Ollama or LM Studio's port to
other machines.

## 2. Share it

Choose the command for your server:

```bash
# Ollama
mesh-llm share http://localhost:11434

# Or LM Studio (use your configured port)
mesh-llm share http://localhost:1234
```

Supply the URL explicitly: bare `mesh-llm share` does not scan for servers.
Keep the terminal open. Mesh starts a **private mesh** by default and prints an
invite token. Treat that token as access to your mesh; share it only with the
people or machines you want to admit. Do not publish it in a public chat.

In another terminal on this machine, check the advertised model IDs:

```bash
curl http://localhost:9337/v1/models
```

The models retain the IDs reported by your server. If the upstream is
unreachable or reports no models when you invoke `share`, the command reports
an error; it does not launch the server or load a model for you. Resolve that
in your existing server's tools, then retry `share`.

While sharing, Mesh refreshes the advertised list as upstream models change.
If the server goes away, Mesh withdraws its models after the health-check grace
period; if it becomes available again, Mesh re-advertises them. **Mesh does not
restart or repair the upstream.**

## 3. Connect a second machine

On your laptop or another machine, replace `<token>` with the invite token from
the sharing node:

```bash
mesh-llm client --join <token>
```

Keep this terminal open too. This machine needs no local model or supported
GPU. In another terminal **on the client machine**, list the available models:

```bash
curl http://localhost:9337/v1/models
```

Wait for the upstream's model IDs to appear. Here, `localhost` means the client
machine: its Mesh API routes requests over the mesh to your server machine.

## 4. Chat from the second machine

Open `http://localhost:3131` on the client machine and select an advertised model
in the console chat. Or send an API request, replacing `<model-id>` with an
exact ID returned by `/v1/models`:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'
```

The curl example above uses Bash-style quoting and line continuation. On
Windows, the console chat is the simplest way to try the same request.

Your existing server runs the inference; Mesh carries the request and response.
Tools on the client machine can use `http://localhost:9337/v1` as their OpenAI
base URL. See [Coding agents](/docs/pages/agents/) for agent setup.

## Join an existing mesh or publish for discovery

To contribute your server to a mesh you already have, supply its invite token
when starting the sharing node:

```bash
mesh-llm share http://localhost:11434 --join <token>
```

To deliberately make this mesh publicly discoverable instead:

```bash
mesh-llm share http://localhost:11434 --publish
```

Only publish a server whose models and capacity you intend to share publicly.
See [Private meshes](/docs/pages/private-meshes/) and
[Publish mesh](/docs/pages/publish-mesh/) for the wider workflows.

## Stop sharing and current limits

- **Ctrl-C stops sharing, not the upstream.** Ollama or LM Studio keeps running
  and serving its own clients. Stop the client command separately when finished.
- **One upstream per run.** To change the URL, stop sharing and restart with the
  new URL. It is not saved to your config.
- **HTTP only, without upstream authentication.** HTTPS, API keys and
  authenticated cloud providers are not supported by `share` yet. Do not put
  credentials in the URL or disable authentication on a publicly exposed server
  to make it work.
- **Use free local ports.** Mesh defaults to API port `9337` and console port
  `3131`. If another Mesh instance is already running, choose different ports
  with `--port` and `--console`, and adjust the local URLs above accordingly.
- **Persistent provider configuration is separate.** Use the
  [openai-endpoint plugin](/docs/pages/plugins/#external-endpoint-only-workflow-plugin)
  when you want the endpoint recorded in `config.toml` and restored on every
  start.

For command options, run `mesh-llm share --help`.
