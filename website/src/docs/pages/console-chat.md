# Console Chat

Use the console before configuring agents or SDKs. It gives the fastest feedback that Mesh is actually serving a model.

## Open the console

Start a node, then open:

```text
http://localhost:3131
```

The console is local to the machine running Mesh. It shows node status, visible peers, loaded models, and chat.

## Send a first prompt

Use a short prompt first:

```text
Say hello in one sentence.
```

If this works, the model is loaded and the local routing path is healthy.

## What to check

The console should show:

| Signal | Meaning |
|---|---|
| A local node | Mesh is running on this machine. |
| At least one model | The API has something to route to. |
| Chat response | Inference is working end to end. |
| Peer count | Other machines in the same mesh are visible. |

## If chat does not respond

Check the API from a terminal:

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If no models are listed, the model did not load or no serving peer is available. Try a smaller model, then restart:

```sh
mesh-llm stop
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```
