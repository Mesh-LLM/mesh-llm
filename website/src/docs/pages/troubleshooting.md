# Troubleshooting

Start with these checks before changing configuration.

## Report a problem

Run `mesh-llm doctor` before opening a bug report. The doctor output gives the Mesh LLM team the local status, runtime diagnostics, and logs needed to debug the issue.

Capture a doctor archive:

```sh
mesh-llm doctor
```

Open a [new GitHub issue](https://github.com/Mesh-LLM/mesh-llm/issues/new) and attach the archive created by `mesh-llm doctor`. Include the command you ran, your OS, GPU/backend flavor, model ref, whether you used a private mesh or `--auto`, and what you expected to happen.

## Is Mesh running?

```sh
curl -s http://localhost:3131/api/status | jq .
```

If this fails, start a node:

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

## Is a model available?

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If no models are listed, the model did not load or no serving peer is available. Try a smaller model:

```sh
mesh-llm stop
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

## Is the console reachable?

Open:

```text
http://localhost:3131
```

If the console is not reachable, another process may be using the port or the node may not be running.

## Stop stale local processes

```sh
mesh-llm stop
```

If you are developing from source, use the repository cleanup commands in the testing docs.

## Agent fails but console works

List models and pass one explicitly:

```sh
mesh-llm goose
```

## Public mesh connection issues

For first-run testing, prefer a private mesh:

```sh
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

Then move back to `mesh-llm serve --auto` once the local install and model path work.
