# Troubleshooting

Start with these checks before changing configuration.

## Report a problem

Use `mesh-llm doctor` when you need local status, runtime diagnostics, or logs for a bug report. It is for troubleshooting, not the normal install flow.

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

## Runtime lifecycle

Inspect the daemon, desired intents, and activity admission separately:

```sh
curl -s http://localhost:3131/api/status | jq '.runtime'
curl -s http://localhost:3131/api/runtime/intents | jq .
curl -s http://localhost:3131/api/runtime/activity | jq .
```

- `ready_idle` is healthy: listeners are ready and no route exists yet.
- `ready_proxying` is healthy without a local model: a plugin or remote route
  is available.
- `degraded` indicates a terminal eager-load failure or a priority restoration
  problem. Inspect `intent_summary.recent_errors` and the intent list.
- A non-`auto` activity override can pause admission. Clear it with
  `DELETE /api/runtime/activity/override`.
- A `draining` instance rejects new work until in-flight work finishes. It is
  forced down when `runtime.drain_timeout_secs` expires, bounded by
  `runtime.drain_timeout_max_secs`.

If startup reports a mode conflict, check `[runtime].mode`. Persisted `client`
cannot be combined with `mesh-llm serve` or explicit local model/serving flags.
Use `serve` or `on_demand` when the node should load local models.

With `best_effort` (default), a bad eager model can leave the daemon running
and degraded. With `fail_fast`, the process exits. Choose the policy
deliberately before treating either result as unexpected.

## Is a model available?

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

An empty list is valid on a healthy idle daemon. If you expected a model,
inspect intents and daemon state. For a local load failure, try a smaller model:

```sh
mesh-llm stop
mesh-llm serve --discover my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

If you expected an external endpoint, check both layers:

```sh
mesh-llm plugins info openai-endpoint
curl -s http://127.0.0.1:8000/v1/models | jq .
```

A healthy plugin process does not guarantee its configured inference endpoint
is reachable or healthy.

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

For a full decision guide, see
[Runtime Lifecycle](/docs/pages/runtime-lifecycle/).
