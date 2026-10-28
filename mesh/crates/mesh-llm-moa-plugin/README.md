# mesh-llm-moa-plugin

Built-in virtual-model plugin that exposes MeshLLM's Mixture-of-Agents engine
through the generic plugin protocol. The plugin owns MoA orchestration while
the host retains model discovery, placement, nested inference, accounting, and
OpenAI-compatible response framing.

The built-in MoA policy strips thinking text from direct and aggregated public
answers. The current path does not preserve a separate copy of that text for
later use; a future arbitration policy that needs it requires an explicit
change to the worker response handling.

## Opt-in turn capture (research)

Set `MESH_MOA_CAPTURE_PATH=/path/to/moa-turns.jsonl` on the node running the
committee to append one JSON line per answer turn: the request `messages`, each
worker `draft`, the synthesized `final` message, `reducer_used`, `turn_kind`,
and per-worker timing. These are the (small worker → synthesized answer)
distillation pairs for a node that trains on its own traffic
([#2575](https://github.com/Mesh-LLM/mesh-llm/issues/2575)). The file is raw
and local-only; nothing is gossiped. Redaction belongs to whatever consumes it.
Unset (default) writes nothing.
