# mesh-llm-moa-plugin

Built-in virtual-model plugin that exposes MeshLLM's Mixture-of-Agents engine
through the generic plugin protocol. The plugin owns MoA orchestration while
the host retains model discovery, placement, nested inference, accounting, and
OpenAI-compatible response framing.

It is off by default. Enable it with a `[[plugin]]` entry named `mesh-moa` in
`~/.mesh-llm/config.toml`; it then answers non-streaming text chat sent to
`model: "mesh"`. Everything else uses core single-model routing.

The built-in MoA policy strips thinking text from direct and aggregated public
answers. The current path does not preserve a separate copy of that text for
later use; a future arbitration policy that needs it requires an explicit
change to the worker response handling.
