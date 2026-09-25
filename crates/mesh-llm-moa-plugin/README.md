# mesh-llm-moa-plugin

Built-in virtual-model plugin that exposes MeshLLM's Mixture-of-Agents engine
through the generic plugin protocol. The plugin owns MoA orchestration while
the host retains model discovery, placement, nested inference, accounting, and
OpenAI-compatible response framing.
