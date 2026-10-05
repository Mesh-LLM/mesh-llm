# Python SDK examples

The Python SDK source and full guide live at:

- [`sdk/python/README.md`](../../sdk/python/README.md)
- [Python SDK website guide](../../website/src/docs/pages/sdk-python.md)

The package exposes `Client` for joining an existing mesh and `Node` for
embedded-node lifecycles. Agent runtimes should use the protocol-preserving
`chat_completions()` and `responses()` methods so OpenAI-compatible tool,
multimodal, structured-output, finish, and usage fields survive unchanged.
`stream_chat_completions()` and `stream_responses()` preserve complete SSE
frames, including incremental tool-call arguments and future protocol fields.
