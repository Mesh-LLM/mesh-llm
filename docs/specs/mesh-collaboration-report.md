# Mesh collaboration report contract

`mesh_collaboration` is an opt-in extension to non-streaming
`/v1/chat/completions`. Its initial mode guarantees that a final prose answer is
returned as one caller-declared report tool call while leaving intermediate tool
calls unchanged.

```json
{
  "mesh_collaboration": {
    "mode": "report_required",
    "version": 1,
    "tool": "buzz_send",
    "body_argument": "content",
    "locked_arguments": {
      "channel": "channel-id",
      "reply_to": "event-id"
    }
  }
}
```

The named tool must be declared exactly once in `tools`. `body_argument` must be
a top-level string property and cannot also be locked. Locked values must satisfy
the tool schema. Mesh constructs final arguments from only the generated report
body and `locked_arguments`; model-supplied extra arguments are discarded.

If the model returns another declared tool call, Mesh passes it through so the
agent can continue investigating. A prose response is wrapped as the report
call. A native call to the report tool is normalized to the same safe output.
Requests without `mesh_collaboration` are unchanged. Streaming is rejected in
this first version.

This contract deliberately requires a distinct structured report tool. It does
not interpret shell commands or try to distinguish investigative and publishing
uses of a generic shell tool.
