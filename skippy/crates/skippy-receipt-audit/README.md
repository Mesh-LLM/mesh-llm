# skippy-receipt-audit

Audits one turn that a client received against the GGUF that the node
claims to serve. The checks themselves, the wire shapes and the VRF are in
[`skippy-receipts`](../skippy-receipts/README.md). This crate runs them on
a real model. It opens the claimed file, loads its layers with the native
runtime, asks the node to open the challenge, and adds the results up to
an outcome and a report.

The design is in [RECEIPTS.md](../../../mesh/docs/design/RECEIPTS.md) and
[RECEIPTS_GLUE.md](../../../mesh/docs/design/RECEIPTS_GLUE.md). The feature
is experimental. To serve and audit receipts, see
[skippy/docs/RECEIPTS.md](../../docs/RECEIPTS.md).

## What the crate owns

| Module | Responsibility |
|---|---|
| `turn_audit` | The audit of one received turn: the binding, the challenge that the VRF draws, the opening that the node signs, and the outcome. `OpenChallenge` is how the caller sends the challenge to the node |
| `recheck` | Runs the same checks again on the evidence that an audit kept |
| `opening_checks` | Every check of an opened turn: the glue and weight rows of each picked layer, the input of layer 0, the output head, the tokens and the recurrent regions |
| `attention_audit` | The attention check, and the second request for the K and V rows |
| `recurrent_audit` | The recurrent check |
| `token_check` | The response and the prompt against the tokens that the receipt names |
| `claimed_model` | `ClaimedModel`, the claimed GGUF as an audit needs it, and the KV cache types of a layer |
| `native_model` | `ClaimedModel` over libllama: one layer at a time on the CPU |
| `claimed_file` | Opens the claimed GGUF, either a local file or `repo@revision/file` on Hugging Face read by range into a sparse copy |
| `head_key_file` | Where the client keeps its secret head key for a model, readable by its owner alone |
| `chat_reply` | The reply that the client received for the turn |
| `outcome` | The four ways an audit ends, and what one check found |
| `output` | The report of an audit as JSON types |
| `report_text` | The report of an audit as text |
| `node_text` | Makes text that the node wrote safe to print |

## What the crate does not own

- The checks on opened values, the wire shapes, the challenge draw and the
  evidence format are in [`skippy-receipts`](../skippy-receipts/README.md).
- The crate sends no chat request and prints nothing. The
  `skippy audit-receipt` and `skippy recheck-receipt` commands, their HTTP
  calls, the evidence file and the process exit code are in
  [`skippy-commands`](../skippy-commands/README.md).
- Which turns a Mesh node audits, the peer transport and what a verdict
  costs a peer are in `mesh-llm-host-runtime`.
- The crate has no dependency on `skippy-commands`, on a Mesh crate, or on
  clap.

## Consumers

- [`skippy-commands`](../skippy-commands/README.md) runs
  `skippy audit-receipt` and `skippy recheck-receipt` on it.
- [`mesh-llm-host-runtime`](../../../mesh/crates/mesh-llm-host-runtime/README.md)
  audits the turns that the peers of a Mesh node serve
  (`src/network/receipt_audit/`). It supplies its own `OpenChallenge` over
  the Mesh transport.
