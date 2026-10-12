# Threat model — mesh-llm

## What this project does
mesh-llm is a peer-to-peer LLM inference mesh written in Rust. Nodes discover each
other over a public network (QUIC/iroh), admit remote peers into inference sessions,
shard model execution across machines, and optionally exchange Lightning payments for
inference. It also exposes a local HTTP/OpenAI-compatible API, a web console, an MCP
surface, and an external plugin protocol.

## Where untrusted input enters (highest value targets)
- **Remote peer traffic**: gossip/discovery, admission and session-setup messages,
  activation/KV tensor payloads from other nodes. Any peer on the public mesh is
  untrusted and may be hostile or pretend to hold capacity it does not have.
- **Payment and wallet paths**: invoices, amounts, budgets, ledger accounting,
  replay of payment messages, wallet plugin RPC.
- **Model artifacts**: GGUF / packaged model parsing, Hugging Face downloads,
  checksum and signature verification, archive extraction (zip-slip style bugs).
- **Plugin protocol**: external plugins are separate processes; the host must not
  trust plugin responses, and plugins must not be able to escalate into the host.
- **Local HTTP surfaces**: OpenAI-compatible endpoint, console server, metrics.
  Treat these as reachable from other local processes and from the browser (CSRF,
  DNS rebinding, missing origin checks are in scope).

## Out of scope — please do not report findings in these
These are developer and CI conveniences. They run on a maintainer's machine or in
CI with trusted input, never on a user's node and never on untrusted data. Shell
quoting, argument injection, `curl | sh`, path handling and temp-file use in them
are intentional and not vulnerabilities.

- `scripts/` and `mesh/scripts/`, `skippy/scripts/` — ~200 dev/release shell scripts.
- `ci/`, `.github/`, `just/`, `Justfile`, `tools/xtask` — build, lint and release automation.
- `mesh/evals/`, `skippy/evals/`, `mesh/deploy/`, `mesh/website/`, `mesh/docs/`,
  `skippy/docs/`, `install.sh`, `install.ps1`.
- `mesh/crates/mesh-llm-test-harness`, `skippy/crates/skippy-bench`,
  `skippy-gpu-bench`, `llama-spec-bench`, and any `benches/`, `tests/`, `examples/`,
  or `#[cfg(test)]` code. Test-only panics, unwraps and weak randomness are fine.
- `skippy/llama_cpp/` and other vendored third-party code — report upstream, unless
  the bug is in how mesh-llm calls into it (per SECURITY.md).
- Anything requiring the attacker to already have shell access on the node.

In short: a finding only counts if it is reachable in a shipped `mesh-llm` binary
from input an attacker controls.

## How you rate severity
- **Critical**: remote pre-auth memory corruption or RCE from peer traffic or a
  model artifact; theft or unauthorised spend of wallet funds; bypass of payment
  or admission authentication; key/secret exfiltration.
- **High**: memory corruption that needs a partly trusted peer; budget/ledger
  accounting bypass (paying twice, being paid nothing); plugin→host escape;
  unauthenticated access to local API/console that can run inference or read
  conversation history.
- **Medium**: remotely triggerable panics/DoS, resource exhaustion, log leakage of
  secrets, missing redaction.
- **Low**: issues needing unrealistic preconditions or local root.

## How to exercise it
`cargo test` across the workspace; `crates/mesh-llm-test-harness` has multi-node
helpers. GPU and real model weights are not available in the scan environment, so
prefer protocol-level and parsing-level reproducers over end-to-end inference runs.

## Anything to leave alone
- Do not report "no TLS on localhost-bound ports" without a concrete attack.
- Do not report needing a model download as a vulnerability.
- Please include a concrete reproducer (test or fuzz input) with each finding.
