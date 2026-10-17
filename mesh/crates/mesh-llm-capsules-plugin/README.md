# capsules

A [mesh-llm](https://github.com/Mesh-LLM/mesh-llm) plugin that keeps a signed,
hash-chained record of every exchange a node serves or asks for, on that
node's own disk, and adds an **Evidence** page to the mesh-llm console. The
page shows the peers this node exchanged with, each exchange, and whether the
node's log still verifies.

Records follow the Agent Action Capsule format: JCS-canonical JSON, Ed25519
COSE_Sign1 signatures, and a checkpointed Merkle log. The plugin is Rust; the
page is a TypeScript bundle the plugin serves through mesh-llm's plugin web UI.

## The plugin's id

The plugin's id is **`capsules`**: the name it installs under, its data
directory, its settings' prefix (`CAPSULES_*`) and its page route
(`/plugins/capsules/evidence`). The repository has the same name, so
`mesh-llm plugins install Mesh-LLM/capsules` finds its release assets
(`capsules-<version>-<target>.tar.gz`) with no `--name`.

Until 0.1.3 the plugin was called `capsule-emit-mesh`. A node upgrading from
it keeps its key and records: see "Upgrading from capsule-emit-mesh" in
[`INSTALL.md`](INSTALL.md).

## Install

[`INSTALL.md`](INSTALL.md): download a release package for your platform,
check it, and install it with `mesh-llm plugins install --archive`. Release
packages are built for macOS arm64 and Linux x86_64/arm64.

## What works today

- The plugin installs on an unmodified mesh-llm, starts with the node, and adds
  the Evidence page.
- Each node seals its own record of every exchange, and the log's checkpoints
  and integrity checks work.
- When you turn it on (off by default), and mesh-llm identifies the other
  side of an exchange, the node sends it its own record at completion.
- A record pushed by the other side is received and checked in-process: the
  sender's announced key, the signature, the claims, and a bundle's proof and
  checkpoint. It is held only if every check passes; anything else gets a
  signed refusal. Record requests are answered in-process too, under the
  sharing policy (see [docs/SHARING-POLICY.md](docs/SHARING-POLICY.md)).
- When you turn it on (off by default), twins that answered the same request
  differently get an independent check: a referee, chosen by yes/no
  eligibility rules, re-answers the twins' request (so it receives the
  prompt) and signs a verdict
  ([docs/TWIN-REFEREE-SELECTION.md](docs/TWIN-REFEREE-SELECTION.md)).
  Every pair with no verdict reads "Not adjudicated" with the reason, and is
  never counted against either twin. A pair exists only when a client marks
  two requests as one (the `x-mesh-twin-bracket` header); otherwise the page
  reads "Not adjudicated: no client marked these as a pair" and nothing is
  called. Asking a referee sends it the twins' request and both their answers.

**An exchange confirmed by both sides** needs mesh-llm 0.78 or later, which
tells the plugin which peer asked and which peer served (see
[Host changes](#host-changes)), and record push turned on at both ends with
each node's key configured (INSTALL.md). It is off by default; on mesh-llm
0.77, or with it off, each node holds only its own record.

## Host changes

Changes to mesh-llm this plugin depends on or would use. Each is a generic
host feature; none names this plugin.

| Change | What it enables | Status |
| --- | --- | --- |
| Provider-side `payment.lifecycle.v1` events | the serving node's record of a paid exchange | mesh-llm#2108, in 0.78.0 |
| The exchange event on the paid serving path | paid exchanges are visible to plugins | mesh-llm#2109, in 0.78.0 |
| `requested_by_node_id` on the served side; `served_by_node_id` with request/response digests on the routed side | matching the two sides' records | mesh-llm#2138, in 0.78.0 |
| An operator's "stop routing to this peer" as a core router feature a plugin can request (behind the operator's `allow_peer_blocks`), plus a routing-choice event | acting on evidence without the plugin touching the router | mesh-llm#2142, in 0.78.0 |
| A plugin page can ask to be a primary console tab; the operator decides | a stable place for the Evidence page | mesh-llm#2130, in 0.78.0 |
| A plugin page can turn off the host's page header; contributions under a chat answer and in the Logs request inspector | the page's own header; the "sealed on this node" line | mesh-llm#2141 and #2140, in 0.78.0 |
| A built-in or default plugin can serve its web UI bundle | shipping this plugin by default | proposed |
| Per-request `skippy.stage.v1` events from split-inference stages | records for multi-stage requests | proposed |

## Layout

The plugin is the `mesh-llm-capsules-plugin` crate in the mesh-llm workspace;
its executable is still called `capsules`. Paths are relative to this
directory.

| Path | What |
| --- | --- |
| `src/`, `tests/` | the plugin (Rust, `mesh-llm-plugin`) |
| `web-ui/` | the Evidence page, built into one ES module (`bundle/register-mesh-plugin-ui.js`) |
| `vectors/` | pinned test vectors; `vectors/SOURCES.md` says where each set comes from |
| `docs/` | sharing policy, verification chain, split-stage records, protocol notes, real-host tests |
| `plugin.toml`, `plugin.package.json` | the package marker and the reviewed plugin manifest |
| `scripts/` | release packaging (`package.sh`) |
| `gates/` | the mutant table the plugin's own CI gates read |

The records are sealed by the [`capsule-emit`](https://crates.io/crates/capsule-emit)
crate on top of [`checkpointed-local-log`](https://crates.io/crates/checkpointed-local-log)
and [`evidencebook`](https://crates.io/crates/evidencebook); those crates are
developed and tested in their own repositories.

## Contributing

See mesh-llm's [CONTRIBUTING.md](../../../CONTRIBUTING.md). The plugin's own
checks beyond fmt, Clippy and its tests run one at a time with
`just plugin-capsules-gate <gate>`: `mutants`, `referee-mutants`,
`unbounded-depth`, `no-keys`, `vectors`, `web-ui` or `package-twice`.

## License

Apache-2.0. See [LICENSE](LICENSE).
