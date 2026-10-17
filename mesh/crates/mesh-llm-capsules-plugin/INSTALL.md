# Install the capsules plugin on a mesh-llm node

`capsules` is a mesh-llm plugin. It keeps a signed, hash-chained
record of every exchange this node serves or asks for, on this node's disk,
and adds an **Evidence** page to the console: the peers you exchanged with,
each exchange, and whether this node's log still verifies. It installs and
runs on an unmodified mesh-llm release (0.77 or newer). No other runtime is
needed.

## What's new in 0.1.4

- **Paid exchanges on the Evidence page:** what each one charged and who paid, from the records
  this node sealed, and "outcome not reported" where the host reported none. An exchange this node
  served for pay shows its own book as the provider.
- **The referee can take a twin pair's texts from the client,** and **peers' keys from the host**
  when mesh-llm announces them (`CAPSULES_PEER_KEYS` stays the fallback). Neither has run yet on a
  stock mesh-llm release: see the CHANGELOG.
- **The console's navigation stays on the Evidence page,** a served exchange with no other side
  reads LOCAL, and twin pairs show on Peers.

Details in [CHANGELOG.md](https://github.com/Mesh-LLM/capsules/blob/main/CHANGELOG.md).

## What's new in 0.1.3

- **The plugin is now called `capsules`** (it was `capsule-emit-mesh`), and so is its repository,
  `Mesh-LLM/capsules`. Its settings are `CAPSULES_*`, its page is `/plugins/capsules/evidence`, and
  its data directory is `~/.local/share/capsules`. A node upgrading keeps its key and records: see
  [Upgrading from capsule-emit-mesh](#upgrading-from-capsule-emit-mesh).
- **Settings saved in the console reach the plugin.** mesh-llm keeps them in its config file and
  does not pass them to the plugin, so the plugin reads them there. A node started with
  `--config <path>` must also have
  `MESH_LLM_CONFIG=<path>` in its environment, or the plugin reads `~/.mesh-llm/config.toml`.

- **Quiet by default.** Pushing this node's record to the counterparty, delivering verdicts, and
  asking a referee are now **off by default**; turn each on under Configuration › Plugins (see
  [What it sends](#what-it-sends-and-where-it-listens)). Answering a peer that asks for a record is
  unchanged.
- **Stop routing after N goes through the host's plugin path.** The rule now asks mesh-llm to block
  the peer as this plugin's request, which the host refuses unless you set `allow_peer_blocks =
  true` for this plugin. The plugin no longer calls the console's own route.
- **Several witnesses.** **Witnesses** in the console is a list: each witness's URL and, if you have
  it, its public key (still none by default, and no witness is contacted until you add one). The
  Evidence page shows what each witness holds, and counts a witness only once its receipt checks
  against the witness's key. A key you don't give is fetched from the witness once and kept, and
  the page says so. Every witness is contacted at exactly the URL you give, and no redirect is
  followed.
- **On mesh-llm 0.78 and later:** the Evidence page can be a primary console tab (your choice, under
  Configuration › Plugins), and a chat answer this node sealed shows "Sealed on this node" with a
  link to its record. An exchange can be confirmed by the other side once you turn record push on
  (see [What works today](#what-works-today)).

## What's new in 0.1.2

- **No models, no provider by default.** 0.1.1 registered an OpenAI-compatible provider on every
  node and listed `blocked-test-model` (it always answers 403) unless `CAPSULE_EMIT_MESH_BLOCKED_MODELS`
  was set. From 0.1.2 the plugin registers no provider and serves no models unless you name blocked
  models in that variable.

## What's new in 0.1.1

- **Settlement-record legs for paid exchanges** (draft-mih-agent-settlement-records-00): both
  sides seal their own legs for each invoice. The provider's leg needs a mesh-llm host whose
  settlement events carry the wallet's credited amount and fee (Mesh-LLM/mesh-llm#2169).
- **A node-unique log id** by default, so a witness accepts every node's checkpoints (see
  [The log's id](#the-logs-id)).
- **A served half is found by its digests** when the host forwarded no nonce. Both nodes need
  0.1.1 for this.
- **The witness indicator** says "off" only when no witness is set.

Details in [CHANGELOG.md](https://github.com/Mesh-LLM/capsules/blob/main/CHANGELOG.md).

## What it sends, and where it listens

Records are kept on this node. **With the default settings the plugin sends
nothing to any peer on its own**, idle or serving: it only answers a peer that
asks it for one of its records. Everything else that reaches a peer is off
until you turn it on:

| What | To whom | Default | To turn it on (or off) |
| --- | --- | --- | --- |
| This node's own sealed record of a completed exchange (a push) | the peer in that exchange | **off** | `share_record_at_completion = "counterparty"` |
| A verdict a referee signed, delivered to the nodes it judges | those nodes | **off** | `share_adjudications = "deliver_to_subjects"` |
| Asking a referee when two twins of a client-marked pair answered differently. **This sends the twins' request (the prompt's `messages`) and both twins' answers to the referee peer**, which answers the request with its own inference | one eligible peer | **off** | `adjudicate_differing_twins = "on"` |
| Stopping routing to a peer after N contradictions | (a request to your own host) | **off** | set `stop_routing_after_contradictions`, and `allow_peer_blocks = true` for this plugin in mesh-llm's config |
| One of this node's records, when a peer asks for it by id | the peer that record names as the other side | on: an answer, never a push | `share_history_segments = "off"` |
| A signed checkpoint of the log | the witness services you name | **off**: none by default | add witnesses under **Witnesses** |

The referee also needs the pair's text on this node. Stock mesh-llm passes no
exchange text to plugins, so the client that marked the pair supplies it
(`referee_supply_twin_texts`: the request and both answers). They are kept
only when the referee is on, each is the body this node's own records sealed,
and the answers differ. Turning the referee on, and supplying a pair's text,
is the decision to send a prompt to a third node.

All peer traffic uses mesh-llm's own peer connections. Nothing goes to any
third party unless you name a witness. The switches are under Configuration ›
Plugins › Sharing policy, and [`docs/SHARING-POLICY.md`](docs/SHARING-POLICY.md)
explains each one. The plugin reads what you save there from mesh-llm's
config file: `~/.mesh-llm/config.toml`, or the file `MESH_LLM_CONFIG` names.
**If you start the node with `--config <path>`, also set
`MESH_LLM_CONFIG=<path>`** in its environment, or the plugin reads the default
file and your console settings don't reach it. Settings in the environment win
over the console. The witness and the checkpoint cadence are read when the
plugin starts: restart the node after changing them.

The plugin listens only on `127.0.0.1`, at a random port the node reaches it
through.

## Install

One path, the same on macOS 11 or newer (Apple Silicon) and on Linux with
glibc 2.35 or newer (x86_64 or arm64).

**1. Download the package for your machine and the checksum file** from the
[releases page](https://github.com/Mesh-LLM/capsules/releases),
and set `VERSION` to the release (without the leading `v`):

```bash
VERSION=0.1.4
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  TARGET=aarch64-apple-darwin ;;
  Linux-x86_64)  TARGET=x86_64-unknown-linux-gnu ;;
  Linux-aarch64) TARGET=aarch64-unknown-linux-gnu ;;
  *) echo "unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
BASE=https://github.com/Mesh-LLM/capsules/releases/download/v$VERSION
curl -fLO "$BASE/capsules-$VERSION-$TARGET.tar.gz"
curl -fLO "$BASE/SHA256SUMS"
```

**2. Check the download.** This must print `OK` for your file:

```bash
shasum -a 256 -c SHA256SUMS --ignore-missing
```

With the GitHub CLI you can also check where it was built:

```bash
gh attestation verify "capsules-$VERSION-$TARGET.tar.gz" \
  --repo Mesh-LLM/capsules
```

The checksum proves your file is the one published. The attestation proves it
was built by this repository's release workflow from the tagged commit.
Neither says the code is safe to run; that is a judgement about the source,
which is public here.

**3. Install it.**

```bash
mesh-llm plugins install --archive "capsules-$VERSION-$TARGET.tar.gz" \
  --name capsules --version "$VERSION"
```

`mesh-llm plugins install Mesh-LLM/capsules` installs a release straight
from GitHub instead, without your own checks in steps 1 and 2. Either way the
command unpacks the
package into the node's plugin directory (`~/.mesh-llm/plugins`, or
`MESH_LLM_PLUGIN_DIR`) and enables it. `mesh-llm plugins info
capsules` shows what was installed.

**4. Restart the node.** An installed, enabled plugin starts with the node; no
config entry is needed.

**5. Open the console.** The plugin adds a page labelled **Evidence**. After
the node serves or asks for an exchange, the page shows the peer, the
exchange, and the log's integrity.

## What works today

| | |
| --- | --- |
| Install, start, the Evidence page | works |
| Each node seals its own record of each exchange | works |
| An exchange confirmed by the other side | works on mesh-llm 0.78 and later, once turned on (below); not on 0.77 |

A confirmed exchange is the serving node's record and the requesting node's
record, matched and each held by the other side. Since mesh-llm 0.78 the
exchange event names both nodes: `requested_by_node_id` on the event the
serving node sees (which peer asked), and `served_by_node_id` with the request
and response digests on the event the requesting node sees for a request
another peer served. mesh-llm 0.77 has neither, so there each node holds only
its own record, and nothing is shown as confirmed.

Confirmation is off by default, because it sends this node's record to the
other side: turn on `share_record_at_completion` (`counterparty`) on both
nodes, and give each node the other's key (below).

**Each node must know the other node's public key.** The plugin accepts a
record only from a peer whose key it knows, and refuses the rest. It knows a
key from either of two places:

- **The host,** when mesh-llm announces plugin keys (it lists the
  `plugin_keys.v1` capability): each directly-connected peer's key for this
  plugin, bound to that peer by its node key, read from the host's
  `GET /api/plugin-keys` (`CAPSULES_HOST_CONSOLE_URL`, default
  `http://127.0.0.1:3131`). This plugin announcing its own key there follows
  once a mesh-llm plugin release carries the request.
- **You:** set `CAPSULES_PEER_KEYS` in the node's environment to a JSON object
  mapping each peer's id to its raw Ed25519 public key in hex. A node writes
  its own id to `<data dir>/self-peer-id` and its public key to
  `<data dir>/keys/node-key.pub.pem`.

If the two give a peer different keys, nothing from that peer verifies until
you remove the stale entry.

**A client-only node (`mesh-llm client`) cannot be named there.** mesh-llm
gives a client node a new peer id each time it starts, so no entry in
`CAPSULES_PEER_KEYS` stays true, and its pushed records are refused by every
peer. Its own records are still sealed and kept on that node. A node started
with `mesh-llm serve` keeps its id across restarts of the same home directory
and can be named.

## Where the records are kept

Under the plugin's data directory: `CAPSULES_DATA_DIR` if set in the
node's environment (an absolute path), else `$XDG_DATA_HOME/capsules`,
else `~/.local/share/capsules`. The sealed log is
`<data dir>/ledger/capsules.jsonl` and its checkpoints are
`<data dir>/ledger/checkpoints.jsonl`. The node's signing key is under
`<data dir>/keys/`; it never leaves the node.

## The log's id

Every checkpoint names the log it covers, and a witness keeps one history per
log id, so no two nodes may share one. A node's log id is
`capsules/<key id>`, where the key id is the first 16 hex characters
of the SHA-256 of its public key. It is chosen at the first start and kept in
`<data dir>/log_id`; a later change of key does not change it. To name it
yourself, set `CAPSULES_LOG_ID` before the node's first checkpoint.

A log keeps the id its checkpoints were cut under: changing it would break the
log's checkpoint chain. So the plugin never changes the id of a log that has a
checkpoint, and refuses to start when `CAPSULES_LOG_ID` names a
different one.

**A node from an earlier release** whose log was checkpointed under the old
default, `capsule-emit-mesh` (the same on every node), keeps that id and logs
a warning at start. A witness accepts that id from one node only and refuses
the others' checkpoints as a fork. To take a log id of the node's own, start a
new log (Evidence › Clean up records › Start a new log) and restart the
node: the old log is kept whole under `<data dir>/archive/<n>/` with its old
id, and the new history runs under `capsules/<key id>/h<n+1>` (or
`<CAPSULES_LOG_ID>/h<n+1>` when that is set).

## Upgrading from capsule-emit-mesh

Until 0.1.3 the plugin was called `capsule-emit-mesh`. To upgrade, install
`capsules` as above, then remove the old one: `mesh-llm plugins delete
capsule-emit-mesh` (this leaves its data directory alone). Then restart the
node.

- **Records and key.** `capsules` keeps using the old default directory
  (`$XDG_DATA_HOME/capsule-emit-mesh` or `~/.local/share/capsule-emit-mesh`)
  where it is, under its old name: nothing is moved or copied. A directory you
  chose with `CAPSULES_DATA_DIR` (or the old `CAPSULE_EMIT_MESH_DATA_DIR`) is
  used as it is.
- **Stop the old plugin first.** On Linux the plugin refuses to start while
  another process has the log open (capsule-emit-mesh before 0.1.3 takes no
  lock), and says which process; two writers on one log would break its chain.
- **Both directories present.** If the node has a key or a log under both
  names, the plugin refuses to start and names both. Keep the one that is this
  node's, move the other out of the way, and start again.
- **The log id.** A log keeps the id it was checkpointed under, so an existing
  node's witness history continues. A new node's log id is
  `capsules/<key id>`.
- **Settings.** Every setting in the node's environment is now named
  `CAPSULES_*`. For this release the plugin still reads the old
  `CAPSULE_EMIT_MESH_*` name of a setting whose new name is unset, and logs a
  warning naming both; when both are set, the new name wins. Console settings
  saved under the old plugin name are read the same way. Rename yours now: the
  old names stop working in the next release.
- **Bookmarks.** The page moved from `/plugins/capsule-emit-mesh/evidence` to
  `/plugins/capsules/evidence`.

## Turn it off or remove it

```bash
mesh-llm plugins disable capsules   # keeps it installed
mesh-llm plugins delete capsules    # removes the installed files
```

Removing the plugin does not delete its data directory.

## What is in the package

```text
capsules/
  capsules                   the plugin executable
  plugin.toml                         package marker (name, version)
  plugin-manifest.json                settings schema and web UI declaration
  bundle/register-mesh-plugin-ui.js   the Evidence page
  README.md                           this file
  LICENSE
```

## Build it yourself

The packaging step (`scripts/package.sh`) is deterministic, so the same
executable always gives the same archive digest. To build a package for your
own machine from a mesh-llm checkout (the packaging step needs GNU tar and jq):

```bash
cargo build --locked --release -p mesh-llm-capsules-plugin --bin capsules
cd mesh/crates/mesh-llm-capsules-plugin
VERSION=$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' plugin.toml)
(cd web-ui && pnpm install --frozen-lockfile && pnpm build)
scripts/package.sh "$VERSION" "$TARGET" ../../../target/release/capsules dist
```

A different compiler or build machine can give a different executable and so a
different digest; `SHA256SUMS` covers the published packages.
