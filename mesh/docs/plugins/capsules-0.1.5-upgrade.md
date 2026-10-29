# Upgrading to the default capsules 0.1.5

mesh-llm 0.79.0 is the first release that installs a default plugin, and its
default is `capsules` 0.1.5. Releases up to 0.78.1 have no default plugins, so
on those a node runs `capsules` only if its operator installed it. This note
says what changes for such a node and for its peers, in what order to upgrade,
and how to go back. Statements about `capsules` itself come from its
[0.1.5 changelog](https://github.com/Mesh-LLM/capsules/blob/v0.1.5/CHANGELOG.md).

`capsules` keeps a signed record of the exchanges a node serves and answers
peers' evidence requests about them. Its default entry allows no inference
endpoint, virtual model or OpenAI exchange hook, and a default plugin that
fails to start does not stop the node (see [Default Plugins](README.md#default-plugins)).
Nothing below changes inference, routing or mesh membership; it changes only
which evidence requests one node answers for another.

## What a node gets when it moves to 0.79

| Before the upgrade | After the first 0.79 start |
| --- | --- |
| No `capsules` installed | `capsules` 0.1.5, installed from the release's bundled copy on Apple Silicon macOS and on x86_64 or arm64 Linux, the platforms it is pinned for. `--no-default-plugins`, `MESH_LLM_NO_DEFAULT_PLUGINS=1`, or a default turned off earlier (`plugins install-defaults --off`, or a deleted default) skips it. |
| `capsules` installed by the operator (`mesh-llm plugins install`, or a `[[plugin]]` entry with `command` or `url`) | Left at the installed version. A node start says nothing about it; `mesh-llm plugins install-defaults` reports that it leaves it alone. |
| A default-managed `capsules` 0.1.4, which only a build from `main` with a bundled copy has (no release does) | Replaced by 0.1.5 at the next start if it is enabled; a disabled one stays at 0.1.4. |

A 0.79 host announces plugin keys (`plugin_keys.v1`, Mesh-LLM/mesh-llm#2299),
so `capsules` 0.1.5 on it signs its evidence requests. On 0.78.1 or older
there are no plugin keys, so 0.1.5 installed there sends unsigned requests.

## Which versions answer each other

| Asking node | Answering node on 0.1.5 answers |
| --- | --- |
| 0.1.5 on a 0.79 host | Everything its sharing settings allow. |
| 0.1.5 on a host older than 0.79 (unsigned requests) | Checkpoints and `history_card/1`. No records under any setting. `served_summary/1` only with `share_standard_history = on`. |
| 0.1.4 | Nothing it can read: 0.1.5 answers in the interop profile's signed envelope, which 0.1.4 cannot parse. Its unsigned requests for records and `served_summary/1` are also refused. |

In the other direction, a 0.1.5 node cannot read a 0.1.4 node's answers. So
between 0.1.4 and 0.1.5 no evidence request succeeds, either way, until both
run 0.1.5. Each node still seals its own records meanwhile. This note does not
cover the peer features that are off by default (record push, the referee)
across versions; that has not been checked.

Signed requests also depend on the host change in
Mesh-LLM/mesh-llm#2620: without it, a `capsules` process the host restarts
does not learn its own node id until the next mesh event, and refuses every
signed request until then (Mesh-LLM/capsules#48). This pin is meant to ship
in the same release as that change.

## Upgrade order

1. **Upgrade mesh-llm on every node to 0.79.** This gives each node plugin
   keys. Nodes without `capsules` get 0.1.5 now; nodes whose operator
   installed it keep their version.
2. **Move every operator-installed `capsules` to 0.1.5 in one window.**
   Evidence requests between nodes that have and have not moved fail during
   this window. For each node, either:
   - install the 0.1.5 package as its
     [INSTALL.md](https://github.com/Mesh-LLM/capsules/blob/v0.1.5/INSTALL.md)
     describes, or point the `[[plugin]]` entry's `command` or `url` at 0.1.5;
     or
   - hand it to the release's default. If a `[[plugin]]` entry for `capsules`
     sets `command` or `url`, remove those two fields first (keep its
     settings): while either is set, the node runs that entry and never the
     default. Then run `mesh-llm plugins delete capsules` and
     `mesh-llm plugins enable capsules`, and restart. Deleting removes only
     the installed files; the records and key in the plugin's data directory
     stay.
3. **Take new anchor pins** (see [Pins and digests](#pins-and-digests)).

A mesh that cannot move every node in one window keeps evidence working on
0.1.4 instead, and later moves the whole mesh to 0.1.5 together. A node new to
0.79 then installs the 0.1.4 package (see [Rolling back](#rolling-back))
**before its first 0.79 start**, so 0.1.5 never writes its log. If the node
has already started 0.79, follow the rollback steps, backup included.

## Keeping `served_summary/1` answered during the transition

On 0.1.5, `served_summary/1` goes only to a counterparty proven by a signed
request. To keep answering it to nodes that cannot sign yet, the 0.1.5 node
that answers sets `share_standard_history = on`: in the console under the
`capsules` sharing policy (Standard history), or with
`CAPSULES_SHARE_STANDARD_HISTORY=on` in its environment, which wins over the
console. A node started with `--config <path>` also needs
`MESH_LLM_CONFIG=<path>` in its environment, or `capsules` reads the default
config file and the console setting does not reach it.

That answers `served_summary/1` to any node, signed or not: before 0.1.5 every
node could read it, so this restores the old exposure, not more. It helps a
0.1.5 node on a host older than 0.79, or one whose host refused its key; it
does not help a 0.1.4 node, which cannot read the answer. Turn it off again
once every peer runs 0.1.5 on 0.79. No setting answers an unsigned request for
a record.

The records status (`evidence_records_status`) shows whether a node's own
requests are signed (`requests_signed`).

## Pins and digests

0.1.5 pins a checkpoint by the SHA-256 of its COSE statement instead of the
digest of its JSON form, and its answers carry different octets. Every
`artifact_digest` and anchor pin taken from a node on 0.1.4 stops reproducing
once that node runs 0.1.5, and a request pinned to an old anchor is refused
`coverage_unsatisfiable`. mesh-llm itself stores no pins.

Whoever kept a pin, a peer's page or an outside verifier, replaces it after the
answering node upgrades: ask that node for `history_card/1`, which every
requester gets and which carries each checkpoint with its
`consistency_from_previous` proof. Before pinning the new anchor, check that
the history still holds the checkpoint you pinned (same log id, tree size and
root) and that every consistency proof from it to the newest checkpoint
verifies; otherwise you would accept a log rewritten at the
upgrade. If the old checkpoint is missing, treat it as a broken history, not
as a migration.

## Rolling back

Before rolling `capsules` back, copy its data directory. It is
`CAPSULES_DATA_DIR` if set, else `$XDG_DATA_HOME/capsules`, else
`~/.local/share/capsules`. A node that started as `capsule-emit-mesh` (before
0.1.3) keeps using that directory instead: `CAPSULE_EMIT_MESH_DATA_DIR`, or
`capsule-emit-mesh` under `$XDG_DATA_HOME` or `~/.local/share`. The 0.1.5
changelog names no change to the on-disk log, but a 0.1.4 node reading a log
that 0.1.5 wrote has not been tested.

- **`capsules` back to 0.1.4 on a 0.79 node.** Download and check the 0.1.4
  package for the node's platform as steps 1 and 2 of the
  [0.1.4 INSTALL.md](https://github.com/Mesh-LLM/capsules/blob/v0.1.4/INSTALL.md#install)
  describe (they set `TARGET`), then run

  ```bash
  mesh-llm plugins install --archive "capsules-0.1.4-$TARGET.tar.gz" \
    --name capsules --version 0.1.4
  ```

  and restart the node. The install is now operator-managed, so later starts
  leave it at 0.1.4. Its peers on 0.1.5 can no longer exchange evidence with
  it. To return to the release's default, run `mesh-llm plugins delete
  capsules` and `mesh-llm plugins enable capsules`, then restart.
- **Turning `capsules` off.** Run `mesh-llm plugins disable capsules`, then
  restart the node: the command changes the plugin store, not the running
  process. A disabled plugin stays installed, and later starts neither run nor
  reinstall it. A `[[plugin]]` entry that sets `command` or `url` runs regardless of the
  store, so set `enabled = false` on that entry instead. Neither deletes the
  data directory.
- **mesh-llm back to 0.78.1.** 0.78.1 has no default plugins and does not
  remove the one 0.79 installed, so `capsules` 0.1.5 stays and runs unsigned.
  Install 0.1.4 as above, or disable it, if the mesh still runs 0.1.4.
