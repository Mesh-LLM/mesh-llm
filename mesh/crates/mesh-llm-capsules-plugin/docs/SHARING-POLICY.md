<!-- SPDX-License-Identifier: Apache-2.0 -->
# Sharing policy

Every sharing decision the plugin makes is one of six switches, and every
default keys on **relationship** (whether this node has exchanged with the
other node), never on proximity, latency, or any computed standing. Nothing
here is a score, and nothing here ranks anyone.

```
share_record_at_completion:  counterparty | off                    # default: off
share_history_segments:      counterparties | peers | off          # default: counterparties
share_standard_history:      on | off                              # default: off
advertise_evidence_requests: on | off                              # default: off
share_adjudications:         deliver_to_subjects | off             # default: off
witness:                     [] | [{endpoint, public_key?}, ...]   # default: [] (off)
```

With these defaults a node sends nothing to a peer on its own: it pushes no
record, delivers no verdict and advertises nothing, and it answers only a
counterparty that asks: for one of its records (`share_history_segments`) or
for its Standard History (`share_standard_history`). The referee
(`adjudicate_differing_twins`) is off by default too.

`share_history_segments` also accepts `prospective`, its default before
0.1.6. It is deprecated: it always served exactly whom `counterparties`
serves, so it is now read as `counterparties`, and a log line says so. Set
`counterparties` instead.

The plugin declares the six switches in its `config_schema`
(`mesh/crates/mesh-llm-capsules-plugin/src/share_policy.rs`), so the console shows them under
Configuration › Plugins › Sharing policy. mesh-llm keeps what you save there in
its config file and does not pass it to the plugin, so the plugin reads it
there (`~/.mesh-llm/config.toml`, or the file `MESH_LLM_CONFIG` names). The
same settings can be set in the plugin's environment, which wins over the
console: `CAPSULES_SHARE_RECORD_AT_COMPLETION`,
`CAPSULES_SHARE_HISTORY_SEGMENTS`, `CAPSULES_SHARE_STANDARD_HISTORY`,
`CAPSULES_ADVERTISE_EVIDENCE_REQUESTS`, `CAPSULES_SHARE_ADJUDICATIONS` and
`CAPSULES_CHECKPOINT_WITNESS_URLS` (a comma list of witness URLs) with
`CAPSULES_CHECKPOINT_WITNESS_KEYS` (a JSON object from witness URL to its key
in hex). An unset or unknown value means the default.

## The record at completion (`share_record_at_completion`)

Off by default. When it is on (`counterparty`) and an exchange ends, each side
sends its own **sealed record** to the other side over the plugin's
`record-push/1` mesh stream: the provider's record to
the requester, and the requester's to the provider. With the checkpoint
cadence on (the default), the record goes out as a **bundle**: the record, a
signed checkpoint of the sender's log, and the record's inclusion proof under
it.

The record contains nothing the other side can't already compute (they hold
the bytes, so both digests are theirs to derive). What it adds is the time it
was sealed, the signature, and, in a bundle, the size and root of the sender's
log and the record's position in it. No other record's content leaves: the
proof is hashes only. With both records exchanged, each side holds the other's
signed statement about the same exchange.

**Symmetric by default:** a node with this switch `off` neither sends its
record nor accepts the other side's.

**In this build:** the plugin sends its record when an exchange ends and
mesh-llm has told it who the other side is. A pushed record is received and
checked in-process (the sender's announced key, the signature, the claims, and
a bundle's proof and checkpoint) and held only if every check passes; anything
else gets a signed refusal. The sender's key must be known:
announced by its host, or configured (`CAPSULES_PEER_KEYS`, see INSTALL.md).
Both are keyed by peer id, and a client-only node gets a new id each time it
starts, so a push from a client node is never accepted (INSTALL.md says more).

## Reading records back (`share_history_segments`)

A peer can ask this node for one of its records by id, over the plugin's
`ledger-fetch/1` mesh stream. The switch decides who gets an answer:

| value | counterparty (a node this record names as the other side) | any other peer |
| --- | :---: | :---: |
| `off` | | |
| `counterparties` (default) | ✓ | |
| `peers` | ✓ | ✓ |

A local block and an owner-maintenance record are never served. A declined
request is answered `not_authorized`. The counterparty is the node the
request's signature proves it came from, checked as for an evidence request
(below) before anything is looked up: the id a request states is never
trusted. Under `counterparties` an unsigned request, or one whose signature
fails, is refused before the record is looked up; under `peers` an unsigned
request is answered, and a signature that fails any check is still refused.

A record of an exchange this node served for a peer names that peer as the
other side (mesh-llm reports it from the mesh connection, never from the
request). That is what makes the peer its counterparty, and under `peers` it
also means any peer that reads the record learns which node asked.

**Evidence requests** (`evidence-request/1`, draft-mih-agent-evidence-request-00)
are answered in-process by `evidence_answer.rs`, with the protocol from the
`capsule-emit-evidence-request` crate. The same rule applies to every record
an answer would carry, for every subject that carries record bodies (`record`,
`range`, `full_history`, `correlation`, `exchange`). An answer is served whole
or refused `not_authorized` whole, never filtered per asker, because an
artifact is the same for every requester.

**Who asks is proven, not stated.** The asking node signs its request with
the key its host announces for it (mesh-llm `plugin_keys.v1`), in a
`requester_auth` member: `{"key_id", "sig"}`, where `sig` is Ed25519 over
`evidence-request/1/requester-auth`, a zero byte, and the canonical JSON (RFC
8785, as the `capsule-emit-evidence-request` crate encodes it) of the rest of
the request. A signed request also carries an `audience` (the mesh id of the
node it is sent to), a `nonce` and a `deadline`.

Each kind of request signs under its own tag, so a request signed as one never
verifies as another: `ledger-fetch/1/requester-auth` for a record fetched by
id, `referee-adjudicate/1/requester-auth` for a referee's `adjudicate` request
(it travels on the evidence stream, and has no channel name of its own), and
`evidence-request/1/requester-auth` for every other evidence request. The
answering node picks the tag from where the request came in, never from a tag
the request names.

Before it looks anything up, the answering node checks, in order:

- the deadline is no more than 5 minutes ahead;
- the signature verifies;
- the `audience` is this node, so a request one node received cannot be
  forwarded to another;
- the nonce has not been used before its deadline. For a key a host
  announced this is strict: a replay is refused, and so is a request once that
  key holds 4,096 live nonces. Keys no host announced share one pool of 16,384
  nonces, held until their deadlines, and the check there is best-effort: a
  replay found in the pool is refused, but while the pool is full of live
  nonces a new one is not held, and the request goes on unchecked (it can then
  only be answered as an unsigned request would be, below);
- a host announced the key for a node. If several nodes announced it, the
  request's `requester_id` must be one of them. If only one did, the request's
  `requester_id`, when given, must be that node. An entry in
  `CAPSULES_PEER_KEYS` alone does not count, and one that disagrees with the
  host's vetoes it.

That node is the asker. Under `counterparties`:

- a record goes only to the node it names, as proven this way;
- an unsigned request is refused `not_authorized`;
- a proven node that no record names is refused `not_authorized` before the
  subject is resolved, the same refusal for any subject, so which refusal it
  gets tells it nothing about the log's size.

Under `peers` an unsigned request is answered, as before. A signed one is
checked all the same. A valid signature under a key no host announced is the
one failure answered as an unsigned request would be, and only where an
unsigned request is answered. Every other failed check is refused.

Each request's digest is recorded in the "asked of you" log, with the key and
node it was proven to come from (or `unbound`), the audience it named, and
`replay`: `checked`, or `unchecked` when a key no host announced found the
pool full.

The checkpoint list, the `history_card/1` derivation and the `served_summary/1`
derivation carry no record bodies: `share_history_segments` does not govern
them, and `share_standard_history` does (below). The golden answers are in
`vectors/parity/evidence_request/`.

## Standard History (`share_standard_history`)

Off by default. Standard History is what an evidence request can get that
carries no record bodies: the checkpoint list, `history_card/1` and
`served_summary/1` (and any other derivation). It carries no records, but it
shows how much and how often this node serves, so with this switch off it goes
only to a counterparty: a node that some record of this node names as the
other side of an exchange. Any other node is refused `policy_declined`.

The refusal carries nothing about who asked. It is decided before the subject
is resolved, so it is the same whether or not what was asked for exists, and
it is signed over the request's digest, the reason and the time alone. Two
nodes sending the same request in the same second get the same bytes back;
two refusals differ only in the digest of the request each refuses (which
covers every member each carries), the time, and the signature over them.

With it on (`on`), every node that asks is answered, as before 0.1.6. It
never releases a record: record bodies stay under `share_history_segments`.

**A counterparty is the node a signed request proves it is**, as above.
These all get the same `policy_declined`, so trying ids teaches a node
nothing about who this one has exchanged with:

- an unsigned request, whatever id it states;
- a request stating another node's id;
- a proven node no record names.

## Advertising evidence requests (`advertise_evidence_requests`)

Off by default: this node does not tell other nodes that it answers evidence
requests. It still answers a node that asks, under the switches above. This
version advertises nothing, so nothing reads the switch yet: the plugin's
`evidence-request/1` channel is declared only to its own host, so that a
request can reach it, and mesh-llm does not pass a plugin's channels on to
other nodes.

## Verdicts (`share_adjudications`)

Off by default. When it is on (`deliver_to_subjects`) and a referee's verdict
about an exchange exists, it is delivered to every node it concerns, and each
seals its own record citing it. A node holds a
delivered verdict only when the referee it names signed it with its announced
key and the verdict concerns that node (it asked for it, or it judges one of
that node's own records). With `share_adjudications: off` the node that asked
still holds the verdict, and delivers it to no one. How a referee is chosen:
[TWIN-REFEREE-SELECTION.md](TWIN-REFEREE-SELECTION.md).

**Asking a referee sends a prompt to a third node.** With
`adjudicate_differing_twins` on (off by default), when two twins of a pair a
client marked (`x-mesh-twin-bracket`) answered differently, this node sends the
twins' own request, including the prompt's `messages`, to one eligible peer
(`x-mesh-target`), which answers it with its own inference; that answer and
both twins' halves then go to it to adjudicate. It needs the exchange text
kept on this node, a separate setting (`CAPSULES_KEEP_EXCHANGE_TEXT`, off)
that keeping text locally does not imply the other way round: keeping text
never sends it to a peer. The operator's "Ask again" is the same switch.

## Witness

`witness` is a list of witnesses, empty for off (the default). Each is an
`endpoint` (the witness's URL) and, if you have it, its `public_key` (its
Ed25519 key in hex, as its operator publishes it): the same endpoint-and-key
pair capsule-cli takes. There is no default witness: with the list empty the
plugin contacts no witness at all.

Each checkpoint is offered to every witness on the list. A witness that
accepts it returns a receipt, and the Evidence page lists every witness by
name with what it holds: the latest checkpoint, an earlier one, or none yet.

A receipt counts only once the plugin has **checked** it, offline. It
recomputes the checkpoint's digest from the checkpoint's own signed fields,
confirms the receipt is for exactly that entry, and verifies the receipt's
inclusion proof and signature under the witness's key: the key you gave, or,
when you gave none, a key fetched from that witness the first time it is
needed and kept (`<ledger>/witness-keys.json`). A different key later from the
same URL is not taken. The page says which: "key configured", or "key pinned
on first contact, not configured". A key you configure is the stronger choice:
you got it from the witness's operator, not from the witness's own answer.
"Witnessed" on the page means at least one checked receipt, from a witness it
names.

A named witness that does not hold the latest checkpoint is asked for its key
again, at most once a minute and only that witness, so its row can say why:
it did not answer in time, refused the connection, answered with an HTTP
error, or presents a different key than the one configured or pinned. A
witness that holds the latest checkpoint is not contacted for this.

Every request to a witness goes to the URL the operator gave, host, port and
path as written; no URL is mapped to another service. No request follows a
redirect. The plugin's own reads of a witness (its key, and what it holds of
this log) read at most 64 KiB; registering a checkpoint, through the
capsule-emit library's client, reads at most 1 MiB.

A witness holding a checkpoint means one thing: a later rewrite of this log is
detectable by someone other than this node. It does not make the records true.

## The two mesh streams are plugin-internal

`record-push/1` (a record sent at completion) and `ledger-fetch/1` (a record
read back by id) are this plugin's own behaviour between two copies of
itself. No public specification defines them. They are not a standardized
wire, other implementations should not target them, and they may change
between releases. Neither stores and forwards a record, and neither promises
delivery.

## Three rules every switch follows

1. **Filter, never rank.** Every switch answers or refuses; none orders or
   weights a peer.
2. **Counts, not sums.** Where counts are reported, they are counts; nothing
   is totalled or averaged into a standing.
3. **Relationship, never proximity.** Every default keys on whether this node
   has actually exchanged with the other node.
