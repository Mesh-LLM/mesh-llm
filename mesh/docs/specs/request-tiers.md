# Request tiers

> **Not in effect on the Skippy engine yet.** The node still classifies every
> request and its hooks and plugins run at that tier. The engine half below
> (admission order, KV reservation, eviction order and per-tier caches) needs
> request tiers in Skippy ([Mesh-LLM/skippy#113](https://github.com/Mesh-LLM/skippy/pull/113)),
> which the Skippy release this build uses does not have. Until then a
> hook on the local server runs at `mesh`, the lowest tier.

A node serves requests for its owner before requests for the rest of the
mesh. Every request it serves gets a **tier** when it enters the node, and
the tier decides the order of admission, which KV cache entries go first
when memory runs out, and which prompt cache the request may reuse.

## Tiers

| Tier | Who |
|---|---|
| `local` | This machine; a node on its `[private] nodes` list; a node with a verified certificate from this node's own owner |
| `trusted` | A node with a verified certificate from an owner in this node's trust store |
| `mesh` | Everyone else, including any node without a verified certificate |

A node on the `[private] nodes` list is `local` whatever its certificate
says, and no owner override changes that. A certificate counts only while it
is valid: an expired, revoked or unverifiable one makes the node `mesh`. The
node reads its trust store when it starts, so an owner added with
`mesh-llm auth trust add` is `trusted` after the next restart.

A shared-serving membership classifies the nodes of the mesh it joined
against that membership's own trust store, its admission allowlist. Every
node the membership admits is therefore `trusted` there, and shares the
`trusted` cache with the main node's trusted owners.

`[request_tiers] owners` in `config.toml` replaces the tier of one owner's
verified nodes:

```toml
[request_tiers.owners]
"8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c" = "trusted"
```

An entry can raise or lower an owner. It never applies to a node without a
verified certificate from that owner. A running node rereads the entries
when the file changes. While the file is missing or does not load, the node
keeps the entries it last read, so a broken edit cannot lift an owner that
an entry lowers; owner ids that differ only in letter case are rejected.

## Assignment

The node a request enters decides its tier once:

- A request from a client of this node (the API port, the console's `/v1`
  passthrough, the native SDK, a plugin's own inference) is `local`. That
  includes a remote caller of a console opened with `--listen-all`.
- A request a peer forwarded is at the tier this node gives that peer.

A node that forwards a request names its client's tier in the
`x-mesh-request-tier` header. The receiving node serves the request at the
lower of that claim and its own tier for the sender, so a peer can lower
its own requests but never raise them. A client may send the header to
lower its own request. A value that is not a tier name, or a head that
cannot be read, counts as `mesh`.

The tier follows the request's work on this node: peer consultations,
including those Skippy's generation hooks start, the peers a virtual
model's turn calls, the peer a realtime session is relayed to and the
plugins that see the request (exchange hooks, request transforms, a
plugin-served model). A plugin calls back into the node on a separate task,
so while a plugin handles a request below `local`, every nested call from
that plugin runs at the lowest tier it is handling. Each call to the plugin
holds that from when the plugin is sent the request until it replies: a
client that leaves does not end it, and a call that ended without a reply
keeps it for five minutes. The binding is per plugin, not per request, so a
`local` request the same plugin handles at the same moment is lowered too;
once the plugin has replied, its later calls are not lowered.

## What the tier changes

Skippy reads the tier from the request and applies it to the model's
admission and caches:

- **Admission.** Waiting requests are admitted highest tier first. Each
  time 16 higher-tier requests are admitted ahead of a waiting request, it
  ranks one tier higher, up to the tier of another waiting request, so a
  lower-tier request that fits the free KV budget gets a share under
  steady higher-tier load. Ranking orders only requests that fit: a
  lower-tier request too large for the tokens higher tiers leave free (for
  `trusted` and `mesh`, outside the `local` reserve) is passed by every
  higher-tier request that fits, and waits until higher-tier load leaves it
  room or its admission timeout ends the wait. Waiting behind its own tier
  does not age a request: a long `mesh` queue never outranks a `local`
  arrival. Within a tier the usual cache-aware order applies whatever the
  ages. Each tier's oldest waiter has the admission starvation bound,
  checked from the highest tier down: past it, the waiter is admitted next
  if it fits and no higher-tier waiter does, and otherwise holds back its
  own and lower tiers, never a higher one, while the pool drains. Only waiting
  requests of the same or a higher tier count against the waiting-room
  limit, so mesh traffic cannot fill the room ahead of a `local` request.
  The advisory predicted wait reported for a request counts the same
  waiters, plus the requests already running.
- **KV reservation.** `[request_tiers] local_kv_reserve_percent` keeps a
  share of each model's KV token budget for `local` requests, rounded up to
  a whole token; other tiers can use only the rest. A request of another tier that needs more than the
  rest is refused as exceeding the context, which a forwarding node retries
  on another host. The default, `0`, reserves nothing; the setting applies
  when a model loads.
- **Eviction.** Every resident prefix and recurrent-state entry records the
  tier that stored it. Under memory pressure the lowest tier is evicted
  first, then the usual cost and age order applies within a tier, however
  old the higher tier's entries are. While higher-tier entries fill the
  exact-state cache, a lower tier's snapshot that would only evict itself
  is not taken.
- **Cache isolation.** Each tier keeps its own prompt cache, and private
  requests (`@private`, `x-mesh-scope: private`) have their own cache within
  their tier. A request never reuses a prefix another tier, or the other
  side of the private line, computed. A `local` request outside the private
  partition keeps the cache keys standalone Skippy uses.

A request already running is not interrupted when a higher tier arrives.

## Split serving

A coordinator stamps the tier and private flag of every request on the
cache-bearing messages it sends its stages. When a node loads a stage, it
records the tier it gives the coordinator; that stage serves the
coordinator's messages no higher than that tier, whatever they claim, and
evicts by the result. It partitions its caches by both the tier it serves
and the one claimed, so claims the cap merges still never share a prefix.
The tier a stage allows is fixed when the stage loads: a change to the
trust store or the owner overrides, or a certificate that expires or is
revoked, applies to the next load.

The last stage sends its predictions back to the coordinator's stage 0, and
only that stage's owner may open a stage connection to it. Stage 0 serves
stage messages no higher than the tier the coordinator gives that node.

This is stage protocol generation 17. All stages of a split must run it;
older stages fail stage negotiation.

## Limits

- Disk (L2/L3) prefix caches are partitioned by tier but are not evicted in
  tier order.
- Skippy's realtime and transcription paths do not read the tier; it only
  chooses the peer a realtime session goes to.
- A middle stage of a split forwards the coordinator's stamp to the next
  stage, which caps it by its own tier for the coordinator. A middle stage
  could change the stamp within that cap; it already sees every token of
  the request.
- A node on a public API port treats every client of that port as `local`.
- A plugin's nested inference is lowered per plugin, not per request: while
  a plugin handles a `mesh` request, or for five minutes after one that
  ended without a reply, the plugin's calls for `local` requests run at
  `mesh` too.
- A native SDK call forwarded to a peer carries no claim; the peer serves it
  at its tier for the calling node. Native SDK requests have no private
  scope, so they are never served as private; carrying one would need a
  private flag in the native request.
- After admission, a request takes one of the runtime's execution lanes,
  which it shares with split-stage clients of the same runtime in arrival
  order, whatever the tier.
