# Evidence page data routes

The Evidence page is a mesh-llm plugin page. The console's plugin web UI
contract lets a page reach its own plugin's routes, and only those, through
`host.network.fetchPlugin(path)`. The console serves a plugin's declared
HTTP bindings at `/api/plugins/capsules/http/<path>` and its tools at
`/api/plugins/capsules/tools/<name>`, so the page asks for
`http/<path>` and `tools/<name>`. It reads **plugin-owned routes only**, and
no host route beyond the two console reads below.

Plugin HTTP routes (`mesh_llm_plugin::http::get`) answer JSON, so files the
plugin keeps on disk arrive wrapped in JSON.

| Plugin route, under `http/` (GET unless noted) | Answers | Read from (under the plugin's data dir) |
| --- | --- | --- |
| `ledger` | `{ "records": [<capsule>...], "node_pub_key_pem": "<PEM>" \| null }`; padding records are never listed; an empty list while nothing is sealed | `ledger/capsules.jsonl` + `ledger/node-key.pub.pem` |
| `ledger/signed-statement?capsule_id=<id>` | `{ "signed_statement_b64": "<COSE_Sign1, base64>" \| null }` | `ledger/signed-statements/<id>.cose` |
| `ledger/disclosure?capsule_id=<id>` | `{ "disclosure": {...} \| null }` | `ledger/disclosures/<id>.json` |
| `panes/pane-a` | pane A JSON (`api/sidecarTypes.ts` `PaneAJson`) | the ledger |
| `panes/pane-b` | pane B JSON (`PaneBJson`) | the ledger |
| `panes/pane-c[?limit=&after_seq=]` | pane C list (`PaneCListJson`) | the ledger |
| `panes/pane-c?exchange_id=<id>` | pane C drilldown (`PaneCDrilldownJson`) | the ledger |
| `peer-key?peer=<64 hex>` | `{ "announced_key_id": "<hex>" \| null }`: the key that peer is known to sign with (announced by its host, or `CAPSULES_PEER_KEYS`); "Ask them for their record" judges a reply only under it, never under a key the reply names | the plugin's environment |
| POST `tools/mesh_ledger_fetch` (a tool, not under `http/`) | the plugin's `mesh_ledger_fetch` tool: `LedgerFetchResponse` (`found` / `not_found` / `not_authorized` / `error`) | a peer's plugin, over a mesh stream |
| POST `tools/mesh_evidence_request` (a tool) | the plugin's `mesh_evidence_request` tool, `{peer_id, request}`, verifying by default: `{ "answer": <the peer's reply, unchanged: an artifact, or the refusal it signed>, "request_digest": "<64 hex: the bytes the plugin sent>", "verification": { "state": "refusal" \| "artifact" \| "not_evidence" \| "no_announced_key", ... } }` (`evidence_answer::verify_response`, under the peer's announced key; `"verify": false` returns the reply alone). The page's "Ask them for their record" sends a -00 `correlation` request by the exchange's client nonce; when the peer answers, signed, that it holds no record of that nonce (a half the peer served carries a placeholder nonce when its host forwarded none), it asks again by both of the exchange's digests, `exchange-digests:<request digest>:<response digest>`. It trusts only a verified reply, and says on the row whether it was verified | a peer's plugin, over a mesh stream |

**Split requests.** A `panes/pane-c` row may carry `split` (`api/sidecarTypes.ts`
`PaneCRow.split`, `lib/split-stage.ts` `SplitRowJson`):
`{ "viewer": "requester" | "coordinator", "main": <capsule>, "stage_records": [<capsule>...] }`,
the coordinator's main record and the stage records it carried (on a requester, from
`received-capsules.jsonl` and `received-split-stage-records.jsonl`), or
`{ "viewer": "stage", "stage_block": <x-mesh-stage-v1> }` on a stage node's own row. The page
runs the hand-off check itself (`verifySplit`) and draws the stage strip under the row. Rows
without `split` render as before. No pane emits it yet.

**Payments.** This node's sealed payment-lifecycle records (`x-mesh-settlement-v1`, one per
`payment.lifecycle.v1` event the plugin observed) are this node's own log entries: Pane A lists
them as `kind: "settlement_observation"`, and they are never exchanges. Panes B and C read them
only through their `exchange_id` join (`src/evidence_panes/settlement.rs`):

- each `panes/pane-c` row carries `settlement`, the payer-book summary of the settlement records
  its own exchange ids join (`api/sidecarTypes.ts` `PayerBook`), or `null` when none does. A
  free exchange, payments off, a request that failed before it was invoiced, and a paid request
  refused before any payment (for example a payment protocol version the seller doesn't speak;
  the host emits no event for it) all read `null`: never "unpaid". The book carries
  `who_paid: "this_node"` and `final_accounted_msat`, the payer's own recorded total copied from its
  record (`null` without one). An invoice with no settlement the wallet reported and no final
  amount is `outcome_not_reported`: the host emits no event when a payment's outcome is uncertain
  (a lost wallet reply) or it failed, so the book can't say which, or whether the exchange was
  interrupted;
- each `panes/pane-c` row of an exchange this node served for pay carries `provider_settlement`,
  its own book as the provider (`observed_by: "provider"`, `who_paid: "requester"`,
  `delivered_tokens`), joined by the exchange id the host names a provider's lifecycle events
  with; `null` otherwise. The list carries `settlement_provider_unjoined`, provider exchange ids no
  row carries;
- each book's `entries` copy the record's own values, and on a settlement the wallet's
  `credited_msat` and `fee_msat` when the host passed them, on `delivered` the `tokens`;
- the list carries `settlement_unjoined` (exchange ids with settlement records that no row
  carries) and `settlement_missing_exchange_id` (records naming no exchange id), so no record is
  dropped;
- when this node holds any settlement record, each `panes/pane-b` row carries `settlement`,
  per-peer counts (`PeerSettlementCounts`) of this node's own facts only, with
  `provider_book: "not_available"`: this node's records cannot see the provider's side, so
  nothing about it is sent.

Amounts are copied from each record and never added up. The panes carry no `payments` field:
whether the host has a payments provider is the host's to say, and the page reads a missing field
as "not known".

Status codes the page tells apart (`LedgerPage.tsx` `describePaneError`):
**404** means this plugin build serves no panes yet, and **503** means the
plugin's pane service isn't running. Any other failure gets the generic
message. A failed peer fetch renders as a transport error, never as a result.

## Where they are served

All of them are the plugin's own code: `src/evidence_routes.rs` declares the
GET bindings, `src/evidence_panes.rs` builds the panes from the plugin's
ledger directory (padding left out of every count), `src/ledger_fetch_bridge.rs` answers `mesh_ledger_fetch`, and
`src/mesh_evidence_bridge.rs` answers `mesh_evidence_request`.
`evidence_routes`'s manifest test pins the bindings. Fixture mode strips the
`http/` prefix and answers from recorded route captures.

## Console routes the page still reads

Two reads are the console's own data, not the plugin's:
`GET /api/status` (the owner link, and the peer list for the Peers tab's
"advertised" column) and `GET /api/models`. The page reads them with a
same-origin `fetch`, because the page is served from the console's origin.
They are public console routes, but a plugin page has no contract for them.
The console could withdraw or reshape them without notice, and they are
isolated in `features/network/api/` so the swap is one place. **Upstream ask
(host vocabulary):** a plugin-page read of node status (owner, peers) through
the host object, e.g. `host.node.status()`. The alternative is for the
plugin to serve peers from the mesh events it already subscribes to
(`peer_up`, `mesh_id_updated`).

Console navigation goes through `host.navigation.navigateTo`: `/chat`, and
the per-exchange `/logs?focusExchangeId=<id>`. The `focusExchangeId` search
parameter is not on upstream mesh-llm; there the link opens Logs unfocused.
