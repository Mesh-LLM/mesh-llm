# pay-stablecoin

This separately installed Mesh plugin owns direct stablecoin payments. It consumes
`wallet.stablecoin.v1` through a host-bound operation handle, negotiates over
`pay.stablecoin.v1`, and gates authenticated mesh requests using lifecycle hooks.
Core contains no pricing, token, address or settlement code.

The demonstration selects mainnet OUSD on Tempo. Install `pay-stablecoin` and
`wallet-ousd-tempo` separately. Only one payment engine is supported per demo node.
Set `[plugin.settings].config_path` to an absolute JSON path. Example:

```json
{
  "data_dir": "/absolute/path/payments",
  "wallet_binding": "wallet",
  "admission_fee": "1",
  "prices": {},
  "buying": {
    "enabled": false,
    "peers": [],
    "maximum_price": {"input_per_million": "100000", "output_per_million": "100000"},
    "max_payment": "1000",
    "max_fee": "1000",
    "daily_budget": "10000"
  }
}
```

All amounts are integer token units encoded as decimal strings; one unit is
0.000001 OUSD. Sellers add the exact advertised model ID under `prices`, with
the same two rate fields as `maximum_price`. Buyers explicitly enable spending,
allowlist seller Mesh node IDs, and set payment, fee and UTC daily limits. `max_fee`
uses the selected wallet's declared fee currency. If it is the payment asset
(including the legacy absent declaration), token-budget reservations include
that maximum fee. If fees use another asset, token-budget reservations and
`max_total` contain the payment amount alone; native fee units are never added
to token units. Zero fee-cap semantics belong to the concrete provider.
Reservations are conservatively retained after failure.

The plugin reads its public node ID once during initialization. It pins the wallet
ID, address, asset, method and effective fee currency in a durable SQLite ledger;
an account or currency change fails closed.
The ledger also pins the local Mesh node ID before recovering any payment. The
buyer must use a persistent Mesh identity: set `[runtime].mode = "on_demand"` in
its Mesh configuration and run `mesh-llm serve --config <path> --join-file <path>`
without model flags or configured models. Set `MESH_LLM_NODE_KEY_PATH` to a stable
absolute file and keep `MESH_LLM_EPHEMERAL_KEY` unset. Ordinary `mesh-llm client`
uses an ephemeral identity even with a node-key path and is unsuitable for this
durable payment demo. A changed node ID fails clearly; restore the original key
instead of resetting the ledger or abandoning outstanding bills. Keep the same
private/public mesh mode because crossing that boundary intentionally rotates keys.
Requests need a finite output limit of at most 4096 tokens and bodies below 64 KiB.
Paid inference supports `/v1/chat/completions` only. The plugin subscribes to all
three lifecycle endpoint families so priced remote `/v1/completions` and
`/v1/responses` requests are denied before payment or dispatch; unrelated free
and local requests keep their normal behavior.
Remote ingress is tied to its authenticated peer with the host's observation ID.
After Mesh resolves an alias, selecting a priced model must consume that peer's
ready admission ticket or recognize the same already admitted observation. An
unpriced ingress alias cannot bypass the later priced-model gate. Tracking is
bounded and cleared at terminal delivery; original endpoint and bounded request
data remain available if the selected-phase event omits them.
The demo serializes **one outstanding request per authenticated peer**. Independent
peer/model requests are not matched by mutable HTTP headers or wallet addresses.

A slow admission can outlast the short inference hook. It returns HTTP 402 with
payment pending, retains the original obligation and reconciles its receipt in
the background. It does not cancel the payment or start inference automatically.
An explicit retry for the same peer and resolved model may reattach that pending
exchange after the preceding hook has returned. Concurrent hooks and another
model cannot steal its ticket. Once finalized, that explicit retry consumes the
original admission and first-request minimum exactly once. A pending quote is
valid for five minutes; expired work remains blocked for reconciliation rather
than creating another charge. Restart detaches only unstarted buyer observations;
running or unpaid inference remains protected against replay.

The first admission earns a nonrefundable minimum `A`. The associated first usage
bill is `max(C-A,0)`; the total is `max(C,A)`. Later requests pay their full cost
`C`. No admission surplus carries to another request. There is no escrow, deposit
balance, credit channel or batching. Missing usage leaves the peer blocked for
operator reconciliation; token counts are never estimated from generated text.

`mesh-llm plugins call pay-stablecoin control --input '{"command":"status"}'`
lists pending work and wallet identity. `balance` reads the bound wallet; `retry`
retransmits durable bills and receipt proofs. Mainnet transfers require a funded,
unlocked wallet and enabled buyer policy. Development tests never fund a wallet.

Automatic recovery only looks up existing payment IDs and retransmits proofs; it
does not submit another transfer. If a wallet call failed before creating a signed
attempt, or died after journaling but before broadcast, inspect `status` and the wallet's attempts first, then explicitly run
`control --input '{"command":"resubmit","request_id":"<existing request id>"}'`.
This local operator action checks LOOKUP and calls idempotent PAY for an unknown
or nonterminal attempt, retaining the same obligation, payment ID, expiry and
amount. A journaled attempt rebroadcasts only its original signed data; finalized,
reverted, held or expired attempts are never submitted again. The stored budget reservation requires
the same fee bound; it never creates a fresh invoice or a second payment identity.

Settlement is post-decode and gates subsequent work. A host crash before terminal
delivery can leave usage unknown; this demo deliberately fails closed and does not
pretend to provide exactly-once inference billing. Payment amounts are bounded by
buyer policy; backend-reported usage is a seller claim, not proof of computation.
