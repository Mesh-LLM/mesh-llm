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
allowlist seller Mesh node IDs, and set payment, fee and UTC daily limits. Budget
reservations include the maximum fee and are conservatively retained after failure.

The plugin reads its public node ID once during initialization. It pins the wallet
ID, address and asset in a durable SQLite ledger; an account change fails closed.
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
attempt, inspect `status` and the wallet's attempts first, then explicitly run
`control --input '{"command":"resubmit","request_id":"<existing request id>"}'`.
This local operator action checks LOOKUP and can call PAY with the same persisted
obligation, payment ID, expiry and amount. The stored budget reservation requires
the same fee bound; it never creates a fresh invoice or a second payment identity.

Settlement is post-decode and gates subsequent work. A host crash before terminal
delivery can leave usage unknown; this demo deliberately fails closed and does not
pretend to provide exactly-once inference billing. Payment amounts are bounded by
buyer policy; backend-reported usage is a seller claim, not proof of computation.
