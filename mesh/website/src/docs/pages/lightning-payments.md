---
title: Paid inference over Lightning
---

# Paid inference over Lightning

> **Note:**
> Paid inference and wallet plugins are in early preview release and use Bitcoin mainnet.
> Wallets can hold and send real funds, so start with small balances and protect
> recovery material and wallet credentials.

MeshLLM providers can advertise per-model prices, and clients can automatically
settle paid inference over the Lightning Network. MeshLLM includes the payment
protocol, ledger, spending policy, and recovery logic; the wallet itself runs as
an external `wallet.v1` plugin.

Install the latest MeshLLM release before enabling payments. This release uses
the token-metered `/mesh/payment/v3` flow. Paid inference currently supports
`POST /v1/chat/completions` and `POST /v1/completions`, with one completion per
request (`n=1`).

## How it works

Payments are added by a wallet plugin. With one installed, the same node can
**earn** by serving models to other peers and **pay** for models served by
others. Inference still travels over Mesh's authenticated QUIC connections;
only the money moves over Lightning, between the two nodes' wallets.

One paid request goes like this:

1. **Request.** The buyer picks a provider and sends the request with the
   model and the provider's advertised prices. The buyer's spending policy
   (for example its daily budget) is checked locally and never leaves the
   buyer. There is no separate quote round trip.
2. **Admission and deposit.** The provider replies with fixed terms. A buyer
   it has not settled with before must first pay a small deposit (1 sat by
   default). This proves the buyer can pay and is credited toward the bill.
   The provider waits for the deposit to settle before any work starts.
3. **Inference.** The provider forwards the request unchanged to its model and
   streams the response back. Token usage comes from the backend's own usage
   report, so any OpenAI-compatible engine works.
4. **Settlement.** When generation finishes, the provider bills prompt tokens
   at the input rate and completion tokens at the output rate, subtracts the
   deposit, and sends one final invoice for the remainder. The buyer
   recomputes the bill from the agreed terms and its wallet pays
   automatically. No human approval is needed.

What happens in the edge cases:

- **Warm peers skip the deposit.** After one settled request, later requests
  to the same running provider normally skip it. Restarting the provider
  clears this.
- **Failures cost nothing.** If the backend fails, the buyer cancels, or no
  usage report arrives, the request is abandoned: no bill, no invoice, no debt.
  A deposit already paid is kept.
- **Unpaid bills block.** A buyer with three unpaid final invoices is refused
  until it pays one.
- **Recovery never reruns.** After a disconnect, the buyer can fetch the
  outcome of a request by its ID. Inference is never run twice.

Prices and token counts are the provider's claims. Mesh does not prove that
inference was done correctly, and there are no refunds. See the
[Lightning payments specification](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/specs/lightning-payments.md)
for wire frames, timeouts, and wallet persistence.

## 1. Install a wallet plugin

The published Lexe preview provisions a self-custodial mainnet wallet when the
first wallet-backed operation opens it:

```bash
mesh-llm plugins install Mesh-LLM/lexe-wallet
```

Restart MeshLLM after installation.

Two other adapters are available as source-build development previews. They do
not yet publish the release assets required by `plugins install OWNER/REPO`:

| Plugin | Wallet backend | Setup |
|---|---|---|
| [`Mesh-LLM/ldk-server-wallet`](https://github.com/Mesh-LLM/ldk-server-wallet) | An external [LDK Server](https://github.com/lightningdevkit/ldk-server) node over gRPC | Build and install its local archive, then configure the server address, TLS certificate, and macaroon file |
| [`Mesh-LLM/nwc-wallet`](https://github.com/Mesh-LLM/nwc-wallet) | An external wallet over [Nostr Wallet Connect](https://nwc.dev/) | Build and install its local archive, then configure a protected file containing the NWC connection URI |

Follow each plugin repository's README for its local-archive command and
configuration. Installing a wallet does not authorize inference spending.

If you pass the global `--config PATH`, use the same path for the node and every
wallet command. If the node uses a nondefault management port, add `--port PORT`
after `wallet`.

## 2. Set provider pricing

Free serving is the default. To charge for a model, use a model ID advertised by
your node and set positive input and output rates:

```bash
mesh-llm wallet pricing "MODEL_ID" \
  --input-msat-per-million INPUT_RATE \
  --output-msat-per-million OUTPUT_RATE
```

Rates use the CLI's `msat` field per million tokens. For example,
`--input-msat-per-million 1000` means ₿1 per million input tokens. MeshLLM bills
the `prompt_tokens` and `completion_tokens` reported by the inference backend at
the rates frozen when the request is admitted.

Inspect current prices or return a model to free serving with:

```bash
mesh-llm wallet pricing
mesh-llm wallet pricing "MODEL_ID" --free
```

The provider must keep its selected wallet plugin available while serving so it
can create invoices and observe their settlement.

## 3. Fund a client wallet and authorize spending

Start the MeshLLM node or client first; wallet-backed commands such as
`fund-wallet` and `get-balance` require the running wallet plugin. In another
terminal, create a BOLT11 funding invoice:

```bash
mesh-llm wallet fund-wallet --amount-sats AMOUNT
```

Pay the returned JSON `bolt11` value from another Lightning wallet, then confirm
the funds arrived and enable automatic payments:

```bash
mesh-llm wallet get-balance
mesh-llm wallet policy --mode automatic --daily-budget-sats BUDGET
```

`AMOUNT` and `BUDGET` must be positive whole base-unit counts; for example,
`1000` means ₿1,000. A fresh profile remains `free-only` until the policy command
is run. Funding the wallet alone never authorizes inference spending. Inspect or
disable the saved policy with:

```bash
mesh-llm wallet policy
mesh-llm wallet policy --mode free-only
```

The daily budget gates new paid admissions using successful payments settled
since 00:00 UTC, including actual routing fees. Pending payments do not reserve
budget, and wallet balance is not checked before admission. It is not a hard
wallet ceiling: concurrent or already-admitted requests can take settled spend
above the budget, and insufficient funds fail when a payment is attempted.

With automatic policy enabled, routing prefers local inference, then eligible
paid providers, then free providers.

## 4. How payments work

For a buyer that has not yet settled a request with a provider, the provider
requires an anti-griefing deposit (₿1 by default) before starting inference.
The deposit proves the buyer can pay and is credited toward the final bill.

Once one of that buyer's requests settles, later requests to the same running
provider normally skip the deposit. The provider streams the response,
calculates the bill from backend-reported token usage after generation
completes, and the client's wallet automatically pays any remaining amount
under its saved spending policy.

## 5. Limitations

Payments in MeshLLM are still experimental, and the protocol may change rapidly.
This early implementation has several known limitations that are being actively
addressed:

- Payments settle with a single provider node; revenue sharing among nodes
  serving a split model is not yet supported.
- The payment protocol does not verify inference results or token counts, so
  buyers must trust providers for correct inference and honest usage reporting.
- Responses stream normally, but the remaining balance is settled in one final
  invoice after the backend response completes; incremental streaming payments
  are not yet supported.

See the [Lightning payments specification](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/specs/lightning-payments.md)
for settlement, recovery, fees, compatibility, and operator details.

