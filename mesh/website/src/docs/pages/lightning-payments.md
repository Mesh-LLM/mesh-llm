---
title: Paid inference over Lightning
---

# Paid inference over Lightning

> **Note:**
> Paid inference is a work-in-progress preview and uses Bitcoin mainnet.
> Wallets can hold and send real funds, so start with small balances and protect
> recovery material and wallet credentials.

The MeshLLM node itself is payment-agnostic. Paid inference is added by two
plugins: the [`pay-lightning`](https://github.com/Mesh-LLM/ln-payments-plugin)
payments plugin, which owns pricing, spending policy, the ledger and settlement,
and a wallet plugin that holds the funds. Providers can then advertise per-model
prices and clients settle automatically over the Lightning Network.

You need a MeshLLM build that includes the generic plugin seams (mesh-llm `main`
after #2654, or the first release after v0.78.0). Paid inference currently
supports `POST /v1/chat/completions` and `POST /v1/completions`.

## How it works

Payments are added by plugins. With `pay-lightning` and a wallet installed, the same node can
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
[ln-payments-plugin README](https://github.com/Mesh-LLM/ln-payments-plugin#readme)
for protocol, state and control details.

## 1. Install pay-lightning and a wallet plugin

Install both plugins, in either order, before starting the node. Pick one wallet:

```bash
mesh-llm plugins install Mesh-LLM/lexe-wallet-plugin Mesh-LLM/ln-payments-plugin
# or, for a Nostr Wallet Connect wallet:
mesh-llm plugins install Mesh-LLM/nwc-wallet-plugin Mesh-LLM/ln-payments-plugin
```

Installing prints the grants `pay-lightning` needs and applies them; no
`[[plugin]]` stanza is required. A wallet alone does not enable payments, and
`pay-lightning` without a running wallet refuses every wallet call. If two
wallet plugins are running, name one before payments will work. Follow the
wallet plugin's README for its own setup (the Lexe wallet provisions itself on
first use; NWC needs a connection URI).

## 2. Control payments with `plugins call`

MeshLLM has no `wallet` command. Everything goes through the plugin's `control`
operation on the node API (add `--port PORT` if it is not on 3131):

```bash
mesh-llm plugins call pay-lightning control --input '{"command":"balance"}'
```

Other commands include `transactions`, `fund`, `pending`, `policy`, `pricing`,
`set_pricing`, `blocked` and `unblock`.

## 3. Sell: price a model

Free serving is the default. Start `mesh-llm --model MODEL_ID`, then:

```bash
mesh-llm plugins call pay-lightning control --input \
  '{"command":"set_pricing","model":"MODEL_ID","value":{"input_msat_per_million":1000,"output_msat_per_million":1000}}'
```

Rates are millisatoshis per million tokens (1000 = ₿1 per million). A request for
a priced model from a buyer without spending enabled gets `402 Payment Required`.
Keep the wallet plugin running while serving so invoices can be created and
settled.

## 4. Buy: fund a wallet and allow spending

Start the client (for example `mesh-llm client --join TOKEN`), fund the wallet
with the `fund` command or from the wallet's own app, then allow spending:

```bash
mesh-llm plugins call pay-lightning control --input \
  '{"command":"policy","value":{"mode":"automatic","daily_budget_msat":600000}}'
```

A fresh install is `free_only` and buys nothing. Funding the wallet alone never
authorizes spending. The daily budget gates new paid requests; it is not a hard
wallet ceiling, since requests already admitted can settle above it.

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

See the [ln-payments-plugin README](https://github.com/Mesh-LLM/ln-payments-plugin#readme)
for current status and known gaps.
