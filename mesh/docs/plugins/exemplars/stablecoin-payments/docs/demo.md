# Mainnet OUSD inference: local buyer and Vast seller

This walkthrough uses real OUSD on **Tempo mainnet, chain 4217**. The native token
contract is `0x20c0000000000000000000006a37da5c996874be`. There is no testnet step,
faucet or automatic funding. Run these steps yourself after reviewing the code
and limits. Building and the offline smoke never generate or fund a real wallet.

The first request sends an earned admission payment, runs inference on the seller,
then settles the exact remaining usage bill. Later requests pay their full usage
cost. One outstanding request per peer is supported; run the demonstration serially.

## 1. Build the exact host and plugin source

Use Mesh from commit `8d03dea4555cbb1ac7cdfce7b9fe1a284f55fb76` on both machines.
Inside that checkout, follow its build instructions and use `just build` for
the complete local product. `just mesh` builds only the host; it requires a
separately verified compatible native bundle supplied through
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` when the product has no adjacent
`native-runtimes/`. The model-free on-demand buyer still initializes that runtime
at this revision, although it does not load a model. For the real model on Vast,
use Mesh's composed release build/bundle workflow
so the matching Skippy native runtime is installed. The existing complete
CUDA 12.8 product may be reused after verifying that exact source commit,
product/native manifests, compatible ABI and GPU runtime selection. This is
preferable on the 32 GB Vast disk: the agreed model is about 22.5 GB, so avoid
a redundant source rebuild or second model copy. An arbitrary released Mesh
binary can have a different plugin ABI. No Mesh core patch is part of this demo.

Build this workspace separately on each architecture:

```sh
just wdk-install # Node.js 22+; required when packaging the Ethereum provider too
just build
just check
just test
just release
bash scripts/package.sh target/release x86_64-unknown-linux-gnu 0.1.0
```

The last command is the Linux x86-64 packaging example. On Apple Silicon use the
appropriate `aarch64-apple-darwin` label and binaries built on that platform.
Do not copy a macOS plugin executable to Linux. Archives contain a root directory
with the plugin executable, `plugin.toml` and generated `plugin-manifest.json`;
the archive has a SHA-256 sidecar. No credentials, state or fixture binaries are
included. Mac binaries may need local ad-hoc signing according to Mesh's normal
macOS deployment instructions.

Optional local transport test, before touching money:

```sh
target/debug/host-smoke --mesh-bin /absolute/path/to/built/mesh-llm --output-dir /absolute/private/new-smoke
```

Supply the complete product binary with its adjacent runtime, or prepend
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR=/absolute/matching/native-runtimes` to that
command. The smoke inherits this bundle for both nodes, disables runtime
downloads, and needs no model weights.

It starts two actual Mesh nodes and the actual payment engine, checks an initial
admission of 3 fixture units and usage remainders of 2 and 5 (one buffered request
and one SSE request with terminal usage), then verifies cleared
debt. Its local wallet fabricates receipts and its backend returns canned text.
It proves host/plugin integration, not a real blockchain transfer or LLM quality.
The output directory is private because Mesh logs/config may contain invitations.

## 2. Install both production plugins

Install the two matching archives on each node **into the same isolated plugin
store that the node will use at runtime**. These are local archive installs;
no public repository release or catalog entry is assumed. Use a sibling store
so installation does not create the fresh wallet profile before step 3:

```sh
MESH_LLM_PLUGIN_DIR=/absolute/private/seller-installed-plugins mesh-llm plugins install --archive /absolute/pay-stablecoin-v0.1.0-TARGET.tar.gz --name pay-stablecoin --version 0.1.0
MESH_LLM_PLUGIN_DIR=/absolute/private/seller-installed-plugins mesh-llm plugins install --archive /absolute/wallet-ousd-tempo-v0.1.0-TARGET.tar.gz --name wallet-ousd-tempo --version 0.1.0
MESH_LLM_PLUGIN_DIR=/absolute/private/seller-installed-plugins mesh-llm plugins info pay-stablecoin
MESH_LLM_PLUGIN_DIR=/absolute/private/seller-installed-plugins mesh-llm plugins info wallet-ousd-tempo
```

On the buyer, run the same commands using its node-local paths and
`/absolute/private/buyer-installed-plugins`. Installation displays the declared
binding and lifecycle permissions. Installed manifests are required for strict
validation of custom plugin settings. Explicit owner configuration grants
authorize the binding and lifecycle access; they do not replace installation
metadata. **Omit `command` overrides** in the template: Mesh must launch each
plugin's installed executable for identity access to be authorized. A matching
binary in `target/release` is not an equivalent runtime path. Use a built wallet
binary for explicit initialization/CLI inspection only, outside the running host.

Do not install `demo-wallet-fixture` on either funded node. Run only the stablecoin
payment engine for this demonstration; leave Lightning payment plugins disabled.

## 3. Explicitly create local wallets

Choose a fresh private directory on each node. The following helper sets owner-only
permissions, writes a random password straight to a file and invokes the local
keystore initialization command. It prints public account metadata, never the
password or private key:

```sh
bash scripts/init-wallet.sh /absolute/private/buyer /absolute/bin/wallet-ousd-tempo
```

Run the equivalent command using `/absolute/private/seller` on Vast. The helper
refuses existing directories. Reuse/reopen an existing wallet instead of creating
a replacement to clear an error. Wallet initialization is local and sends no
transaction. The private directory contains `wallet/` and a separate `password`.

```sh
/absolute/bin/wallet-ousd-tempo --state-dir /absolute/private/buyer/wallet address
```

Before funding, back up the encrypted keystore and public wallet profile. After
use, also back up its transaction journal and payment-engine ledger. Keep the
password separately. Never commit these files, print their contents, send a key
to Vast through chat, or enable shell tracing around credential commands. The
local and Vast nodes should have different keys.

Transfer a small quantity of **mainnet OUSD** from an existing holder to the
buyer's displayed address. A 0.05 OUSD pilot balance comfortably exposes the demo's
chosen 0.05 OUSD daily spend ceiling; the ceiling is a limit, not a required balance.
The buyer needs enough for amount plus maximum fee reservations. A seller which
only receives and verifies payments needs no gas balance; fund it if it will send.
Wallet generation and receiving have no signup or KYC step. A fiat on-ramp can.

Before starting Mesh, check the buyer's finalized balance:

```sh
/absolute/bin/wallet-ousd-tempo --state-dir /absolute/private/buyer/wallet balance
```

## 4. Configure the Vast seller

Copy `demo/mesh.toml.example` to the seller's private directory as `config.toml`.
Replace `/ABS/...` with actual node-local paths. Set the wallet state path and
payment config path. Leave `password_file` absent for a receive-only seller.
Copy `demo/seller-payments.json.example` to `payments.json`.

The example prices mean **1 OUSD per million input tokens and 10 OUSD per million
output tokens**, with a 0.000010 OUSD earned admission minimum. They are explicit
demo prices, not a claim about model-market pricing. All JSON monetary values are
integer strings in millionths of OUSD. The first final charge is `max(C-A,0)`;
the seller retains `A` if it exceeds the actual first-request cost. No excess
continues to a future request.

The agreed live Vast model is **`Qwen3.6-27B-Q6_K`**, with context **16384** and
**one parallel lane**. Start it through Mesh's matching native CUDA runtime:

```sh
env -u MESH_LLM_EPHEMERAL_KEY MESH_LLM_CONFIG=/absolute/private/seller/config.toml MESH_LLM_PLUGIN_DIR=/absolute/private/seller-installed-plugins MESH_LLM_NODE_KEY_PATH=/absolute/private/seller/node.key mesh-llm serve --no-default-plugins --model Qwen3.6-27B-Q6_K --ctx-size 16384 --parallel 1 --port 9337 --console 3131 --log-format json
```

The isolated plugin store keeps the demo's default-plugin policy separate from
other local Mesh instances; Mesh resolves the installed executables by plugin
name. Keep it private: do not add `--publish`. Let this first run acquire/prepare the
model as needed; reuse its existing resolved artifact if present. The offline
fixture model remains `mesh-payment-fixture` and is never the live seller model.
From a second SSH session, inspect its concrete model ID:

```sh
curl --fail --silent http://127.0.0.1:9337/v1/models | jq -r '.data[].id'
```

Replace `REPLACE_WITH_EXACT_ADVERTISED_MODEL_ID` in `payments.json` with that exact
ID. Stop the foreground server cleanly and restart the same command/profile so
the payment engine reloads its prices. Do not issue a paid request before this
change. Use the same exact ID in the buyer request; avoid `auto` or `mesh` aliases.

Check initialization and save the invite without printing it:

```sh
umask 077
mesh-llm plugins call pay-stablecoin control --input '{"command":"status"}' --port 3131 > /absolute/private/seller/payment-status.json
jq -er '(.structuredContent // .structured_content // .).node | select(type == "string") | select(test("^[0-9a-fA-F]{64}$"))' /absolute/private/seller/payment-status.json > /absolute/private/seller/peer-id.txt
curl --fail --silent http://127.0.0.1:3131/api/status | jq -er '.token' > /absolute/private/seller/invite.token
```

The full 64-character public peer ID comes from the payment plugin's `status.node`,
not `/api/status.node_id`, which is a shortened display value. The filter above
accepts a structured MCP result or direct JSON result and rejects shortened IDs.
The full ID goes in the buyer allowlist. Securely copy only `invite.token`
and `peer-id.txt` to the buyer; keep the invite file owner-only. Do not copy the
seller wallet key or password. Management/API ports can stay loopback-only and
be inspected over SSH. Mesh's usual iroh relay/NAT traversal handles the peer path;
direct UDP exposure is an optional deployment/network optimization.

## 5. Configure and enable the local buyer

Copy `demo/mesh.toml.example` and `demo/buyer-payments.json.example` into the buyer
directory. Replace every path and the seller's full 64-character hex peer ID.
Uncomment the buyer's `password_file` setting so its local provider can send.

The template initially has `buying.enabled=false`. After checking the seller,
asset, prices, funded balance and limits, change it to `true`. This is the explicit
authorization for automatic admission and post-decode transfers to that seller.
Restart Mesh after changing JSON policy; it is loaded at plugin initialization.
The demo permits 0.01 OUSD per payment, 0.001 OUSD maximum network fee and a
0.05 OUSD UTC daily budget, with conservative fee reservations. No arbitrary peer
may spend the wallet merely by connecting.

For the buyer only, uncomment the template's runtime table:

```toml
[runtime]
mode = "on_demand"
```

Start the model-free buyer with `serve`, no `--model` and an explicit persistent
node-key path. Mesh's `client` command uses an ephemeral identity at this pinned
revision, which would change payment peer identity after a restart. The on-demand
server routes remote inference without loading a local model. Set `--max-vram 0`
and `--no-enumerate-host` to prevent a cached local model from being selected.
Its matching
native runtime must nevertheless be available beside the host or through
`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`. Ensure
`MESH_LLM_EPHEMERAL_KEY` is unset on both nodes; merely setting it to `0` still
enables ephemeral identity. Preserve the node key, wallet profile and ledger
together across restarts:

```sh
env -u MESH_LLM_EPHEMERAL_KEY MESH_LLM_CONFIG=/absolute/private/buyer/config.toml MESH_LLM_PLUGIN_DIR=/absolute/private/buyer-installed-plugins MESH_LLM_NODE_KEY_PATH=/absolute/private/buyer/node.key mesh-llm serve --no-default-plugins --max-vram 0 --no-enumerate-host --join-file /absolute/private/buyer/invite.token --port 9337 --console 3131 --log-format json
```

Check the buyer's plugin state and remote model list:

```sh
mesh-llm plugins call pay-stablecoin control --input '{"command":"status"}' --port 3131
mesh-llm plugins call pay-stablecoin control --input '{"command":"balance"}' --port 3131
curl --fail --silent http://127.0.0.1:9337/v1/models | jq -r '.data[].id'
```

Both nodes must have a running payment engine and ready wallet binding. Config
uses `[[plugin.operation_bindings]]` plus `[plugin.openai_exchange_grant]`; the
wallet's signing/password path is a provider setting, not peer input.
Keep all three OpenAI endpoint grants in the template: the engine must see and
reject unsupported completions/responses instead of letting paid peers bypass
the chat-only demonstration's gate through another endpoint.

## 6. Run one real request, then verify settlement

Create `request.json` with the exact model ID and a small finite output limit:

```json
{
  "model": "REPLACE_WITH_EXACT_ADVERTISED_MODEL_ID",
  "messages": [{"role": "user", "content": "Reply with a short greeting."}],
  "max_tokens": 8,
  "stream": false
}
```

Submit once through the local Mesh API. Do not add automatic HTTP retries:

```sh
curl --fail-with-body --silent --max-time 120 http://127.0.0.1:9337/v1/chat/completions -H 'content-type: application/json' --data-binary @request.json
mesh-llm plugins call pay-stablecoin control --input '{"command":"status"}' --port 3131
```

The response must contain nonempty generated text and authoritative token usage.
Settlement may finish asynchronously after the HTTP response. Poll `status` on
both nodes until `active` is empty. The first request must have a finalized
admission transfer and, when usage exceeds the minimum, a separate finalized
remainder transfer. Submit the second request only after the first is settled;
it must pay full usage and avoid a second admission charge.

On stopping a node cleanly, inspect its wallet without competing with the running
provider's exclusive state-directory lock:

```sh
/absolute/bin/wallet-ousd-tempo --state-dir /absolute/private/buyer/wallet attempts
/absolute/bin/wallet-ousd-tempo --state-dir /absolute/private/buyer/wallet balance
```

Each attempt has public transaction hash(es), an obligation reference, exact
amount and finalized delivery evidence. The provider verifies chain/token,
payer, recipient, amount and memo, matching both OUSD delivery and memo events in
a canonical finalized block. The seller independently verifies and consumes the
receipt once. Compare the public hashes in the [Tempo explorer](https://explore.tempo.xyz).
Record usage, admission amount, final charge, network fee and receipt hashes;
do not substitute a successful HTTP response for proof of payment.

## Failure and restart checks

- An insufficient balance or fee cap should prevent a new transfer. Keep the
  existing obligation and top up or inspect the error; do not replace its ID.
- After a timeout, inspect `status` and existing attempts. `retry` resends durable
  bills/proofs and performs reconciliation; it does not replay inference or sign
  another economic payment. Use documented explicit recovery only after checking
  whether the original intent exists. If the original call never produced an
  attempt, the explicit operator command uses the existing request identity:

  ```sh
  mesh-llm plugins call pay-stablecoin control --input '{"command":"resubmit","request_id":"EXISTING_REQUEST_ID"}' --port 3131
  ```

  It checks LOOKUP before submitting the same persisted obligation/payment ID.
- Restart both nodes with the same profiles and verify pending work reconciles.
  Never delete a journal, clone an active wallet or bypass a debt by making a new
  profile. Finalized receipts and admission standing must survive restart.
- A receiver-policy hold is **not paid**, even if the transaction reports success.
  The funds may already have left the sender. Halt and investigate/recover them;
  never pay again automatically.
- A missing terminal-usage event cannot be reconstructed from elapsed time. The
  demonstration fails closed and requires operator reconciliation.

The live mainnet checks above are operator acceptance steps. Documentation or
offline tests alone do not establish that they have passed.
