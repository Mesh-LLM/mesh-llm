# Proposed Lightning payment presets

Choose one opt-in catalog set to install the shared Lightning payments engine
and one wallet provider. All three stay outside `recommended`.

| Set | Ordered members | Wallet setup |
| --- | --- | --- |
| `payments-lexe` | `pay-lightning`, then `lexe-wallet` | Fresh/unpinned profile or separately validated legacy adoption |
| `payments-nwc` | `pay-lightning`, then `nwc-wallet` | Existing NWC wallet connection supplied through a private URI file |
| `payments-ldk-server` | `pay-lightning`, then `ldk-server-wallet` | Wallet release pending; requires an existing LDK Server and connection credentials |

Each set attempts missing eligible members in that order. Already-installed
members are skipped. Installation does not execute plugins, and this ordering
does not control runtime startup or create a dependency system. Install and
configure both members before starting the node; the engine resolves its
`wallet.v1` binding through the host at runtime.

This directory contains a reviewable catalog proposal. Merging it into GitHub
does not activate the sets: the live catalog is the Hugging Face
[`meshllm/plugin-catalog` dataset](https://huggingface.co/datasets/meshllm/plugin-catalog/tree/main).
Publication there and a host with
[plugin-set support (#2978)](https://github.com/Mesh-LLM/mesh-llm/pull/2978)
are required before the set commands below work.

## Members and release readiness

| Plugin | Release pin | Purpose |
| --- | --- | --- |
| `pay-lightning` | [v0.1.2](https://github.com/Mesh-LLM/ln-payments-plugin/releases/tag/v0.1.2) | Payment admission, policy, usage settlement and `pay.ln.v1` |
| `lexe-wallet` | [v0.1.0](https://github.com/Mesh-LLM/lexe-wallet-plugin/releases/tag/v0.1.0) | Provides `wallet.v1` through Lexe |
| `nwc-wallet` | [v0.1.0](https://github.com/Mesh-LLM/nwc-wallet-plugin/releases/tag/v0.1.0) | Provides `wallet.v1` through an existing NWC wallet |
| `ldk-server-wallet` | **Unreleased; no pin or archive hashes** | Provides `wallet.v1` through an existing LDK Server |

The engine pin includes the
[cached-ticket budget-admission fix](https://github.com/Mesh-LLM/ln-payments-plugin/commit/ba121a1f02674dfd8527b7e328d0b1adb557879b).
[plugins.additions.jsonl](plugins.additions.jsonl) contains four unique plugin
records: three released plugins with archive SHA-256 pins, and a discovery-only
LDK Server entry. [plugin-sets.additions.jsonl](plugin-sets.additions.jsonl)
contains the three sets; the shared engine appears once in the plugin catalog.

As checked on 2026-11-10, LDK Server has no published wallet-plugin release or
native archives. Its source package version is not a published release pin.
The entry therefore deliberately omits `version` and `sha256`; its wallet must
resolve as unavailable. Installing that set now could install only the engine.
**A verified LDK Server wallet release and archive pins are required before
publishing `payments-ldk-server` as installable.**

Lexe and NWC both publish wallet assets for six targets. Their complete pairs
with the pinned engine have assets for Apple Silicon macOS, Linux ARM64,
Linux x64 and Windows x64. Intel macOS and Windows ARM64 have wallet assets
but no engine asset. The LDK Server pair has no published target support yet.
Partial installation, including skipped unavailable members, can still report
success; verify both members and wallet setup before treating a pair as ready.
An earlier member's installation failure does not prevent later attempts.

## Host requirements

Use a host containing the generic binding, exchange-hook and peer-block seams
from [#2654](https://github.com/Mesh-LLM/mesh-llm/pull/2654), plus #2978 for set
installation. The engine documents main commit
[`b63cea23c22f3f0f8813de2c699161662f57d41d`](https://github.com/Mesh-LLM/mesh-llm/commit/b63cea23c22f3f0f8813de2c699161662f57d41d)
as its minimum host revision for those payment seams.

Host compatibility metadata is deliberately absent from this proposal.
Compatible source builds and incompatible older releases share the `0.78.0`
version string, and the pinned packages declare no host version bounds. A
numeric minimum of `0.78.0` would therefore admit unsupported hosts. Before
publication, coordinate a compatible host release boundary with the maintainers
and add truthful machine-readable constraints when that boundary exists.
Missing metadata is unknown compatibility, not certification of older releases.
Validate the eventual LDK Server release against that host as well.

## Wallet setup requirements

Presets group packages; they do not supply wallet credentials, adopt existing
wallet state, or switch the engine's pinned identity. Choose one wallet provider.
Installing another preset reuses an already-installed engine and can leave
multiple providers installed; it does not switch the engine to the new wallet.
Explicitly bind the intended provider when more than one is present. See the
engine's [operator binding and configuration guide](https://github.com/Mesh-LLM/ln-payments-plugin/blob/983f0fda8de269bdc83d799d34c67242164cb028/crates/mesh-llm-pay-lightning/README.md#explicit-operator-binding-optional).

### Lexe

For initial adoption, the pinned
[Lexe v0.1.0 release](https://github.com/Mesh-LLM/lexe-wallet-plugin/releases/tag/v0.1.0)
supports fresh or unpinned profiles. Existing `wallet-lexe` pins require a
separate identity-checked adoption procedure **before starting this pair**.
This preset does not implement that procedure. Do not delete wallet pins or
recovery seeds to bypass an identity mismatch.

Check the existing wallet directory before startup. Binding `lexe-wallet` can
select `payments/wallets/lexe-wallet` while legacy state remains under
`payments/wallets/wallet-lexe` or `payments/lexe`. An empty destination can
cause a new wallet to be provisioned before the engine checks the existing pin;
the pin then rejects the new identity and the funded wallet remains unavailable
to this pair. Establish a validated adoption path for existing state before
startup, and keep the funded wallet state, pins and recovery seeds intact.

Installing packages or inspecting manifests does not open a wallet. Starting
`pay-lightning` opens and warms the bound wallet and can provision a **mainnet
Lexe wallet**: `allow_wallet_provisioning` defaults to `true`. Review the engine's
wallet binding, exchange-hook and peer-block grants before installing.

### NWC

Configure an existing NWC wallet connection before starting the node. The
provider requires `--uri-file` pointing to an absolute private file containing
the connection URI; installation alone leaves that credential unconfigured.
Keep the file readable only by its owner (mode `0600` on Unix, or a private
Windows ACL). Grant `make_invoice`, `lookup_invoice`, `get_balance` and
`pay_invoice` (or NWC-321 `pay`); grant `list_transactions` for history and
NWC-321 `receive` for amountless funding invoices. The URI
authorizes wallet access; keep it out of the catalog, inline command arguments,
logs and source control.

```toml
[[plugin]]
name = "nwc-wallet"
args = ["--uri-file", "/absolute/private/nwc-uri"]
```

Use a mainnet wallet and grant `get_info` to check the reported network and
node identity. If that request is restricted or unimplemented, this release
falls back to assuming mainnet and using the URI public key as identity.
Confirm wallet-side spending limits: core NWC `pay_invoice` cannot enforce the
plugin's fee ceiling, and an NWC-321 wallet may ignore `max_fee`.

NWC connects to the existing wallet; it does not create or fund one. See the
[pinned NWC setup guide](https://github.com/Mesh-LLM/nwc-wallet-plugin/blob/28f77e5503dbbe120e05122d6fcb1454044b5793/README.md)
for supported connection permissions and wallet requirements.

### LDK Server

Complete the release and checksum publication prerequisite first. The provider
connects to an existing LDK Server over gRPC using a server address, TLS
certificate file and hex macaroon file. Configure these before node startup;
the preset does not install an LDK Server, create a wallet, supply credentials
or fund it. Follow the
[LDK Server wallet setup guide](https://github.com/Mesh-LLM/ldk-server-wallet-plugin/blob/0ab9de609f9f531c3f29278b3cf80723867e828c/README.md)
and verify the eventual release supports the required mainnet wallet operations.
Use `HOST:PORT` without a URL scheme, and a private file containing a reusable
hex v2 macaroon (mode `0600` on Unix). The needed permission labels are
`node:read`, `channels:read`, `invoices:create`, `payments:send`,
`payments:read` and `events:read`. Keep client/server clocks within 60 seconds.

```toml
[[plugin]]
name = "ldk-server-wallet"
args = [
  "--server", "127.0.0.1:3536",
  "--cert-file", "/absolute/private/tls.crt",
  "--macaroon-file", "/absolute/private/wallet.macaroon",
]
```

Treat the macaroon as a spending credential and keep it out of the catalog and
source control. Use the host's `wallet.v1` binding for the external engine;
the upstream guide's built-in `[payments]` wallet selector is historical.
The source guide reports no live LDK Server test, even on regtest. Release
publication also needs runtime qualification with the intended host and engine.
Preserve the provider's `issued-invoices.json` and the engine's identity pin
when reviewing an adoption or node change.

## Install and check

After a set's release requirements, host requirements and catalog publication
are satisfied, choose one set:

```sh
mesh-llm plugins sets show payments-lexe
mesh-llm plugins sets install payments-lexe
```

Substitute `payments-nwc` or `payments-ldk-server` after satisfying that wallet's
setup and release requirements. Avoid installing all three as wallet fallbacks.
Set installation skips already-installed members. It does not update an older
engine to v0.1.2, enable a disabled plugin, or change existing settings, wallet
state or spending policy. Check installed versions and explicitly update an
older engine before using these presets' budget behavior.

Default payment state lives under `~/.mesh-llm/payments`, including
`payment.json`, the ledger and wallet state. A different host `--config` file
alone does not isolate those files. For a separate profile, explicitly set the
engine's `config_path` and its payment `data_dir`; an explicit wallet binding
must also select a separate wallet directory. Keep existing funded state intact.
For NWC and LDK Server, separate local directories do not separate remote wallet
funds: two profiles pointing to the same connection or node use the same wallet.
Use a separate remote wallet or node when fund isolation is required.

Use `mesh-llm plugins list` to check the installed members. After starting the
node, inspect `GET /api/plugins` on the node API for each member's runtime
`status` and the engine's `manifest.openai_exchange_status`. In
`GET /api/plugins/providers`, check that the intended `wallet.v1` provider has
`available: true`; a listed provider with `available: false` cannot be bound.
A successful balance operation below verifies that the engine can use its
bound wallet. Inspect the wallet and policy through the control operation:

```sh
mesh-llm plugins call pay-lightning control --input '{"command":"balance"}'
mesh-llm plugins call pay-lightning control --input '{"command":"policy"}'
```

Add `--port <node-api-port>` when the node API uses a port other than 3131.
A new payment profile defaults to `free_only` with no daily budget; installing
or funding a wallet does not authorize paid inference. Set installation
preserves existing policy; plugin startup reapplies the policy from
`payment.json`, creating that file with free-only defaults if absent. Operators
enable buying and choose limits separately. Persist the intended policy in
`payment.json`: a control-only policy update can be replaced at startup. The
daily budget accounts for settled spend; concurrent admitted requests can exceed
it, so it is not a hard spending cap.

## Publish the catalog additions

1. Start from a fresh checkout of the Hugging Face dataset. Review changes since
   this payload was prepared; do not replace newer entries with stale pins.
2. Publish and verify an LDK Server wallet release, then fill its plugin entry
   with the exact release tag and per-target archive hashes. Resolve the host
   compatibility boundary above. Hold `payments-ldk-server` until these are ready;
   if publishing Lexe/NWC first, omit the LDK entry and set from that publication.
3. Merge `plugins.additions.jsonl` into the dataset's `plugins.jsonl` by unique
   plugin `name`. Preserve all other entries; keep one shared engine entry.
4. Merge `plugin-sets.additions.jsonl` into `plugin-sets.jsonl` by unique set
   `name`, creating that file if needed. Preserve every existing set and leave
   `recommended` unchanged.
5. Validate both complete files using the intended host's parser/resolver,
   including exact engine-first membership, release pins and target availability.
   Submit both files in one Hugging Face dataset PR for maintainer review.
6. After approval and publication, check `sets show`, both member pins and wallet
   setup on supported targets before advertising a preset as available.

The `.additions.jsonl` files are deltas, not replacements for the live catalog.
Do not upload either over the full dataset file. GitHub review of this proposal
and Hugging Face publication are separate steps.

## Validation of this proposal

The ten unchanged engine/Lexe archive pins were verified from downloaded bytes
on 2026-11-09. On 2026-11-10, all six NWC archives were downloaded, their hashes
matched both GitHub digests and publisher checksum sidecars, and their package
identities were inspected. No LDK Server archive exists to verify.

Six temporary integration tests passed against #2978's actual catalog parser,
resolver and installation planner at
`66f13a4834dca83dd7f29a61ec8dd0ff2c8e82e7`: exact three-set membership/order,
released-pair targets, wallet-only targets, unavailable unreleased LDK provider,
engine-first plans with existing/disabled shared-engine preservation, and
unchanged unrelated entries plus an existing `recommended` fixture.

These checks establish catalog shape and asset availability, not host protocol
or wallet runtime qualification. No archive was executed and no wallet was
opened, created or funded for this validation.
