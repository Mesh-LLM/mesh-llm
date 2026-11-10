# Proposed `payments-lexe` preset

`payments-lexe` is an opt-in catalog set that installs the Lightning payments
engine and one Lexe wallet provider together. It stays outside `recommended`.
The set only groups installation; the engine resolves its `wallet.v1` binding
through the host at runtime.

This directory contains a reviewable catalog publication payload. Merging it
into GitHub does not activate the set: the live catalog is the Hugging Face
[`meshllm/plugin-catalog` dataset](https://huggingface.co/datasets/meshllm/plugin-catalog/tree/main).
Publication there and a host with
[plugin-set support (#2978)](https://github.com/Mesh-LLM/mesh-llm/pull/2978)
are required before the set commands below work.

## Pinned members

| Plugin | Release | Purpose |
| --- | --- | --- |
| `lexe-wallet` | [v0.1.0](https://github.com/Mesh-LLM/lexe-wallet-plugin/releases/tag/v0.1.0) | Provides `wallet.v1` through Lexe |
| `pay-lightning` | [v0.1.2](https://github.com/Mesh-LLM/ln-payments-plugin/releases/tag/v0.1.2) | Payment admission, policy, usage settlement and `pay.ln.v1` |

The engine pin includes the
[cached-ticket budget-admission fix](https://github.com/Mesh-LLM/ln-payments-plugin/commit/ba121a1f02674dfd8527b7e328d0b1adb557879b).
[plugins.additions.jsonl](plugins.additions.jsonl) contains both complete plugin
entries, with release tags and per-platform archive SHA-256 pins.
[plugin-sets.additions.jsonl](plugin-sets.additions.jsonl) contains the single
`payments-lexe` set. Wallet-first ordering is for installation; install both
members before starting the node.

The published pair has assets for Apple Silicon macOS, Linux ARM64, Linux x64
and Windows x64. Lexe also publishes Intel macOS and Windows ARM64 assets, but
the engine does not. On those targets, a set install may skip the engine and
install only the wallet. A successful partial install is insufficient for
payments.

## Host requirements

Use a host containing the generic binding, exchange-hook and peer-block seams
from [#2654](https://github.com/Mesh-LLM/mesh-llm/pull/2654), plus #2978 for set
installation. The engine documents main commit
[`b63cea23c22f3f0f8813de2c699161662f57d41d`](https://github.com/Mesh-LLM/mesh-llm/commit/b63cea23c22f3f0f8813de2c699161662f57d41d)
as its minimum host revision for those payment seams.

Host compatibility metadata is deliberately absent from this proposal.
Compatible source builds and incompatible older releases share the `0.78.0`
version string, and the plugin packages declare no host version bounds. A
numeric minimum of `0.78.0` would therefore admit unsupported hosts. Before
publication, coordinate a compatible host release boundary with the maintainers
and add truthful machine-readable constraints when that boundary exists.
Missing metadata is unknown compatibility, not certification of older releases.

## Wallet profile requirements

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

## Install and check

After the host requirements and catalog publication are satisfied:

```sh
mesh-llm plugins sets show payments-lexe
mesh-llm plugins sets install payments-lexe
```

Set installation skips already-installed members. It does not update an older
engine to v0.1.2, enable a disabled plugin, or change existing settings, wallet
state or spending policy. Check installed versions and explicitly update an
older engine before using this preset's budget behavior.

Select one running `wallet.v1` provider. If another wallet such as NWC is
already present, explicitly bind `pay-lightning` to the intended provider;
multiple providers are not automatic fallbacks. See the engine's
[operator binding and configuration guide](https://github.com/Mesh-LLM/ln-payments-plugin/blob/983f0fda8de269bdc83d799d34c67242164cb028/crates/mesh-llm-pay-lightning/README.md#explicit-operator-binding-optional).

Installing packages or inspecting manifests does not open a wallet. Starting
`pay-lightning` opens and warms the bound wallet and can provision a **mainnet
wallet**: `allow_wallet_provisioning` defaults to `true`. Review the engine's
wallet binding, exchange-hook and peer-block grants before installing.

Default payment state lives under `~/.mesh-llm/payments`, including
`payment.json`, the ledger and wallet state. A different host `--config` file
alone does not isolate those files. For a separate profile, explicitly set the
engine's `config_path` and its payment `data_dir`; an explicit wallet binding
must also select a separate wallet directory. Keep existing funded state intact.

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
or funding the wallet does not authorize paid inference. Set installation
preserves existing policy; plugin startup reapplies the policy from
`payment.json`, creating that file with free-only defaults if absent. Operators
enable buying and choose limits separately. Persist the intended policy in
`payment.json`: a control-only policy update can be replaced at startup. The
daily budget accounts for settled spend; concurrent admitted requests can exceed
it, so it is not a hard spending cap.

## Publish the catalog additions

1. Start from a fresh checkout of the Hugging Face dataset. Review changes since
   this payload was prepared; do not replace newer entries with stale pins.
2. Merge the two records from `plugins.additions.jsonl` into the dataset's
   `plugins.jsonl` by unique plugin `name`. Preserve all other entries.
3. Merge the record from `plugin-sets.additions.jsonl` into
   `plugin-sets.jsonl` by unique set `name`, creating that file if needed.
   Preserve every existing set and leave `recommended` unchanged.
4. Resolve the host compatibility boundary above, validate both complete files
   using the intended host's catalog parser/resolver, then submit both files in
   one Hugging Face dataset PR for maintainer review.
5. After approval and publication, check `sets show` and both member pins on
   each supported target before advertising the preset as available.

The `.additions.jsonl` files are deltas, not replacements for the live catalog.
Do not upload either over the full dataset file. GitHub review of this payload
and Hugging Face publication are separate steps.

## Validation of this proposal

On 2026-11-09, all ten pinned release archives were downloaded and their
SHA-256 hashes matched the entries here. A temporary integration harness used
#2978's parser and resolver at
`66f13a4834dca83dd7f29a61ec8dd0ff2c8e82e7`: four tests passed for exact membership
and pins, all four pair targets, wallet-only platform resolution, and preservation
of an existing `recommended` set when merging additions. This checks catalog
shape and asset availability; it does not certify host protocol compatibility.
No archive was executed and no wallet was created or funded for this validation.
