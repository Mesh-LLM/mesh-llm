# Model-package generation-defaults migration inventory

Catalog snapshot: `meshllm/catalog` dataset revision `7b06a82063c4c672e3d7430b1c1cb29ac9bd400c` (2026-09-15).

This inventory covers every active `layer-package` mapping in that catalog revision. All 22 package manifests are schema v2 and predate the portable `generation.request_defaults` field, so their generation profiles are intentionally absent and the runtime fallback applies.

## Publication order

1. Release a runtime that accepts `generation.request_defaults` and Skippy ABI 0.1.55.
2. Research each exact source revision using the layer-package skill and create a reviewed defaults JSON file with immutable provenance.
3. Run `mesh-llm models package ... --generation-defaults <file> --dry-run` and review the printed profiles and citations.
4. Publish metadata-updated package revisions, certify each package, then pin the catalog to the new revisions.

Publishing the new manifest field before step 1 is unsafe: v0.76.1 uses `deny_unknown_fields` for the canonical generation object and would reject an otherwise valid package. Tensor artifacts do not need to change when only request defaults change.

## Active packages

| Package | Package revision | Source revision | ABI | Current validation | Defaults migration |
|---|---|---|---|---|---|
| `meshllm/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-layers` | `7c936b8c1dc370c615b306a9d3b78ea6ca6c10a5` | `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/DeepSeek-V4-Flash-UD-Q4_K_XL-layers` | `cef324ac0536d1868fef56a9ea1ac01037b6edb9` | `unsloth/DeepSeek-V4-Flash-GGUF@e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL-layers` | `e600064b1375e898dc758638228f76cab932b99c` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF@7b92b5b28818151e8669af2e45e88d6086f490dd` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/gemma-4-31B-it-qat-UD-Q4_K_XL-layers` | `04e1f6e043f7d766ccb860dd7b5834719c39f8d3` | `unsloth/gemma-4-31B-it-qat-GGUF@43cc1aeb31adf47ec06a854507ce552cd9862e6f` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/gemma-4-E4B-it-Q4_K_M-layers` | `b48f77afdbc97659eac9acc1d1d3466de2662364` | `unsloth/gemma-4-E4B-it-GGUF@bfc15c382204943c3a8fff0c750b94ae2364d7a3` | `0.1.54` | schema v2 rebuilt and inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers` | `a2b45d798d4e3a80703a68383a9bcecab798657d` | `unsloth/GLM-5.3-Flash-GGUF@621d456e93e926e4b52f85cff5f634358c1828f9` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/GLM-5.3-UD-Q4_K_XL-layers` | `cc7db4ed2c65b1785172b130192816ffabd2f2c5` | `unsloth/GLM-5.3-GGUF@346b3591c7f28d1a23716f97a065ecf12ec14771` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/inkling-UD-Q4_K_XL-layers` | `7c1f4bf9defc587af662fe23aaf32f8cbf2534bf` | `unsloth/inkling-GGUF@d3e9ffca48751dbe8b59dab5cfa364621257c682` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Kimi-K3-UD-Q4_K_XL-layers` | `79c7dbdd23a468e970195075b07a01a6bab605be` | `unsloth/Kimi-K3-GGUF@a0836360ce58dfec088d966a97f2ddc8a606279b` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Laguna-S-2.1-UD-Q4_K_XL-layers` | `b62056cee4480406d15286fd5a1979d42678ac19` | `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Llama-3.2-1B-Instruct-Q4_K_M-layers` | `7ad0d24934d3e0127e31bb48646ebcacb2028970` | `unsloth/Llama-3.2-1B-Instruct-GGUF@b69aef112e9f895e6f98d7ae0949f72ff09aa401` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Muse-Glimmer-30B-UD-Q4_K_XL-layers` | `645fd278c71c19c82a1ae07ea1ad8dc4d6498a70` | `unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL-layers` | `d36e5d8240628a1ddb07ba877713ffa646553d5b` | `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF@571758804835f56154718683f5c0e388b7d0fef9` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-layers` | `c27196ac33ac53b8295d188d101ad96e5591b2bb` | `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF@036038fb30334a2d56a146c6f0d4871ab5edccbb` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-layers` | `8550dda67869321972acdc33988a3a8e3faeb7b0` | `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-GGUF@360a2016f3b898925d7dc460d717852ba942948e` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/NVIDIA-Nemotron-3-Ultra-550B-A55B-UD-Q4_K_XL-layers` | `93a9f334e7cea4402e1875ae565992f6304cbb09` | `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF@2fb7d5b3f4eae7aedb18b4839b6a6300111e46f6` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL-layers` | `a8e296b2c0e10d069a3d007ed2c4d5328eef34f4` | `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF@f2d3fe3694501008786e81e5f20360cbf715496a` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Ornith-1.5-35B-Q4_K_M-layers` | `2501e17146ab3b0b1e32100ba960396acbeb028b` | `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL-layers` | `6c64523bea5b82217d812cd4bbf3164da4d9bd5e` | `unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF@a9b37aaac12b2bd0098783a443429543dd76a14d` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Qwen3.8-2.4T-A95B-UD-IQ4_XS-layers` | `c9e611eb6d66afbc64b561b111ecc590bc15832f` | `unsloth/Qwen3.8-2.4T-A95B-GGUF@567d3e6ac26c5474b18311e619c04350fb9a5556` | `0.1.54` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Qwen3.8-27B-UD-Q4_K_XL-layers` | `ea4e706fe98b2adb7de45e4c03986afa7807a26f` | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |
| `meshllm/Qwen3.8-Flash-Next-UD-IQ4_XS-layers` | `d80eb8d45d0e5fa3f76e9d76ee0e8f148b44fd56` | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | `0.1.53` | schema v2 inspected | Pending reviewed official source after compatible runtime release |

Coverage: **22/22 active catalog layer packages inventoried**. Package publication and certification remain an ordered post-release operation because of the compatibility gate above.

## Verification sources

- Catalog entries: exact `entries/**/*.json` files at the catalog revision above.
- Existing package manifests: Hugging Face package revisions recorded in this table; 21 were inspected during the 2026-09-15 full catalog audit.
- Gemma recovery: `meshllm/gemma-4-E4B-it-Q4_K_M-layers@b48f77afdbc97659eac9acc1d1d3466de2662364`, produced by HF Job `6aa8c4945527934177ee35df` and inspected as schema v2.
- Runtime compatibility: `crates/skippy-package-format/src/lib.rs` uses `deny_unknown_fields` on `Generation`, which is why publication follows runtime release.
