# Model-package generation-defaults migration inventory

Catalog snapshot: `meshllm/catalog` dataset revision `7b06a82063c4c672e3d7430b1c1cb29ac9bd400c` (2026-09-15).

This inventory covers every active `layer-package` mapping in that catalog revision. All 22 package manifests are schema v2 and predate the portable `generation.request_defaults` field, so their published generation profiles remain absent until a compatible runtime is released. The official sources have now been reviewed: 19 validated JSON assets cover 20 mappings, and two mappings intentionally use the bounded Mesh fallback because their official repositories do not publish general deployment recommendations.

## Publication order

1. Release a runtime that accepts `generation.request_defaults` and Skippy ABI 0.1.55.
2. Use the reviewed file under `model-packages/generation-defaults/`, or the documented fallback decision for Kimi K3 and Inkling.
3. Run `mesh-llm models package ... --generation-defaults <file> --dry-run` and review the printed profiles and citations.
4. Publish metadata-updated package revisions, certify each package, then pin the catalog to the new revisions.

Publishing the new manifest field before step 1 is unsafe: v0.76.1 uses `deny_unknown_fields` for the canonical generation object and would reject an otherwise valid package. Tensor artifacts do not need to change when only request defaults change.

## Active packages

| Package | Package revision | Source revision | ABI | Current validation | Defaults migration |
|---|---|---|---|---|---|
| `meshllm/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-layers` | `7c936b8c1dc370c615b306a9d3b78ea6ca6c10a5` | `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768` | `0.1.53` | schema v2 inspected | Reviewed: `deepseek-v4-flash-0731.json` |
| `meshllm/DeepSeek-V4-Flash-UD-Q4_K_XL-layers` | `cef324ac0536d1868fef56a9ea1ac01037b6edb9` | `unsloth/DeepSeek-V4-Flash-GGUF@e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0` | `0.1.53` | schema v2 inspected | Reviewed: `deepseek-v4-flash.json` |
| `meshllm/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL-layers` | `e600064b1375e898dc758638228f76cab932b99c` | `unsloth/gemma-4-26B-A4B-it-qat-GGUF@7b92b5b28818151e8669af2e45e88d6086f490dd` | `0.1.54` | schema v2 inspected | Reviewed: `gemma-4-26b-a4b-it-qat.json` |
| `meshllm/gemma-4-31B-it-qat-UD-Q4_K_XL-layers` | `04e1f6e043f7d766ccb860dd7b5834719c39f8d3` | `unsloth/gemma-4-31B-it-qat-GGUF@43cc1aeb31adf47ec06a854507ce552cd9862e6f` | `0.1.54` | schema v2 inspected | Reviewed: `gemma-4-31b-it-qat.json` |
| `meshllm/gemma-4-E4B-it-Q4_K_M-layers` | `b48f77afdbc97659eac9acc1d1d3466de2662364` | `unsloth/gemma-4-E4B-it-GGUF@bfc15c382204943c3a8fff0c750b94ae2364d7a3` | `0.1.54` | schema v2 rebuilt and inspected | Reviewed: `gemma-4-e4b-it.json` |
| `meshllm/GLM-5.3-Flash-UD-Q4_K_XL-layers` | `a2b45d798d4e3a80703a68383a9bcecab798657d` | `unsloth/GLM-5.3-Flash-GGUF@621d456e93e926e4b52f85cff5f634358c1828f9` | `0.1.54` | schema v2 inspected | Reviewed: `glm-5.3-flash.json` |
| `meshllm/GLM-5.3-UD-Q4_K_XL-layers` | `cc7db4ed2c65b1785172b130192816ffabd2f2c5` | `unsloth/GLM-5.3-GGUF@346b3591c7f28d1a23716f97a065ecf12ec14771` | `0.1.54` | schema v2 inspected | Reviewed: `glm-5.3.json` |
| `meshllm/inkling-UD-Q4_K_XL-layers` | `7c1f4bf9defc587af662fe23aaf32f8cbf2534bf` | `unsloth/inkling-GGUF@d3e9ffca48751dbe8b59dab5cfa364621257c682` | `0.1.53` | schema v2 inspected | Reviewed: no general recommendation; bounded fallback |
| `meshllm/Kimi-K3-UD-Q4_K_XL-layers` | `79c7dbdd23a468e970195075b07a01a6bab605be` | `unsloth/Kimi-K3-GGUF@a0836360ce58dfec088d966a97f2ddc8a606279b` | `0.1.54` | schema v2 inspected | Reviewed: benchmark settings only; bounded fallback |
| `meshllm/Laguna-S-2.1-UD-Q4_K_XL-layers` | `b62056cee4480406d15286fd5a1979d42678ac19` | `unsloth/Laguna-S-2.1-GGUF@750f92f90cf54159c4d7a610cb7b3e74498e75c6` | `0.1.54` | schema v2 inspected | Reviewed: `laguna-s-2.1.json` |
| `meshllm/Llama-3.2-1B-Instruct-Q4_K_M-layers` | `7ad0d24934d3e0127e31bb48646ebcacb2028970` | `unsloth/Llama-3.2-1B-Instruct-GGUF@b69aef112e9f895e6f98d7ae0949f72ff09aa401` | `0.1.53` | schema v2 inspected | Reviewed: `llama-3.2-1b-instruct.json` |
| `meshllm/Muse-Glimmer-30B-UD-Q4_K_XL-layers` | `645fd278c71c19c82a1ae07ea1ad8dc4d6498a70` | `unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39` | `0.1.53` | schema v2 inspected | Reviewed: `muse-glimmer-30b.json` |
| `meshllm/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL-layers` | `d36e5d8240628a1ddb07ba877713ffa646553d5b` | `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF@571758804835f56154718683f5c0e388b7d0fef9` | `0.1.54` | schema v2 inspected | Reviewed: `nemotron-3-nano-omni-30b-a3b-reasoning.json` |
| `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-layers` | `c27196ac33ac53b8295d188d101ad96e5591b2bb` | `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF@036038fb30334a2d56a146c6f0d4871ab5edccbb` | `0.1.54` | schema v2 inspected | Reviewed: `nemotron-3-super-120b-a12b.json` |
| `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-layers` | `8550dda67869321972acdc33988a3a8e3faeb7b0` | `meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-GGUF@360a2016f3b898925d7dc460d717852ba942948e` | `0.1.54` | schema v2 inspected | Reviewed: `nemotron-3-super-120b-a12b.json` |
| `meshllm/NVIDIA-Nemotron-3-Ultra-550B-A55B-UD-Q4_K_XL-layers` | `93a9f334e7cea4402e1875ae565992f6304cbb09` | `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF@2fb7d5b3f4eae7aedb18b4839b6a6300111e46f6` | `0.1.54` | schema v2 inspected | Reviewed: `nemotron-3-ultra-550b-a55b.json` |
| `meshllm/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL-layers` | `a8e296b2c0e10d069a3d007ed2c4d5328eef34f4` | `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF@f2d3fe3694501008786e81e5f20360cbf715496a` | `0.1.54` | schema v2 inspected | Reviewed: `nemotron-3.5-lightning-30b-a3b.json` |
| `meshllm/Ornith-1.5-35B-Q4_K_M-layers` | `2501e17146ab3b0b1e32100ba960396acbeb028b` | `ornith-ai/Ornith-1.5-35B-A3B-GGUF@12393612fd4f730ff5aadc23e9b8f9648aa49ceb` | `0.1.53` | schema v2 inspected | Reviewed: `ornith-1.5-35b-a3b.json` |
| `meshllm/Qwen3-30B-A3B-Thinking-2507-UD-Q4_K_XL-layers` | `6c64523bea5b82217d812cd4bbf3164da4d9bd5e` | `unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF@a9b37aaac12b2bd0098783a443429543dd76a14d` | `0.1.53` | schema v2 inspected | Reviewed: `qwen3-30b-a3b-thinking-2507.json` |
| `meshllm/Qwen3.8-2.4T-A95B-UD-IQ4_XS-layers` | `c9e611eb6d66afbc64b561b111ecc590bc15832f` | `unsloth/Qwen3.8-2.4T-A95B-GGUF@567d3e6ac26c5474b18311e619c04350fb9a5556` | `0.1.54` | schema v2 inspected | Reviewed: `qwen3.8-2.4t-a95b.json` |
| `meshllm/Qwen3.8-27B-UD-Q4_K_XL-layers` | `ea4e706fe98b2adb7de45e4c03986afa7807a26f` | `unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502` | `0.1.53` | schema v2 inspected | Reviewed: `qwen3.8-27b.json` |
| `meshllm/Qwen3.8-Flash-Next-UD-IQ4_XS-layers` | `d80eb8d45d0e5fa3f76e9d76ee0e8f148b44fd56` | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | `0.1.53` | schema v2 inspected | Reviewed: `qwen3.8-flash-next.json` |

Coverage: **22/22 active catalog layer packages inventoried and reviewed**. Nineteen schema-validated defaults files cover 20 mappings; two reviewed mappings deliberately retain the bounded Mesh fallback. Package publication and certification remain an ordered post-release operation because of the compatibility gate above.

## Reviewed profile sources

| Official model revision | Applied profiles | Publisher input |
|---|---|---|
| [`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1f`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md) | `recommended`: temp 1.0, top-p 1.0 | `deepseek-v4-flash-0731.json` |
| [`deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md) | `recommended`: temp 1.0, top-p 1.0 | `deepseek-v4-flash.json` |
| [`google/gemma-4-26B-A4B-it-qat-q4_0-unquantized@f1e06dc52098`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized/blob/f1e06dc520982d9b9edd76859fdb7ab209449949/README.md) | `recommended`: temp 1.0, top-p 0.95, top-k 64 | `gemma-4-26b-a4b-it-qat.json` |
| [`google/gemma-4-31B-it-qat-q4_0-unquantized@1e4d8beecacb`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized/blob/1e4d8beecacb8b7590c1d8bedd7335f687bf311f/README.md) | `recommended`: temp 1.0, top-p 0.95, top-k 64 | `gemma-4-31b-it-qat.json` |
| [`google/gemma-4-E4B-it@ee0ef6023621`](https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/README.md) | `recommended`: temp 1.0, top-p 0.95, top-k 64 | `gemma-4-e4b-it.json` |
| [`zai-org/GLM-5.3-Flash@eb9eb208eb0d`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/generation_config.json) | `recommended`: temp 1.0, top-p 0.95 | `glm-5.3-flash.json` |
| [`zai-org/GLM-5.3@aca966e4e027`](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/generation_config.json) | `recommended`: temp 1.0, top-p 0.95 | `glm-5.3.json` |
| [`thinkingmachines/Inkling@828496eeae4c`](https://huggingface.co/thinkingmachines/Inkling/blob/828496eeae4c243ff1a22f7f28ff83694f2f7bc9/README.md) | No general deployment recommendation; bounded fallback | — |
| [`moonshotai/Kimi-K3@f831ab668142`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/README.md) | Benchmark-only settings; bounded fallback | — |
| [`poolside/Laguna-S-2.1@0f573140834b`](https://huggingface.co/poolside/Laguna-S-2.1/blob/0f573140834b11cfac0c2af97a101a7a69a13e22/generation_config.json) | `recommended`: temp 1.0, top-p 1.0, top-k 20, min-p 0 | `laguna-s-2.1.json` |
| [`meta-llama/Llama-3.2-1B-Instruct@9213176726f5`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/blob/9213176726f574b556790deb65791e0c5aa438b6/generation_config.json) | `recommended`: temp 0.6, top-p 0.9 | `llama-3.2-1b-instruct.json` |
| [`meta-models/Muse-Glimmer-30B@a4e59da52a7b`](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/README.md) | `recommended`: temp 1.0, top-p 0.95, top-k 64 | `muse-glimmer-30b.json` |
| [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16@e5e9932441de`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/blob/e5e9932441de940c9a62185c870ea5bcd4cd24e2/README.md) | `thinking`: max 20,480, budget 16,384, temp 0.6, top-p 0.95; `direct`: max 1,024, temp 0.2, top-k 1 | `nemotron-3-nano-omni-30b-a3b-reasoning.json` |
| [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16@2dc98e2afe4f`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/2dc98e2afe4face0e4ce40972a915c45368bd34a/README.md) | `recommended`: temp 1.0, top-p 0.95 | `nemotron-3-super-120b-a12b.json` |
| [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16@77df655d5e9f`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/77df655d5e9f8362164ed14dd8b48f8bce657498/generation_config.json) | `recommended`: temp 1.0, top-p 0.95 | `nemotron-3-ultra-550b-a55b.json` |
| [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16@a9904d24bcc1`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/a9904d24bcc1d289a1950fa9d2b978c47cf903b9/README.md) | `recommended`: temp 1.0, top-p 0.95 | `nemotron-3.5-lightning-30b-a3b.json` |
| [`ornith-ai/Ornith-1.5-35B-A3B@10fbf86fed7e`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/10fbf86fed7ecee4a061f8b499a618f46001cac1/README.md) | `thinking`: temp 0.6, top-p 0.95, top-k 20, budget auto | `ornith-1.5-35b-a3b.json` |
| [`Qwen/Qwen3-30B-A3B-Thinking-2507@144afc2f379b`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507/blob/144afc2f379b542fdd4e85a1fcd5e1f79112d95d/README.md) | `thinking`: max 32,768, temp 0.6, top-p 0.95, top-k 20, min-p 0, budget auto | `qwen3-30b-a3b-thinking-2507.json` |
| [`Qwen/Qwen3.8-2.4T-A95B@207bd685a7e3`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/README.md) | `thinking`: temp 1.0, top-p 0.95, top-k 20, min-p 0, presence 0, repeat 1, budget auto | `qwen3.8-2.4t-a95b.json` |
| [`Qwen/Qwen3.8-27B@1d4bf0f2ff60`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) | `thinking`: temp 1.0, top-p 0.95, top-k 20, min-p 0, presence 0, repeat 1, budget auto; `direct`: temp 0.7, top-p 0.8, top-k 20, min-p 0, presence 1.5, repeat 1 | `qwen3.8-27b.json` |
| [`Qwen/Qwen3.8-Flash-Next@de4b8e4d43b9`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md) | `thinking`: temp 1.0, top-p 0.95, top-k 20, min-p 0, presence 0, repeat 1, budget auto; `direct`: temp 0.7, top-p 0.8, top-k 20, min-p 0, presence 1.5, repeat 1 | `qwen3.8-flash-next.json` |


## Verification sources

- Catalog entries: exact `entries/**/*.json` files at the catalog revision above.
- Reviewed publisher inputs: `model-packages/generation-defaults/*.json`, each pinned to an official model repository revision.
- Existing package manifests: Hugging Face package revisions recorded in this table; 21 were inspected during the 2026-09-15 full catalog audit.
- Gemma recovery: `meshllm/gemma-4-E4B-it-Q4_K_M-layers@b48f77afdbc97659eac9acc1d1d3466de2662364`, produced by HF Job `6aa8c4945527934177ee35df` and inspected as schema v2.
- Runtime compatibility: `crates/skippy-package-format/src/lib.rs` uses `deny_unknown_fields` on `Generation`, which is why publication follows runtime release.
