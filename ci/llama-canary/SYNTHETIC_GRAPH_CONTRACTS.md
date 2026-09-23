# Synthetic graph-contract coverage

`just skippy-native-tests cpu` (or `metal`) includes sparse synthetic GGUFs for
all 89 canary registry families (83 distinct architecture identifiers). Generation
and graph planning require no downloaded weights. These tests do not run numerical
inference or replace real-model, projector, audio, OCR or encoder-decoder certification.

The native CTest table in `src/skippy/tests/graph_contract_cases.cmake` is carried
by core patch `0024-test-skippy-complete-canary-graph-contracts.patch`.
`scripts/tests/test_synthetic_graph_registry.py`, discovered by `just ci-validate`,
compares every family, architecture, trunk depth, activation width, MTP count and
test mode against `family-certified.json`. Missing, duplicate or stale rows fail.
Adding an architecture requires an executable fixture; rejection modes are explicit.

## Contracts checked

For each admitted graph, the suite checks every interior cut and a three-stage
chain. Profiles cover single- and two-sequence decode, eight-token prefill with
all outputs, and prefill with only the last output, with token and raw-embedding
inputs. Granite Switch admits only the four token profiles because its adapter
router requires token IDs; the test asserts that exact profile set.

Parameter and state-effect unions must equal the whole-model plan. State accesses
must respect layer ownership; a feed-forward-only stage may have no state, while
the whole causal graph must retain reads and writes. Invalid ranges, impossible
output counts, missing middle stages and incompatible backend chains must reject.

Every declared MTP head must appear exactly once in the terminal stage execution
contract, with contiguous offsets and resident parameters. Earlier stages must
have no auxiliary heads. GLM4.5-Air and Nemotron have one head; MiMo2 has three,
with fused QKV tensors matching its MTP builder. Granite Switch's router cache
slot is explicitly not an MTP head.

T5 is the intentional rejection case: encoder-decoder cross-attention staging
is unsupported by the canary. Its synthetic GGUF loads, invalid ranges reject,
and every tested valid stage range must return the exact unsupported diagnostic
before attempting decoder-only graph construction. This catches the native
planner abort previously triggered by a two-sequence T5 request. The test does
not claim encoder-decoder execution coverage.

## Structural fixtures

Layer counts and activation widths follow the battery. Other dimensions remain
small and synthetic; the fixtures are not exact production metadata replicas.
The generator checks loaded trunk depth, output width and logical MTP count.

- Qwen3-VL dense and MoE preserve three deepstack layers and widened raw inputs.
- Qwen4 preserves four hyperconnections, compressed QSA indexers, PLE history and
  the registry boundary width (embedding width is one quarter of that boundary).
- Kimi preserves cross-layer residual checkpoints and hybrid attention. DeepSeek32
  and MiniMax M3 preserve sparse indexers; recurrent and convolution families
  use their native builders and state.
- Gemma3n shares KV after layer 20; cuts after 18 must reject. Gemma4's synthetic
  alternating local/global pattern shares KV after layer 24; cuts after 22 must
  reject. The latter is a synthetic producer-pair variant, not the battery's full
  production cut policy. All legal cuts must pass.
- Granite Switch includes two token-activated adapters and its router cache.
- Inkling includes MoE, relative attention and packed short-convolution state.
- Nomic-BERT and Jina-BERT use stateless graph checks. OCR and speech rows test
  their language graph contracts; real media encoding/decoding remains in the
  class-specific canary lanes.

The original Kimi/ARWKV activation-export and DeepSeek32 initial-input-leaf
regressions remain covered. Core patch `0025` rejects encoder-decoder plans cleanly
and excludes router cache slots from auxiliary MTP contracts. No skip-on-failure
or expected-crash tests are used.

## Registry dimensions

| Family | Architecture | Trunk layers | Activation width | MTP heads |
| --- | --- | ---: | ---: | ---: |
| qwen3-dense | qwen3 | 28 | 1024 | 0 |
| llama | llama | 16 | 2048 | 0 |
| glm47-flash | deepseek2 | 47 | 2048 | 0 |
| glm4 | chatglm | 40 | 4096 | 0 |
| glm45-air | glm4moe | 46 | 4096 | 1 |
| deepseek2 | deepseek2 | 27 | 2048 | 0 |
| gpt-oss | gpt-oss | 24 | 2880 | 0 |
| qwen3-moe | qwen3moe | 48 | 2048 | 0 |
| qwen2-moe | qwen2moe | 24 | 2048 | 0 |
| falcon-h1 | falcon-h1 | 24 | 2048 | 0 |
| jamba2 | jamba | 28 | 2560 | 0 |
| kimi-linear | kimi-linear | 27 | 2304 | 0 |
| qwen3-next | qwen3next | 48 | 2048 | 0 |
| qwen4exp | qwen4exp | 48 | 10240 | 0 |
| nemotron | nemotron_h_moe | 52 | 2688 | 1 |
| mamba | mamba | 24 | 768 | 0 |
| mamba2 | mamba | 64 | 4096 | 0 |
| rwkv6 | rwkv6 | 24 | 2048 | 0 |
| rwkv7 | rwkv7 | 32 | 2560 | 0 |
| granite-hybrid | granitehybrid | 32 | 768 | 0 |
| qwen35 | qwen35 | 24 | 1024 | 0 |
| granite-dense | granite | 40 | 2048 | 0 |
| lfm2 | lfm2 | 16 | 1024 | 0 |
| lfm2-vl | lfm2 | 16 | 1024 | 0 |
| laguna | laguna | 40 | 2048 | 0 |
| qwen2-vl | qwen2vl | 28 | 1536 | 0 |
| qwen3-vl | qwen3vl | 36 | 4096 | 0 |
| gemma2 | gemma2 | 26 | 2304 | 0 |
| phi3 | phi3 | 32 | 3072 | 0 |
| phi4 | llama | 40 | 5120 | 0 |
| falcon-dense | falcon | 32 | 4544 | 0 |
| internlm2 | internlm2 | 24 | 2048 | 0 |
| mistral-small | llama | 40 | 5120 | 0 |
| cohere2 | cohere2 | 32 | 4096 | 0 |
| smollm3 | smollm3 | 36 | 2048 | 0 |
| qwen2 | qwen2 | 24 | 896 | 0 |
| bloom | bloom | 24 | 1024 | 0 |
| gemma | gemma | 18 | 2048 | 0 |
| gemma3 | gemma3 | 26 | 1152 | 0 |
| phi2 | phi2 | 32 | 2560 | 0 |
| starcoder2 | starcoder2 | 30 | 3072 | 0 |
| stablelm | stablelm | 24 | 2048 | 0 |
| olmo2 | olmo2 | 16 | 2048 | 0 |
| exaone4 | exaone4 | 30 | 2048 | 0 |
| minicpm3 | minicpm3 | 62 | 2560 | 0 |
| nemotron-expansion | nemotron | 32 | 3072 | 0 |
| arcee | arcee | 36 | 2560 | 0 |
| mpt | mpt | 32 | 4096 | 0 |
| apertus | apertus | 32 | 4096 | 0 |
| glm4-expansion | glm4 | 40 | 4096 | 0 |
| afmoe | afmoe | 32 | 2048 | 0 |
| arwkv7 | arwkv7 | 28 | 3584 | 0 |
| bailingmoe3 | bailingmoe3 | 24 | 1536 | 0 |
| chameleon | chameleon | 32 | 4096 | 0 |
| deepseek32 | deepseek32 | 4 | 7168 | 0 |
| dots1 | dots1 | 27 | 2048 | 0 |
| ernie4-5 | ernie4_5 | 18 | 1024 | 0 |
| ernie4-5-moe | ernie4_5-moe | 28 | 2560 | 0 |
| gemma3n | gemma3n | 35 | 2048 | 0 |
| gemma4 | gemma4 | 42 | 2560 | 0 |
| gptneox | gptneox | 44 | 6144 | 0 |
| granite-swa | granite_swa | 28 | 1280 | 0 |
| granite-moe | granitemoe | 24 | 1024 | 0 |
| granite-switch | graniteswitch | 40 | 2560 | 0 |
| grovemoe | grovemoe | 48 | 2048 | 0 |
| hunyuan-dense | hunyuan-dense | 32 | 2048 | 0 |
| hy-v3 | hy_v3 | 48 | 2048 | 0 |
| jais | jais | 40 | 5120 | 0 |
| jais2 | jais2 | 32 | 3328 | 0 |
| kimi-k3 | kimi-k3 | 8 | 1024 | 0 |
| lfm2-moe | lfm2moe | 24 | 2048 | 0 |
| maincoder | maincoder | 32 | 1536 | 0 |
| mellum | mellum | 28 | 2304 | 0 |
| minicpm | minicpm | 32 | 4096 | 0 |
| qwen35moe | qwen35moe | 40 | 2048 | 0 |
| qwen3vlmoe | qwen3vlmoe | 48 | 2048 | 0 |
| mistral4 | mistral4 | 36 | 4096 | 0 |
| step35 | step35 | 45 | 4096 | 0 |
| mimo2 | mimo2 | 48 | 4096 | 3 |
| minimax-m3 | minimax-m3 | 60 | 6144 | 0 |
| spark2-5 | spark2_5 | 36 | 2560 | 0 |
| nomic-bert-embedding | nomic-bert | 12 | 768 | 0 |
| jina-bert-v2-rerank | jina-bert-v2 | 6 | 384 | 0 |
| t5-encoder-decoder | t5 | 8 | 512 | 0 |
| paddleocr | paddleocr | 18 | 1024 | 0 |
| qwen3tts | qwen3tts | 28 | 2048 | 0 |
| ultravox | llama | 16 | 2048 | 0 |
| inkling | inkling | 42 | 4096 | 0 |
| llama4 | llama4 | 48 | 5120 | 0 |
