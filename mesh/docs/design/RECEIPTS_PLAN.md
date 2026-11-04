# Receipts implementation plan

This plan shows the status of each milestone of
[RECEIPTS.md](RECEIPTS.md). Receipts stay off by default, and `skippy
serve` works with receipts off after every milestone.

The current build has receipt wire version 7 and Skippy ABI 0.1.81. The
native code is one patch,
`skippy/llama_cpp/patches/model_support/0017-skippy-add-receipt-recorder-and-replay.patch`.
The llama.cpp pin is in `skippy/llama_cpp/upstream.txt`.

A status is one of these:

- **Done:** the code is in this repository.
- **Partial:** a part is built. The last column says what is left.
- **Open:** not built.

The numbers for each item are in
[RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md). The operator guide is
[skippy/RECEIPTS.md](../../../skippy/docs/RECEIPTS.md).

## Milestone 0: end-to-end demo

Milestone 0 stored every matmul column. Milestone 1 replaced its native tap
and its leaf layout. Serving, the HTTP API and the auditor continued into
milestone 1.

| Item | Status | What is left |
|---|---|---|
| Native tap that reports every weight matmul with its input and output columns | Done, then replaced | None. The recorder of milestone 1 replaced it |
| `skippy-receipts` crate: wire shapes, BLAKE3 leaves, Merkle paths, recorder, challenge draw, GGUF row reads, row checks | Done | None |
| Serving flags and the receipts HTTP API on each stage | Done | None |
| `skippy audit-receipt` with a local GGUF | Done | None |

## Milestone 1: replay instead of storage

| Item | Status | What is left |
|---|---|---|
| Native recorder: reports each ubatch with its geometry, the committed tensors of every layer, and the KV-cache operations | Done | None |
| Native replay of one sequence in a scratch context, with the recorded KV cells | Done | The creation of the scratch context runs with the runtime lock held |
| Receipt tensors from graph structure, with no architecture or tensor names ([RECEIPTS_GLUE.md](RECEIPTS_GLUE.md)) | Done | None |
| Leaves: `2 · layers + 1` per position. The residual leaf of layer 0 binds the token | Done | None |
| Recorder keeps leaf hashes and a replay log, not activations | Done | None |
| A replay stops after the last picked layer | Done | None |
| Kept openings (128 MiB per stage), a limit of challenges per receipt, and a scratch context that is freed after 60 seconds idle | Done | None |
| Challenge of four layers and four positions each. Every stage opens its part in one replay | Done | None |
| Bindings across layers: tokens, input of layer 0, stage boundaries, layer coverage | Done | None |
| Glue checks with libllama on the CPU | Done | Some errors that only the graph finds are reported as "could not run here" |
| Throughput: hash on the device and copy digests | Partial | Done for CUDA. Other backends copy the values |
| Serving budget: decode at most 3% slower, prefill at most 10% slower | Done on CUDA | On 2026-11-10, decode is about 1.5% slower (200.3 tok/s on, 203.4 tok/s off) and prefill about 3% slower, over 14 rounds on one RTX 5090. The decode rate depends on the idle states of the processor ([RECEIPTS_MEASUREMENTS.md](RECEIPTS_MEASUREMENTS.md#serving-cost-on-the-final-build)). Other backends and more than one lane are not measured on this build |
| Replay only the turn, from copies of the KV cells that it read | Done | None |
| Replay only the ubatches that hold a picked position | Done | None |
| Pin the cells of a bound turn for the challenge window | Done | A full cache writes pinned cells, and the turn is then unopenable |
| Prefix cache: each stage commits a prefix root, and a turn from a cached prefix opens | Done | A hybrid stage gives no receipt for such a turn. A stage with a quantized KV cache gives no receipt at all |
| Prefix check: a challenge picks two positions of the cached prefix, and the stage opens each by a replay of the ubatch that wrote it | Done | Not run on a real model. A turn whose prefix the stage cannot replay gets no receipt. A stage after the first keeps at most 1 GiB of input rows for these replays |
| Exact replay of cached generated tokens | Done | It is off while receipts are on |
| Native MTP: the stage issues receipts, and the receipt ends at the last token that the turn fed | Done | The receipt covers the trunk, not the MTP head |
| An opening of a turn with a context of about 32,000 tokens finishes in the 10-second window | Done | None |
| Replay gives the committed leaves bit for bit on CUDA and CPU through `skippy serve` | Done | Not run on the current build for ARM CPU, Vulkan and Metal |

## Milestone 2: protocol on the response

| Item | Status | What is left |
|---|---|---|
| Check the output head on the final residual | Done | None |
| Check the sampled token against the sampler | Partial | A greedy token must be the best logit. A sampled token must be one that the stated filters keep, and the one that the `indexed-v1` draw picks, within the draw band (0.01% of the logit root mean square). Penalties, grammars and other samplers get no judgement. A dynamic temperature and `ignore_eos` get no replay of the draw |
| Judge the tokens of a turn with a reasoning budget | Partial | The client replays the sampler of the budget: a forced token must be the forced one, and every other token is judged as usual. The tags are the statement of the node, not derived from the chat template. A lazy grammar gets no judgement |
| Compare the sampler statement with the request | Done | The client cannot know the default of a parameter that it did not set |
| Compare the response with the sampled tokens exactly. An empty response fails | Done | The prompt check is a containment check. A response that the node cut at the token limit can leave out an unfinished character or the start of a marker |
| Receipt in the final response body and in the last stream chunk | Done | None |
| Sign the turn root with the node key | Done | None |
| Bind the client key, the sampler and seed, the runtime, the backend name, the ubatch boundaries and the prefix root into the turn root | Done | The root does not name a rounding profile or an LFS hash |
| Chain the turn roots | Partial | The node binds the parent that the client names. `turn::verify_chain` exists. No shipped client keeps a chain |
| Draw the challenge with a VRF. The node checks the proof before it opens | Done | None |
| Later stages check the client proof too | Done | A later stage holds a turn to the first root with a correct proof. This needs stage authentication |
| Delegate the receipts VRF key from the node key | Open | The client makes a new key per request |
| Draw the checked rows from the VRF of the opening | Done | None |
| The signature of an opening covers the challenge. The signature of rows covers the blocks | Done | None |
| Check rows with the profiles of the backend that the receipt names | Done | The auditor takes the best profile of the backend family |
| Count the sampled tokens in the tie band, and flag a turn with too many | Done | The rate is not calibrated. More than one tie in four positions is flagged |
| Commit and open over the binary stage connection | Done | None |
| Receipts on the stages that Mesh starts | Done | A stage on another peer gives no acknowledgement that it records |

## Milestone 3: host runtime verifier and payments

| Item | Status | What is left |
|---|---|---|
| `--claimed-model repo@revision/file`: read the header, the rows and the small tensors from Hugging Face by range | Done | The bytes are pinned by the commit. No LFS hash is checked |
| Limits on every read of a claimed GGUF, and a strict header parser | Done | None |
| Split GGUF named by its first shard | Done | None |
| Verifier in the Mesh host runtime, on one worker thread that the node owns | Done | No automated test sends a challenge over a real mesh tunnel |
| Mesh audit against the file that the model name of the request stands for, not the file that the receipt names | Done | A bare runtime name, an alias, `local-gguf/sha256-...` and `owner/repo` without a selector cannot be audited. A `repo:selector` request that the node rewrote to the bare runtime name of a peer is usually not auditable |
| Bounds on every answer of a peer, and escaped peer text in logs and block reasons | Done | None |
| Output head check with a secret head key | Done | None |
| Loose attention check with partials and drawn blocks | Done | Softmax attention only. A layer that the check cannot follow is reported as not checked |
| MoE (position, expert) pairs, and the router on every row | Done | None |
| Rounding profiles for K-quants and the common types | Partial | Done for CPU and CUDA. Vulkan, Metal and ROCm have none. q4_1, q5_1 and the i-quants other than iq4_nl and iq4_xs have none |
| CUDA K-quant profile for the block sums of the current llama.cpp pin | Done | The profiles must be measured again on real hardware after each change of the pin. CI has no GPU |
| Attention combine tolerance that increases with the square root of the context | Done | The combination check is looser at long contexts. Backends other than CUDA are not measured at long contexts |
| Audit 10% of turns with a 10-second challenge window | Done | None |
| Four outcomes of an audit in Skippy (pass, fail, no audit, no answer) and four verdicts in Mesh (passed, failed, strike, no audit) | Done | None |
| A receipt check on every watched turn of a peer that advertises receipts | Done | None |
| Strike rule: three strikes in 24 hours block a peer for 7 days | Done | Strikes are in memory only. A node with `[receipts] offline` watches no turn and logs nothing per turn |
| Act on the verdict in payments | Open | Payments are in the `pay-lightning` plugin. No plugin seam carries the verdict. A failed audit blocks the peer and does not withhold an invoice |

## Milestone 4: consequences and trust tiers

| Item | Status | What is left |
|---|---|---|
| Block a peer locally for 7 days after a provable failure | Done | None |
| Keep the evidence of a failure in a bounded store that only the owner can read | Done | A node without an absolute state directory keeps none. `skippy recheck-receipt` does not read a kept refused receipt |
| `skippy recheck-receipt`: check the evidence again from public data | Done | None |
| Save the blocks from audits in a file of their own | Done | None |
| A request policy that requires `receipts` ([TRUST_TIERS.md](TRUST_TIERS.md)) | Done | None |
| Advertise the `receipts` flag only when the runtime issues receipts | Done | None |
| Advertise the `receipts` flag only when the backend of the runtime has a rounding-profile family (CPU, CUDA) | Done | None |
| Advertise the `receipts` flag only when the receipts name a Hugging Face file at a commit | Done | A model from a local path (`local-gguf/sha256-...`) does not qualify |
| A split with a stage on another node satisfies the tier | Open | The stage must acknowledge that it records, and its turns must be audited |
| `[receipts] offline`: no audit and no contact with Hugging Face | Done | None |

## Milestone 5: coverage

| Item | Status | What is left |
|---|---|---|
| Vulkan through `skippy serve` | Open | Not audited since 2026-11-10: a Vulkan stage issues no receipts. An honest node with Q4_K_M weights on an integrated GPU passed 2 of 16 audits, Q8_0 passed there with 1% of margin, and prefill was 2.4 to 2.5 times slower with receipts on. Vulkan needs a profile for the integer-dot kernels on K-quants. The native patch still forces f32 accumulation for a Vulkan recorder, which no stage starts now |
| Metal through `skippy serve` | Partial | Replay works. Prefill rows fit no rounding profile, so Metal stages get no audit |
| KV caches with a stream per sequence | Done | None |
| Hybrid memory: commit the recurrent state at the start of a turn, and replay from it | Done | A turn that continues a cached prefix gets no receipt |
| Recurrent check: run the token-mixing ops of the layer on the CPU over an opened region | Done | The state from an earlier turn is bound only by its digest |
| Quantized KV cache (q8_0), with the key rotation | Withdrawn | An honest stage fails the loose attention check at prompt positions, so a stage with a quantized KV cache issues no receipt. Model the error of the stored rows, then measure again |
| Attention check through gated outputs | Done | None |
| Dense-mask attention and f32 attention accumulation while a recorder is installed | Done | Support the per-sequence row lists natively, to remove the decode cost |
| Image tokens | Open | A turn with image tokens gets no receipt |
| Draft-model speculation | Open | Not tested |
| Open the lanes that shared ubatches in one replay | Open | Deferred. At a 10% audit rate, two lanes that shared ubatches are rarely challenged together |

## Later

| Item | Status | What is left |
|---|---|---|
| DSA openings (DeepSeek-V4, GLM-5.x) and the glue check of the indexer | Open | The opening format is not designed. Selection and replay were measured in the lab |
| ROCm profiles and replay | Open | Not measured |
| Phone verifier | Open | The embedded client has no peer blocks and no payments |
