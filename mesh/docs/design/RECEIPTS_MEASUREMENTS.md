# Receipts measurements

This appendix holds the measurements for [RECEIPTS.md](RECEIPTS.md). Each
number appears once, with its date, its hardware and its build. The other
receipts documents link here and do not repeat the numbers.

A number is valid only for the build that produced it. The table shows the
three groups of builds.

| Group | Receipt version | Skippy ABI | llama.cpp pin | Date |
|---|---|---|---|---|
| Final build | 7 | 0.1.82 | `a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5` | 2026-11-10 |
| Build of 2026-11-09 | 6 | 0.1.80 | `372dc5ca157276beb6c67b2e92477e8d662e964a` | 2026-11-09 |
| Earlier builds | 0 to 5 | Development numbers | The pin before the rebase | Before 2026-11-09. The day of each run was not recorded |

The final build is the only one that the code in this repository
reproduces. Use the other numbers as history. Do not use them as the cost
of the final build.

## Build of 2026-11-09

Unless a row says otherwise, the hardware is one RTX 5090 (CUDA) that
holds both stages of a two-stage Qwen3-4B Q8_0 split, with a release
build.

### Serving cost

The serving budget is decode at most 3% slower and prefill at most 10%
slower than with receipts off.

Decode with receipts on is inside the run-to-run noise of decode with
receipts off: 159.2 tok/s on, against 157.1 tok/s off. Prefill is about 4%
slower.

The runs of 2026-11-09 are below. The arms alternated in each run.

| Run | Build | Decode, receipts off | Decode, receipts on | Prefill, off | Prefill, on |
|---|---|---|---|---|---|
| 5 generations of 256 tokens and 5 prefills of 1,010 tokens per arm, session 1 | Before the integration of that day | 160.5 tok/s | 155.3 tok/s | About 14,000 tok/s | About 14,000 tok/s |
| The same, session 2 | Before the integration of that day | 160.2 tok/s | 152.5 tok/s | About 14,000 tok/s | About 14,000 tok/s |
| 10 generations of 384 tokens per arm, three alternating pairs | Integrated build | 162.9 tok/s | 150.7 tok/s (7.5% slower) | About 13,500 tok/s | About 13,000 tok/s (4% slower) |
| 10 generations of 384 tokens per arm, three alternating rounds | Final build | 157.1 tok/s | 159.2 tok/s | Not measured again | Not measured again |

- Prefill is inside the budget in every run.
- The decode number of the integrated build is not the expected cost. It
  held a regression, which the final build does not have. In the same
  session as the final build's row, the build before the regression
  decoded at 159.9 tok/s with receipts on, and the build with the
  regression at 148.5 tok/s.
- The regression came from hashing the leaves of a one-token ubatch on the
  recorder thread. Before, and again in the final build, the recorder
  hashes them on up to four short-lived threads. Those threads keep the
  cores of the stage out of the deepest idle state, and the decode
  threads then wake faster. The hashing on one thread was faster, and the
  decode thread did not wait for it. The decode rate thus depends on the
  idle states of the processor. It can be different on other hardware.
- The noise between the short runs is about 3%.
- A recorder turns on the dense-mask attention (see
  [RECEIPTS_GLUE.md](RECEIPTS_GLUE.md#attention-with-a-recorder)). A stage
  that serves more than one sequence from a unified cache pays for that
  in decode.

### Opening time

The opening time is the time from the challenge to the complete opening.

| Turn | Opening time |
|---|---|
| Short turn | 0.15 to 0.25 s |
| 15,000 fresh tokens, and 31,000 fresh tokens | 2.1 to 2.3 s |
| Turn that continues a cached prefix | 0.19 s |

All of them are inside the 10-second challenge window.

### Client audit time

These times are the wall time of `skippy audit-receipt` with a local GGUF.
They include the time to generate the turn.

| Turn | Audit time |
|---|---|
| Short turn | About 4.5 s |
| 15,000 tokens | 8.4 s |
| 31,000 tokens | 12 s |

Most of the time for a short turn is the native load of the challenged
layers. It is not network time.

### End-to-end outcomes

| Case | Result |
|---|---|
| Honest node | `AUDIT PASS`, exit code 0 |
| Node that serves another quantization than it claims | `AUDIT FAIL`, exit code 1, with provable failures |
| Claimed model that the auditor cannot read | `NO AUDIT`, exit code 2 |
| Challenge to a closed port | `NO ANSWER`, exit code 3 |
| Challenge that picks 36 layers | HTTP 400, `the challenge picks 36 layers; a draw picks at most 4` |
| Receipt id that the stage does not know | HTTP 404 |

### Detection on the build of 2026-11-09

| Split | Hardware | Node | Result |
|---|---|---|---|
| Two-stage Qwen3-4B Q8_0 | CPU | Serves Q8_0, claims Q8_0 | Pass |
| Two-stage Qwen3-4B | CPU | Serves Q4_K_M, claims Q8_0 | Fail. Every weight role fails 16 of 16 columns. The input of layer 0 and the output head fail. The recheck of the evidence fails the same weight roles |
| Two-stage Qwen3-4B | RTX 5090 | Serves Q4_K_M, claims Q8_0 | Fail. Every weight role fails 16 of 16 columns. The input of layer 0 and the output head fail |
| Two-stage Qwen3-4B Q4_K_M | RTX 5090 | Serves Q4_K_M, claims Q5_K_M | Fail on all seven weight roles |

### CUDA K-quant profile

llama.cpp #31247 changed the Q8_1 block sum to which the CUDA MMQ kernels
apply the minimum of a weight. The profile `q8_32_mul_dh_sums` models the
new rule (see [RECEIPTS.md](RECEIPTS.md#parameters)). RTX 5090, two-stage
Qwen3-4B Q4_K_M, honest node:

| Profile | Honest audits that pass | Failing columns | Largest error |
|---|---|---|---|
| Old rule, on the current pin | 2 of 4 | 5 to 13 per audit | Up to 1.9e-3 |
| New rule | 6 of 6 | None | 9.4e-4 on `attn_output`, which is the near-tie case. About 1e-7 on the other matrices |

### Hybrid models and quantized KV cache

| Case | Hardware | Result |
|---|---|---|
| Two-stage Qwen3.5-2B Q4_K_M split, prompts of 170 to 2,960 positions | RTX 5090 | 4 of 4 honest audits pass. The recurrent regions were checked. The largest row error is 2e-4 |
| Two-stage Qwen3.5-2B split, short and long turns | CPU | Pass. Recurrent regions of up to 256 positions check with error 0 |
| Qwen3-4B with a q8_0 KV cache | RTX 5090 | 2 of 2 honest audits passed on 2026-11-09. That sample was too small: see [Quantized KV cache](#quantized-kv-cache-2026-11-10). A stage with a quantized KV cache now issues no receipts |
| Qwen3-4B with a q8_0 KV cache | CPU | Passed on 2026-11-09, with attention combine errors of at most 1.9e-6. Not measured again. A stage with a quantized KV cache now issues no receipts on any backend |

### Attention accumulation on the CPU

Turns that continue a cached prefix of 1,500 positions, with a ubatch of 2
to 63 tokens.

| Accumulation | Result |
|---|---|
| f16, as the kernel did before | The kernel output is 2 to 7% from the exact value. The attention check of an honest node fails |
| f32, which a recorder asks for | Combine errors are at most 6.9e-3 |

### Attention combine error on long contexts

RTX 5090 (CUDA), two-stage Qwen3-4B Q8_0 split with an f16 KV cache. The
node was honest. The picks are 163 prefill picks and 157 decode picks,
from prompts of 3,000 to 31,000 positions.

| Positions that the head attends to | Prefill picks | Worst honest combine error |
|---|---|---|
| Up to 4,095 | 80 | 1.48e-2 |
| 4,096 to 8,191 | 38 | 1.94e-2 |
| 8,192 to 12,287 | 13 | 9.2e-3 |
| 12,288 to 16,383 | 11 | 1.32e-2 |
| 16,384 to 20,479 | 7 | 2.85e-2 |
| 20,480 to 24,575 | 9 | 8.9e-3 |
| 24,576 to 28,671 | 3 | 1.26e-2 |
| 28,672 to 32,767 | 2 | 3.90e-2 |

- The largest error divided by the square root of the context was
  2.65e-4, which is `0.54 · 2^-11`.
- Every decode pick was less than 5e-5.
- The cause is the CUDA prefill attention kernel, which accumulates in
  f16.
- With the fixed tolerance of 2e-2, one of the first long honest audits
  failed at 30,290 positions, with an error of 3.8e-2.
- With the tolerance that increases with the context, 12 of 12 honest
  audits of 31,000 positions pass.
- The cost is a looser combination check at long contexts: 6.3e-2 at 4,096
  positions and 1.8e-1 at 32,768. The check of the two drawn blocks stays
  exact.
- Other backends were not measured at long contexts.

## Final build, 2026-11-10

Unless a row says otherwise, the hardware is one RTX 5090 (CUDA) that
holds both stages of a two-stage Qwen3-4B Q8_0 split, with a release
build. This build checks the cached prefix, replays the sampling draw and
the reasoning budget, and issues no receipts for a quantized KV cache or
for Vulkan.

### Serving cost on the final build

The serving budget is decode at most 3% slower and prefill at most 10%
slower than with receipts off.

14 rounds, 7 with receipts on and 7 off, in alternating order. Each round
is 10 generations of 384 tokens and 5 prefills of about 1,000 tokens.

| | Receipts off | Receipts on | Cost |
|---|---|---|---|
| Decode, mean of the round means | 203.4 tok/s | 200.3 tok/s | 1.6% |
| Decode, mean of the round medians | 202.6 tok/s | 200.1 tok/s | 1.2% |
| Prefill, mean of the round means | 15,308 tok/s | 14,880 tok/s | 2.8% |
| Prefill, mean of the round medians | 15,576 tok/s | 15,263 tok/s | 2.0% |

- The round means with receipts off were between 193.9 and 210.1 tok/s.
  With receipts on they were between 198.2 and 201.7 tok/s. The cost is
  thus smaller than the noise of the arm without receipts.
- The decode rates are higher than on 2026-11-09 for both arms. The
  repository moved to a newer `main` between the two days.
- A request in these rounds asked for no receipt. The recorder ran for
  every request, which is the cost that a node pays for all its traffic.

### Opening time on the final build

The opening time is the time from the challenge to the complete opening.
An audit of a turn that continues a cached prefix now also opens two
positions of the prefix, each by a replay of the ubatch that wrote it.

| Turn | Hardware | Opening time |
|---|---|---|
| Short fresh turn | RTX 5090 | 0.2 to 0.3 s |
| 30,000 fresh tokens | RTX 5090 | 1.6 to 2.3 s |
| Turn that continues a cached prefix of 1,917 positions | RTX 5090 | 0.20 to 0.31 s |
| Turn that continues a cached prefix of 28,672 positions | RTX 5090 | 1.15 to 1.47 s |
| Short fresh turn | CPU | 0.7 to 1.1 s |
| Turn that continues a cached prefix of 1,917 positions | CPU | 5.0 to 7.5 s |

The CPU stage answers a cached turn inside the 10-second challenge window
at this length. It has less room than before: the same turn opened in
about 1 s when the prefix was not checked. A longer prefix on a CPU stage
was not measured.

### End-to-end runs on the final build

Every node in this table is honest unless the row says otherwise.

| Case | Hardware | Result |
|---|---|---|
| Greedy, sampled and cached-prefix turns, with rechecks of the evidence | RTX 5090 | 17 of 17 audits pass. 15 of 15 rechecks agree |
| Sampled turns over four sampler settings | RTX 5090 | 24 of 24 pass |
| Four lanes that share a prefix, 12 concurrent requests in three rounds | RTX 5090 | 12 of 12 pass. The turns started at positions 512, 1,916, 1,917 and 1,919 |
| Fresh turns and cached turns of about 30,000 tokens | RTX 5090 | 12 of 12 pass. The largest attention combine error was 0.048, which is 0.28 of the limit |
| Q4_K_M weights, greedy, sampled and cached | RTX 5090 | 5 of 5 pass |
| Hybrid Qwen3.5-9B Q8_0, greedy, sampled and a fresh turn of 1,900 tokens | RTX 5090 | 3 of 3 pass. A turn that continues a cached prefix gets no receipt |
| Requests with a reasoning budget of 0, 8 and 4,096 tokens, greedy and sampled | RTX 5090 and CPU | 10 of 10 pass with every token judged. The runtime enforces no budget for a chat without a grammar, so no token was forced |
| Greedy, sampled and cached-prefix turns | CPU | 9 of 9 pass. Rechecks agree |
| Node that serves Q4_K_M and claims Q8_0, fresh and cached turns | RTX 5090 and CPU | Every audit fails, with every weight role and the picked positions of the prefix. The recheck of the evidence fails the same roles |
| Stage with a q8_0 KV cache | RTX 5090 | The stage serves, prints a warning and issues no receipt. The audit reports no audit |

### Third-party recheck on the final build

A recheck now checks the logits that the node opened against every row of
the output matrix, and judges the token on them, as the audit does. Before,
it computed the logits without the rounding of the node. Those differed
from the logits of the node by up to 4% of the logit root mean square
(about 2% among the likely tokens, CPU, Q8_0), which is more than the tie
band of 1%. On the final build, over 168 head positions, the largest error
of the check of every row was 4.2e-7 on CPU and 1.6e-7 on CUDA. The limit
is 1e-5. The head part of a recheck takes 1.2 to 1.5 s, as before.

### Quantized KV cache, 2026-11-10

RTX 5090 (CUDA), two-stage Qwen3-4B Q8_0 split, with `cache_type_k` and
`cache_type_v` set to q8_0. The node was honest. The run was 290 turns
of 40 to 3,055 positions, and each audit drew 8 attention picks. The limit
of the combination check is the larger of 2e-2 and the square root of the
context divided by 1,024.

| Picks | Count | Worst honest combine error | Worst error divided by the limit | Picks past the limit |
|---|---|---|---|---|
| Prompt positions (prefill) | 1,142 | 1.135e-1 | 5.55 | 72 |
| Generated positions (decode) | 1,178 | 9.2e-3 | 0.46 | 0 |

The prefill picks by the length of the turn:

| Positions in the turn | Prefill picks | Worst honest combine error | Worst error divided by the limit | Picks past the limit |
|---|---|---|---|---|
| Fewer than 420 | 498 | 7.62e-2 | 3.81 | 29 |
| 420 to 1,499 | 415 | 1.135e-1 | 5.55 | 35 |
| 1,500 or more | 229 | 8.15e-2 | 1.57 | 8 |

- 65 of the 290 honest turns had at least one pick past the limit, about
  1 in 4.5. Each such audit is a provable failure, which blocks an honest
  Mesh peer for 7 days.
- The error is in the prompt positions only. The cause was not found. The
  largest error did not settle as the run went on, so no limit was chosen
  from it.
- With the default f16 cache on the same day and hardware, the worst
  honest error over 272 picks was 0.28 of the limit.
- Result: a stage with a quantized KV cache (q8_0 or q4_0) issues no
  receipts, and a client reports no audit for a receipt that names one.
  q4_0 was not measured. It is excluded because its rows are coarser than
  q8_0 rows.

### Vulkan on the final build

Measured on 2026-11-10. Two-stage Qwen3-4B split, context 4,096, all
layers on the device. The nodes were honest. The devices were an Intel
integrated GPU (Mesa ANV, integer-dot kernels) and an RTX 5060 Ti through
the NVIDIA Vulkan driver.

| Device | Weights | Honest audits that passed | Worst row error | Note |
|---|---|---|---|---|
| Intel integrated GPU | Q8_0 | 28 of 28 | 9.9e-4, against a limit of 1e-3 | 5 tokens passed only after a rounding repair. The margin is 1% |
| Intel integrated GPU | Q4_K_M | 2 of 16 | 5.8e-3 | 12% of the token columns were past 1e-3, all on q4_K matrices |
| Intel integrated GPU, integer dot off | Q4_K_M | 6 of 6 | 1e-5 | Profile `raw`, with `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` |
| RTX 5060 Ti | Q8_0 | 28 of 28 | Not recorded | |
| RTX 5060 Ti | Q4_K_M | 17 of 17 | 1.18e-3 | Three prefill `ffn_down` tokens were past 1e-3, two of them in one turn. A third in that role fails the turn |

- Each failed audit is a provable failure, which blocks an honest Mesh
  peer for 7 days.
- Cause of the Q4_K_M failures on the integrated GPU: the device uses the
  integer-dot kernels of Vulkan. The only candidate profile, `cpu_q8_0`,
  does not model how those kernels round K-quant weights.
- Rechecks agreed with the audits: 8 of 8 on each device with Q8_0
  weights.
- Sampled turns: all 52 draws on each device were exact, with no help
  from a band.
- Serving cost on the RTX 5060 Ti: decode was 72.1 tokens/s with receipts
  on and 71.5 with receipts off. Prefill was about 1,800 tokens/s with
  receipts on and 4,480 with receipts off, 2.4 to 2.5 times slower. The
  figure of an earlier build was 1.66 times.
- Opening time on the integrated GPU: 13.9 s for a fresh turn of 1,900
  tokens, 28.8 s for one of 3,000 tokens, and 8.1 s on average for a turn
  from a cached prefix, with a maximum of 11.2 s. The challenge window is
  10 s.
- Result: Vulkan has no rounding profiles. A Vulkan stage issues no
  receipts and does not record, a Mesh runtime on Vulkan does not
  advertise the `receipts` flag, and a client reports no audit for a
  receipt that names a Vulkan backend.
- To bring Vulkan back: a profile for the integer-dot kernels on K-quant
  weights, more margin for Q8_0 weights on integrated GPUs, and a prefill
  cost inside the serving budget.

### Metal, 2026-11-11

Measured on an Apple M6 with 24 GB, macOS 27. Two-stage Qwen3-4B split
with all layers on Metal. The shipped build issues no receipts on Metal.
The numbers below come from a local build that let a Metal stage record
and gave the client every profile as a candidate for a Metal backend.
That build is not in the repository.

The shipped build behaved as designed. Both stages printed the warning
and served, the response carried no receipt, and `skippy audit-receipt`
ended as no audit.

| Weights | Honest audits that passed | Worst row error at prompt positions | Worst row error at generated positions |
|---|---|---|---|
| Q8_0 | 54 of 54 | 8.4e-4, against a limit of 1e-3 | 2.0e-7 |
| Q6_K | 12 of 12 | 8.4e-4 | 2e-7 |
| Q4_K_M | 0 of 12 | 8.4e-3, with 466 of 666 q4_K columns past 1e-3 | Exact |
| Q4_0 file | No audit | Its `ffn_down` tensors are q4_1, which has no profile | |

- Each failed Q4_K_M audit is a provable failure, which blocks an honest
  Mesh peer for 7 days.
- Only prompt batches of more than about 16 tokens differ from the exact
  reference. The median error of a Q8_0 prompt column was 3.8e-4. The
  margin to the limit is 1.19 times.
- On this chip the batched matmul of Metal uses the tensor API of the
  platform. No profile models its rounding. With
  `GGML_METAL_TENSOR_DISABLE=1` the older kernel runs, and Q8_0 prompt
  columns then fit weights and activations that are each rounded to f16
  and summed in f32, with a largest error of 3.4e-6. Prefill then drops
  from 1,988 to 760 tokens/s. Q4_K columns stay near 2e-3 on the older
  kernel too.
- Replay: 54 of 54 openings succeeded, 9 of them for turns that continue
  a cached prefix, with prompts of 21 to 7,992 tokens. No replay diverged.
- Attention: the largest combine error was 4.8e-3 at prompt positions,
  which is 24% of the limit of 0.02.
- Output head and draws: 216 of 216 head positions passed, with a row
  error of at most 1.4e-7. 96 of 96 sampled draws were exact.
- Detection: a node that served Q4_0, Q6_K or Q4_K_M and claimed Q8_0
  failed every audit, with 16 of 16 columns past the limit. The smallest
  error of such a node was 3.1e-3, which is 3.7 times the largest honest
  error.
- Serving cost, three alternating rounds: decode was 33.6 tokens/s with a
  recorder and 34.0 without, 1.2% slower. Prefill was 1,764 against 1,993
  tokens/s, 11.5% slower, which is over the budget of 10%.
- Opening time: 0.3 to 0.5 s for a short turn, 3.2 to 5.2 s for a turn
  that continues a cached prefix of about 2,000 positions, and 4.7 to
  7.9 s for a fresh turn of 8,000 tokens. The challenge window is 10 s.
- The CPU runtime bundle for macOS uses the Accelerate BLAS, so its
  backend name is `BLAS`, which has no profile either. A CPU stage on
  macOS thus issues no receipts. A CPU bundle built without BLAS passed
  honest greedy, sampled and cached-prefix audits on this machine, and a
  node that claimed other weights failed.
- Result: Metal stays without receipts. To bring it back, the client
  needs a profile for the tensor API kernel, or a stage must run the
  older kernel at the prefill cost above. It also needs a profile for how
  Metal dequantizes K-quant weights, or K-quant weights on Metal must
  stay without an audit.

### Sampling draws, 2026-11-10

Every honest draw that an audit replayed picked the sampled token on the
logits that the node opened, with no help from a band: about 130 draws on
CPU, 120 on CUDA (RTX 5090, Q8_0 and Q4_K_M weights) and 160 on Vulkan
(before Vulkan lost its profiles). None passed only within the band, and
none failed. Result: the draw band is 0.01% of the logit root mean
square, 100 times narrower than the tie band.

### Two real Mesh nodes, 2026-11-10

Two `mesh-llm` nodes on one machine, both on CPU, with separate homes and
a direct join. Node A serves Qwen3-4B Q8_0 from a Hugging Face cache
layout with `[receipts] enabled = true`. Node B routes to A with
`[receipts] audit = true`. A test build of B audited every turn: the
shipped build draws about 1 turn in 10.

| Case | Result |
|---|---|
| Honest peer: greedy and sampled turns, streamed and not, a cached-prefix turn, a chat of three turns | 15 of 15 audits pass on the first run, and 8 of 8 on the final build with every token judged. No strike and no block |
| Shipped audit rate | 40 turns: 2 drawn and passed, 38 not drawn |
| `x-mesh-trust: receipts` | 200 when A issues receipts. 409 `trust_policy_unsatisfied` when A has them off, or serves a model from a local path |
| Peer that serves Q4_K_M bytes under the name of the Q8_0 file | The audit fails on the weight rows. B writes the evidence, blocks A for 7 days and stops routing to it. `skippy recheck-receipt` confirms the failure from Hugging Face alone in about 11 s |
| Peer that does not answer challenges | Strikes 1, 2 and 3, then a block. An answer after 4 s or 8 s is a pass |
| Model with no Hugging Face identity, and 30 turns in quick succession | No audit, and no penalty |
| Restart of B | The strikes are gone. The block stays |

Latency, with both nodes on the same CPU:

- An audit took 4.6 to 12.4 s for a short turn (median 7.3 s over 26
  audits), and 13.5 to 14.7 s for a prompt of 1,300 tokens. The audit runs
  after the response.
- A request of 48 tokens through B took 3.12 s on average with audits
  off, 3.00 s at the shipped rate, and 3.23 s when every turn was audited
  and the audits overlapped the requests. The differences are inside the
  noise of about 15%.

The two-node test found four defects that the unit tests with a scripted
transport did not. Each is corrected in this build: the tier accepted a
model that no audit covers, a thinking model got no token judgement, and
two sampler statements named a draw or a reasoning sampler that the
runtime did not run, which failed an honest node.

### Fail rule, 2026-11-11

On 2026-11-11 the scale of the row check changed from the norm of the
opened output to `max(|| y_ref[R] ||, || x || * || W[R] || / sqrt(K))`, and
one column past 1e-2 now fails its role. The client checked the stored
evidence of the final build again with each scale, on the CPU. That is 80
audits of honest Qwen3-4B nodes (Q8_0 and Q4_K_M, CPU and CUDA stages)
and 2 audits of a node that served other weights.

| Scale | Honest columns past 1e-3 | Largest honest `e` | Smallest `e` of the cheating node |
|---|---|---|---|
| Norm of the opened output (before) | 2 of 9,584 | 1.2e-3 | 1.3e-2 |
| Norm of the reference alone | 5 of 9,584 | 2.0e-3 | 1.2e-2 |
| Reference or weights and activation, the larger (after) | 2 of 9,584 | 1.7e-3 | 1.1e-2 |

- With the new scale, no honest role failed and no honest column was past
  the cap of 1e-2, which is 6 times the largest honest `e`. The 2 failing
  columns are in one audit of a CPU stage, at layer 0, in `attn_k` and
  `attn_v` of one position. Each role had 1 failing column, within its
  allowance.
- With the norm of the reference alone, 4 of 46 honest `attn_v` columns
  of layer 0 failed. The reference of such a column is small by
  cancellation. The second term of the scale removes these failures.
- All 224 columns of the cheating node were past the cap of 1e-2 with the
  new scale, so all 7 roles failed.
- The 320 checks of the output head had an `e` of at most 1.6e-7. The
  output head uses the same scale.
- 14 of the 80 honest audits now give no audit when they are checked
  again. Their attention picks have more than 2 blocks, and the client
  now draws the first block by its weight, so it asks for other blocks
  than the node signed. The row checks of these audits pass.
- These are Qwen3-4B audits only. No MoE model was checked again, so the
  margin of the expert roles under the new scale is not measured.

### Validation after the review fixes, 2026-11-11

An independent review on 2026-11-11 changed the fail rule, the attention
picks, the limits of a serving node and the Mesh policy. The runs below
checked that build again with real models. Unless a row says otherwise,
the hardware is one RTX 5090 that holds both stages of a two-stage
Qwen3-4B Q8_0 split. Small numbers in this section are written as
fractions: 1/1,000 is the limit of a column, and 1/100 is its cap.

Every node in this table is honest unless the row says otherwise.

| Case | Hardware | Result |
|---|---|---|
| Greedy turns, and sampled turns over four sampler settings | RTX 5090 | 36 of 36 audits pass |
| Turns that continue a cached prefix | RTX 5090 | 17 of 22 pass. 5 got no receipt, which was a defect of this build (see below) |
| Fresh turns of 30,700 tokens and cached turns of 28,200 positions, context 65,536 | RTX 5090 | 13 of 13 pass |
| Four lanes that share a prefix, 12 concurrent requests | RTX 5090 | 12 of 12 pass |
| Q4_K_M weights, greedy, sampled and cached | RTX 5090 | 10 of 10 pass |
| Hybrid Qwen3.5-9B Q8_0 | RTX 5090 | 5 of 5 pass. A turn that continues a cached prefix gets no receipt, as designed |
| MoE Granite 3B-A800M Q8_0 | RTX 5090 | 12 of 14 pass. 2 cached turns got no receipt, by the pin defect below |
| MoE Qwen3-30B-A3B Q4_K_M | RTX 5090 | 10 of 10 pass |
| Greedy, sampled and cached-prefix turns | CPU | 18 of 19 pass. 1 got no receipt, by the same defect |
| Node that serves Q4_K_M and claims Q8_0 | RTX 5090 and CPU | 11 of 11 audits fail, each with provable failures. 11 of 11 rechecks fail |
| Stage with a q8_0 KV cache | RTX 5090 | The stage serves, warns and issues no receipt. The audit reports no audit |

- Rechecks: 229 of 229 rechecks of honest evidence pass.
- Draws: 312 sampled draws were replayed, and each one picked the sampled
  token with no help from the band.
- Row errors on the RTX 5090 with Qwen3-4B: none of 22,362 honest columns
  was past 1/1,000. The largest was 1/1,160, in `attn_output`.
- Row errors of the MoE models, which were not measured under the new
  scale before: the expert roles of Granite reached 1/2,900 and its
  router 1/3,600. The expert roles of Qwen3-30B-A3B reached 1/7,100. One
  `attn_output` column of that model was at 1/980, which is inside the
  allowance of its role.
- Row errors on the CPU: one `attn_q` column was at 1/750, inside the
  allowance.
- Detection: all 1,444 columns of the node that claimed other weights
  were past 1/1,000, and 1,435 of them past 1/100.
- Attention: the largest combine error was 37% of its limit on the RTX
  5090 and 23% on the CPU. The CPU now sums attention in f32 only inside
  a receipt context. The largest CPU error did not grow by that change.
- Opening time on the RTX 5090: 0.11 to 0.22 s for a short turn, 1.9 to
  2.6 s for 30,700 fresh tokens, and 1.3 to 1.4 s for a cached prefix of
  28,200 positions.
- Opening time on the CPU: 0.5 to 1.1 s for a short turn, 3.3 to 8.2 s for
  a cached prefix of 1,870 positions, and 7.7 and 9.1 s for a fresh turn of
  1,900 tokens. The CPU was shared with two other nodes. The challenge
  window is 10 s.
- Serving cost, six alternating rounds: decode was 2.1% slower with
  receipts on by the means and 1.5% by the medians. Prefill was 4.6% and
  2.1% slower. On the CPU, decode was 15.2 tokens/s with receipts off and
  15.6 with receipts on.
- Concurrent audits: 16 of 16 parallel audits pass, and 30 of 30. Of 40
  parallel audits, 36 pass, 3 requests were refused by the full
  generation queue, and one replay could not allocate its context on the
  full device (see RECEIPTS.md, open limitation 29).

Two real Mesh nodes on one machine, both on the CPU, node B with
`audit_one_in = 1` unless the row says otherwise:

| Case | Result |
|---|---|
| Honest peer: greedy, sampled, streamed, cached-prefix and multi-turn chats, `/v1/messages`, a chunked body, a reasoning budget | 31 of 31 audits pass. No strike |
| Shipped audit rate, 40 turns | 8 drawn. 4 passed, and 4 were no audit because the one audit worker was busy or the challenge was late |
| `x-mesh-trust: receipts`, with a third node that serves the same model without receipts | 21 of 21 requests were served by the node with receipts, also when the third node held the cached prefix |
| Peer that serves Q4_K_M bytes under the name of the Q8_0 file | Provable failure on all 7 weight roles, evidence kept, block for 7 days. The recheck fails in 12 s |
| Peer that answers after 4 s and 8 s, and one that does not answer | The late answers pass. Three unanswered challenges block the peer |
| Peer on an older receipts build with another patch digest | No audit, and the peer leaves the tier for 24 hours. No strike |
| Stage with a q8_0 KV cache | No receipt, no audit, no strike. A tier request gets 409 |

- Latency of a 48-token request through B, 20 requests each: 3.43 s with
  audits off, 3.60 s at the shipped rate, and 3.91 s when every turn is
  audited.
- An audit took 4.8 to 13.3 s for a short turn (median 6.2 s) and 11.3 to
  20.1 s for a prompt of 1,000 tokens (median 14.3 s).

These runs found defects that the unit tests did not. Each is corrected
on the branch, and the case was run again:

| Defect | Before | After |
|---|---|---|
| The stages of a split each restored a cached prefix alone, so the recorder of one stage missed positions and the turn got no receipt | 14 of 22 cached-prefix turns pass | 20 of 22 pass, and 22 of 24 in a second run. The others hit a full KV cache (RECEIPTS.md, open limitation 30) |
| A turn past a share of the pinned cells got no receipt. That was every turn longer than half of the context | 13 of 13 turns of 30,700 tokens got no receipt at a context of 32,768 | 3 of 3 such turns get a receipt and pass. The new turn keeps its receipt, and the oldest pins give way |
| A pin ended 12 s after the commit, also while the stage replayed the turn for a challenge. A long replay on a CPU then lost its cells to the next request, and the stage signed "cannot open" | Six back-to-back turns with a shared prefix of 1,000 tokens: 3 strikes and a block | No strike in two runs. A turn stays pinned from its challenge to the end of its opening |
| The auditor counted the time a drawn turn waited for its one audit worker against the answer time of the peer | The peer had as little as 6.4 s to answer | Each challenge has 15 s from the time it is sent |
| A model served under a local name satisfied the `receipts` tier, and no audit could tie its name | 200 for a tier request, and every audit was no audit | 409. The serving node does not advertise receipts for such a name |

In Mesh a turn without a receipt is a strike, so the first three defects
could strike or block an honest node.

A last run on release binaries of the branch with every correction
repeated the cases above. On the RTX 5090, 12 of 12 greedy and sampled
audits pass, 3 of 3 audits of a node that claimed other weights fail,
and 2 of 2 Granite turns that continue a cached prefix of 2,891
positions at a context of 4,096 pass. On the CPU, 8 of 8 pass. On two
Mesh nodes, the back-to-back test gave no strike in two runs, 10 of 10
audits of an honest mix pass, the tier answers 200 by a Hugging Face
reference and 409 by a local file name, and the peer that served other
weights was blocked. `skippy audit-receipt --model` audits a Mesh node
directly.

### Reasoning budgets, 2026-11-12

Since 2026-11-12 the runtime enforces a reasoning budget for every chat
with a finite budget, not only for a chat with a grammar. The statement
of a turn then names the reasoning sampler, and the client replays it to
know which tokens were forced. These runs checked that on real models,
with reasoning turned on for the stage (`--reasoning enabled`, or
`enable_thinking` in the request). Two-stage Qwen3-4B Q8_0 split, honest
nodes, on runtime bundles of this build.

| Case | Hardware | Result |
|---|---|---|
| Budgets 0, 8, 64 and 4,096 and no budget, greedy and sampled, 256 generated tokens | RTX 5090 | 20 of 20 audits pass. The reasoning ran exactly 0, 8 and 64 tokens |
| Budget 8 with 12 generated tokens, so that the forced token is often checked | RTX 5090 | 30 of 30 pass. 12 audits checked the forced end tag, and each marked it as forced |
| Budgets 8 and 64, greedy and sampled | CPU | 10 of 10 pass. 1 audit checked the forced end tag |
| Two Mesh nodes, every turn audited: the default budget of the node (100), a budget of 8, and no limit | CPU | 32 of 32 audits pass. No strike |

- Every recheck of the evidence agreed with its audit.
- A turn with reasoning off states `reasoning` as `null`, and every token
  of it is judged by the usual rule.
- Reasoning is off by default in `skippy serve` and in Mesh. A request
  with a budget on such a node states the budget and no sampler, and no
  token is forced.

### New llama.cpp pin, 2026-11-12

On 2026-11-12 the pinned llama.cpp commit moved from 372dc5ca to
a4d4f2fd, and the Skippy ABI to 0.1.82. The rounding profiles model
kernels of one pin, so the honest and the cheating cases were run again.
Every number above this section was measured on the old pin. Two-stage
splits, honest nodes unless the row says otherwise.

| Case | Hardware | Result |
|---|---|---|
| Qwen3-4B Q8_0: greedy, sampled and cached-prefix turns | RTX 5090 | 25 of 25 audits pass, and 25 of 25 rechecks. No column of 3,242 past 1/1,000. Largest 1/1,534, in `attn_v` |
| Qwen3-4B Q4_K_M, Q5_K_M, Q6_K and IQ4_XS | RTX 5090 | 7 of 7 each. Largest errors 1/6,245, 1/1,542, 1/1,674 and 1/3,761 |
| Fresh turns of 28,300 tokens and turns on a cached prefix of 28,672 positions | RTX 5090 | 7 of 7 pass. Openings took 2.1 to 2.5 s and 1.4 to 1.7 s |
| MoE Granite 3B-A800M Q8_0 and Qwen3-30B-A3B Q4_K_M | RTX 5090 | 7 of 7 each |
| Hybrid Qwen3.5-9B Q8_0 | RTX 5090 | 6 of 6 fresh turns pass. A turn that continues a cached prefix gets no receipt |
| Reasoning budget of 8 tokens, greedy and sampled | RTX 5090 | 20 of 20 pass. 9 audits checked the forced end tag |
| Qwen3-4B Q8_0 and Q4_K_M | CPU | 9 of 9 each. Q8_0 had 2 of 1,114 columns past 1/1,000, at 1/670 and 1/777 in layer 0, inside the allowance |
| Node that serves Q4_K_M or Q6_K and claims Q8_0 | RTX 5090 and CPU | 10 of 10 audits fail, each with provable failures, and every checked column fails |

- Serving cost, six alternating rounds: decode was 199.8 tokens/s with
  receipts on and off. Prefill was 4.9% slower with receipts on.
- The largest attention combine error was 34% of its limit.
- The nearest substitution measured: a node that served Q6_K and claimed
  Q8_0 had an error of at least 1/369 on its best matrix. The largest
  honest error of a Q8_0 node on the same hardware was 1/1,534.

Weights in F16 failed honest audits on this pin, and the profiles
changed for it:

| Backend | Before | Cause | Now |
|---|---|---|---|
| CPU | 0 of 9 honest audits pass. The output head fails at every position | The new pin multiplies F16 weights by an f32 activation. The profile rounded the activation to f16 | The profile is `raw`. The 9 stored audits pass, and 6 of 6 new ones. Largest row error 1/1,100,000 |
| CUDA | 0 of 9 pass. 278 of 1,120 columns past 1/1,000 | Decode sums in half precision and prefill computes in half precision. No profile models that, on the old pin either | No profile. A stage whose matrices are F16 or BF16 issues no receipts on CUDA, and an audit of such a receipt is no audit |

A stage now reads the weight types of its own layers when it starts. It
issues no receipts when one of them has no profile on its backend, so a
Mesh node does not advertise receipts for such a model. A Mesh node also
does not advertise receipts for a hybrid model: a turn that continues a
cached prefix would get no receipt, which is a strike.

Not measured on the new pin: BF16 weights on the CPU, Gemma 4 E2B (the
split planner refuses it on this tree), Metal, and two Mesh nodes.

### Not measured on the final build

- The RTX 5060 Ti on CUDA.
- Throughput with more than one lane. Audits with four lanes pass.
- A CPU stage with a cached prefix longer than 1,900 positions.
- Two Mesh nodes on two machines, a split across nodes, and more than
  two nodes.
- Metal with the shipped build, beyond the check that it issues no
  receipts. See [Metal, 2026-11-11](#metal-2026-11-11).

Older numbers for some of these cases are in
[Earlier builds](#earlier-builds).

## Earlier builds

Every number below is from a build before the rebase onto the current
llama.cpp pin. The runs were made before 2026-11-09, and the documents did
not record the day of each run. Three things changed since then, and each
one can change a number:

- The challenge. The first builds opened one layer at 16 positions. The
  current build opens four layers at four positions each.
- The recorder. The first builds copied every committed value to the host.
  On CUDA, the current build hashes on the GPU and copies digests.
- The attention kernel. The current build uses the dense-mask attention
  and f32 accumulation while a recorder is installed.

### Detection

The lab harness hooks llama.cpp and compares backend outputs with a CPU
reference. It runs a 3,072-token prompt plus 16 decode tokens.

CUDA on an RTX 5090, with real kernels, no near-tie repair and no committed
activation blocks. Each column check is one matrix and one token, checked on
16 random rows. Each tensor has eight simulated row sets. `e` is the relative
error of the fail rule in [RECEIPTS.md](RECEIPTS.md#fail-rule).

| Run | Column checks | Median `e` | Share over 1e-3 | P(audit fails) |
|---|---|---|---|---|
| Honest, 6 quants of Qwen3-4B | 28k to 29k each | 1e-7 | 0 to 0.03% | 0 |
| Honest, Qwen3-8B Q4_K_M and Q8_0 | 29k each | 1e-7 | 0.003 to 0.05% | 0 |
| Honest, Qwen3.5-9B (hybrid) Q4_K_M and Q8_0 | 24k each | 1e-7 | 0 | 0 |
| Honest, Gemma 4 E2B Q4_K_M and Q8_0 | 33k each | 1e-7 | 0.04 to 0.22% | 0 |
| Honest, Qwen3-30B-A3B MoE Q4_K_M | 59k | 1e-7 | 0.01% | 0 |
| Honest, Qwen3-4B Q8_0, 8 batched lanes | 172k | 1e-7 | 0.02% | 0 |
| Honest, Qwen3-4B Q4_K_M, 16 batched lanes (prefill and mixed batches only) | 355k | 1e-7 | 0.22% | 0 |
| Q6_K served as Q8_0 (4B, 8B) | 29k each | 1.2e-2 to 1.3e-2 | 100% | 1 |
| Q6_K served as Q8_0 (Qwen3.5-9B, Gemma 4 E2B) | 24k, 34k | 1.1e-2 to 1.2e-2 | 76 to 89% (f32 and bf16 tensors, unquantized in both files, pass) | 1 |
| IQ4_XS served as Q4_K_M (4B) | 29k | 5.8e-2 | 99.5% | 1 |
| Q4_K_M as Q5_K_M, Q5_K_M as Q6_K (4B) | 29k each | 2e-2 to 4e-2 | 71% (tensors with the same type in both files pass) | 1 |
| Q3_K_M as Q4_K_M, Q4_K_M as Q5_K_M (Qwen3-30B-A3B MoE) | 59k each | 4.3e-2 to 7.2e-2 | 78 to 80% | 1 |

P(audit fails) is simulated: one layer, 16 tokens, the fail rule,
bootstrapped from the runs, 20,000 trials. "0" means no failure in 20,000
trials. "1" means that every trial failed.

The runs found two groups of honest outliers. Both were gaps in the rounding
tables, not noise. The tables now hold both profiles:

- Gemma's `per_layer_model_proj` (bf16, cuBLAS). cuBLAS rounds the
  activations to bf16 in prefill. With that profile, `e` decreases from
  1.4e-3 to 1.7e-6.
- q4_1 `ffn_down` in decode. The CUDA MMVQ kernel multiplies the block
  scales in half precision (`__hmul2`), so `d_w·d_x` and `m_w·s_x` are each
  rounded to f16. With that profile, `e` decreases from 1.3e-2 to 3e-8. CPU
  q4_1 (x86 and ARM) has its own rule: the q8_1 block sum is `f16(d·Σq)` with
  the unrounded `d`. With that profile, `e` decreases from 9.7e-4 to 1e-6.

Before the correction, these values were near or more than 1e-3, and a
strict rule failed honest nodes. The tables must prevent this failure mode.

In prefill, attention projections reach 1e-4 on up to 15% of columns. These
errors come from near-ties in activation quantization. That is why the
tolerance is 1e-3 and not 1e-4.

### Honest failure rate and near-tie repair

| Build | Model and backend | Failing column checks |
|---|---|---|
| Before the near-tie repair | All honest backends | Up to 1 in 300 |
| With the repair | Qwen3-4B Q8_0 on CUDA | 0 of 33,384 |
| With the repair | Qwen3-4B Q4_K_M on CUDA | 5 of 66,798, all on `attn_output` |

Before the repair, one role of 16 columns failed about 1 in 47,000 audits.
An audit of the 7 roles of a dense layer failed about 1 in 6,700. At the
worse Q4_K_M run with the repair (1 in 8,350 column checks), a role fails
about 1 in a billion audits. An audit fails about 1 in 150 million.

With q8 activations, fewer than 1 in 2,560 values of an opened activation
change their rounding when they move by 8 f32 ulps.

### Other backends

**Apple M6 (Metal tensor API), lab harness.** Seven honest runs (Qwen3-4B
in four quants, Qwen3-8B, Qwen3.5-9B, Gemma 4 E2B) have a median `e` of
4e-7 and a maximum of 5.5e-6. No column check is over 1e-3. Q6_K served as
Q8_0 fails every simulated audit on all three models tested. These numbers
did not reproduce through `skippy serve` (see the Metal entry below).

**DGX Spark (GB10, CUDA sm_121), lab harness.** The method and the rounding
profiles are those of the RTX 5090. Seven honest runs (Qwen3-4B in Q4_K_M,
Q4_0, Q4_1 and Q8_0, Q4_K_M with 8 batched lanes, Gemma 4 E2B and
Qwen3.5-9B) have a median `e` of 1.3e-7 to 1.8e-7. None fails a simulated
audit. The share of column checks over 1e-3 is 0 to 0.34%, which is higher
than on the RTX 5090. Most of them are in prefill `attn_output`: one token
in one layer, on every row set. The allowance of two failing columns
absorbs them. The less frequent decode outliers (up to 1e-4) come from CUDA
op fusion at near-ties in activation rounding. `GGML_CUDA_DISABLE_FUSION=1`
removes most of them. The q8 block scale is `amax · (1/127)` rounded, which
differs from `amax / 127` in 4.4% of blocks. That does not explain the
outliers.

**Vulkan (RTX 5090, NV_coopmat2), lab harness.** Decode matches the f32
profile (`e` ≤ 3e-7). For quantized weights, prefill accumulates in f16 by
default. Honest `e` is then 1e-3 to 4e-3, which is more than the tolerance,
so every honest node fails. With f32 accumulation, prefill matches the
f16-weight profile (`e` ≤ 1.1e-5). For q6_K, the profile also rounds
`d·scale` to f16 before the weight (two roundings). Coopmat off or f16 off
alone does not correct it. f32 accumulation costs 27% of prefill throughput
on Qwen3-8B Q4_K_M (8,150 to 5,870 tokens/s). Decode does not change. Other
Vulkan devices pick other paths (KHR coopmat, the scalar path, or integer
dot products), and each path needs its own profile entry.

**Vulkan through `skippy serve`.** The auditor gives Vulkan three profiles:
`raw`, `f16_weights` (f16 weights times f16 activations) and `cpu_q8_0`
(integer dot products on q8_1 activations). Two-stage Qwen3-4B splits, 14
honest audits each, all passed:

| Device | Quant | Best profile, rows checked | Failing columns | Max `e` |
|---|---|---|---|---|
| Intel Arc iGPU (Mesa ANV, integer dot) | Q8_0 | `cpu_q8_0` on all 392 | 1 of 1,556 | 1.0e-3 |
| RTX 5060 Ti (NVIDIA driver) | Q8_0 | `raw` on all 392 | 0 of 1,562 | 3.7e-5 |
| RTX 5060 Ti (NVIDIA driver) | Q4_K_M | `raw` on 370, `f16_weights` on 18, `cpu_q8_0` on 4 | 2 of 1,559 | 1.5e-3 (q4_K `ffn_down`) |

Q4_K_M served and claimed as Q5_K_M fails every role of every challenged
layer. On the RTX 5060 Ti, receipts make the prefill of a 1,665-token
prompt 1.66 times slower: 1,680 ms, against 1,010 ms with receipts off.
Most of that cost is the f32 accumulation. This number is the cost of
receipts on a serving stage. The 27% above is the cost of f32 accumulation
alone in the lab harness, on another model and GPU.

**Metal through `skippy serve`.** Qwen3-4B Q8_0 on an Apple M6. Decode rows
match `raw` within 1e-7. Every prefill ubatch differs from the exact
product by 1e-4 to 8e-4. This is true for all ubatch lengths, with the
tensor API on or off, and with op fusion on or off. No rounding model fits:
f16 activations, f16 weights and activations, f16 products, f16 outputs,
bf16, or f16 partial sums over 8 or 32 terms. A profile that is only near
the values fails honest nodes at the 1e-3 rule, so Metal has no backend
family. Replay on Metal needed one correction: Metal has no pinned host
buffers, and its asynchronous read accepts only memory that it mapped. So
the recorder reads values into pageable memory with blocking copies.

**ARM CPU (DGX Spark, Neon with i8mm and SVE), lab harness.** The CPU
profile matches as on x86: `e` ≤ 4e-7 for q4_K, q4_0 (repacked) and q8_0,
and ≤ 9e-7 for q4_1 with the rule above.

### Row fetch

These timings are from one server to the Hugging Face CDN. The client uses
range requests only, because the CDN rejects multi-range requests. Each
audit uses 16 rows per matrix and new rows, so the client keeps no stored
key.

| Model | Size | Header | One audit |
|---|---|---|---|
| Qwen3.5-27B Q4_K_M | 15.6 GiB | 16 MiB, 2.2 s | 144 ranges, 0.6 MiB, 0.7 to 1.1 s |
| Qwen3-235B-A22B Q4_K_M (3 parts) | 132 GiB | 8.5 MiB, 3.3 s | 800 ranges, 3.5 MiB, 1.2 to 2.5 s |
| Qwen3.5-397B-A17B Q4_K_M (6 parts) | 227 GiB | 11.7 MiB, 4.5 s | 900 ranges, 10 to 14 MiB, 1.4 to 2.0 s |
| DeepSeek-V3.2 Q4_K_M (9 parts) | 378 GiB | 10 MiB, 6.4 s | 1,300 ranges, 10 MiB, 1.3 to 2.3 s |

For MoE audits, most of the bytes are the whole expert router. A check of
every routed (token, expert) pair, not 16 pairs, took 21 to 26 seconds.
That is why the client samples pairs.

The fetched bytes were the same as the local copy of `unsloth/Qwen3-4B-GGUF`
(4,048 ranges).

**Sparse copy, Qwen3-4B Q8_0 (4.0 GiB file), Hugging Face file read by
range.** Before the head key existed, the sparse copy held 405 MiB. Almost
all of it was the output matrix. The first audit took 19 s, and later
audits took about 9 s. With the head key, the copy does not keep the output
matrix.

**Stored row keys (not built).** A client that prefers a stored key can
fetch one before the audit. A key holds 16 rows per matrix and 8 experts
per layer:

| Model | Key size |
|---|---|
| Qwen3.5-27B | 36 MiB |
| GLM-4.7-Flash | 37 MiB |
| Qwen3.5-397B-A17B | 86 MiB |
| Qwen3-235B-A22B | 106 MiB |
| DeepSeek-V3.2 | 129 MiB |

**Limits seen from one IP.**

- The resolver (`huggingface.co/.../resolve/...`) permits 3,000 requests
  per 300 s (`ratelimit-policy` header). It redirects to a signed CDN URL
  that expires after about 60 minutes.
- The CDN (`cdn.hf.co`) set no limit over 24,000 range requests at about
  800 per second.
- HTTP/2 with 64 requests in flight on one connection: 423 requests per
  second, median 131 ms. A pool of 64 HTTP/1.1 connections: 342 per second.

**Layer packages.** A `meshllm/*-layers` package stores each layer as its
own GGUF (`layers/layer-NNNNN.gguf`) plus `shared/*.gguf`. Its manifest
names the source repo, the revision and the file hashes. The tensor bytes
are the same as those of the source GGUF: every tensor of Llama-3.2-1B, and
layers 40 and 93 of Qwen3-235B-A22B (24 tensors).

### Node: commitment and replay

**Replay probe, CUDA (RTX 5060 Ti), lab tree.** The probe hashed the
residual output of every layer for every token. Then it compared the hashes
across recomputations:

| Recomputation | Qwen3-8B | Qwen3.5-9B (hybrid) | Gemma 4 E2B (sliding window) |
|---|---|---|---|
| Full rerun, same ubatches | identical | identical | identical |
| Drop KV from a ubatch boundary, replay to the end | identical | not supported (recurrent) | identical |
| Restore a sequence-state snapshot, replay | identical | identical (98 MiB snapshot) | differs |
| Replay decode tokens from the cache | identical | n/a | identical |
| Different ubatch size | differs | differs | differs |
| One token replayed alone | differs | n/a | differs |

A probe on Skippy's llama tree with 4 lanes and a unified KV cache replayed
one lane with dummy tokens in the columns of the other lanes, at the
recorded cells. The residuals were identical for dense, MoE, sliding-window
and hybrid models.

**The same probes on other backends, lab tree.** Qwen3-4B, Gemma 4 E2B and
Qwen3.5-9B on every backend, plus Qwen3-30B-A3B on Vulkan:

| Backend | Rerun, prefix replay, snapshot restore, decode replay | 4-lane replay with dummy tokens | Different ubatch size |
|---|---|---|---|
| Vulkan (RTX 5090) | identical | identical (all 4 models) | differs |
| CUDA (DGX Spark GB10) | identical, except Gemma's snapshot restore (as on the RTX 5060 Ti) | identical | differs |
| ARM CPU (DGX Spark) | identical | identical | identical |
| Metal (Apple M6) | identical | identical | identical |

ROCm was not tested.

**Replay on Skippy's serving stack.** RTX 5090, CUDA graphs, the real
scheduler, 4 concurrent requests. A scratch build of Skippy wrote a replay
log. An offline tool replayed each lane in a scratch context and compared
every logits row with the served one:

| Serving setup | Model | Logits rows identical |
|---|---|---|
| 4 lanes, staggered prefill and batched decode | Qwen3-8B | 181 of 181 |
| 4 lanes, prefix cache on, multi-turn reuse across lanes | Qwen3-8B | 141 of 141 |
| Same | Qwen3-30B-A3B (MoE) | 141 of 141 |
| Same | Gemma 4 E2B (sliding window) | 141 of 141 |
| Native MTP, 1 lane (26 verify windows, 5 rejected) | Qwen3.5-9B (hybrid) | 55 of 55 |
| Native MTP, 4 lanes | Qwen3.5-9B (hybrid) | 197 of 197 |
| 2-stage split, 4 lanes: stage 1 logits | Qwen3-8B | 180 of 180 |
| 2-stage split: stage 0 exports against stage 1 inputs | Qwen3-8B | 3,511 of 3,511 activation rows |

The same tools on a DGX Spark (GB10, CUDA), 4 lanes, prefix cache on, two
waves of requests that share a prefix:

| Model | Logits rows identical |
|---|---|
| Qwen3-4B Q4_K_M | 241 of 241 |
| Gemma 4 E2B (sliding window) | 241 of 241 |
| Qwen3.5-9B with native MTP (hybrid) | 267 of 267 |
| Qwen3-4B Q4_K_M on the Spark's ARM CPU (GPU hidden) | 241 of 241 |

The Qwen3.5 replay must use the server's slot count for recurrent state
(`n_rs_seq`). With a different count, most rows differ.

On the CPU, a replay with 4 threads differs on every decode row from a
server with 20 threads. A `-march=native` build of the same tree differs
from the release runtime, which builds ggml-cpu for generic ARM
(`GGML_NATIVE=OFF`).

**Anchored replay, Qwen3-30B-A3B, two stages on an RTX 5090.** With fusion
on, a full replay drifted 9e-3 from the committed values, and every opening
failed. With the anchored replay, every opening passed, with and without
native MTP. The glue errors were at most 4.7e-6, and an opening took 1.3 to
1.9 s. The design of the anchored replay is in
[RECEIPTS.md](RECEIPTS.md#node-commitment-and-replay).

**Two-node split with sparse attention.** DeepSeek-V4-Flash (43 layers,
145 GiB, DSA with a lightning indexer, hyper-connections) from its
`meshllm` layer package, split across two DGX Sparks over their direct
link. Layers 0 to 20 ran on one, and layers 21 to 42 on the other, with 4
lanes and the prefix cache on:

| Run | Stage 1 logits rows | Stage 0 exports against stage 1 inputs | Checkpoints |
|---|---|---|---|
| 6 requests, prompts of 274 to 1,532 tokens | 240 of 240 | 6,386 of 6,386 | 48 of 48 |
| 5 requests sharing a prefix, prompts of 4.6k to 7k tokens (top-k selective) | 100 of 100 | 13,810 of 13,810 | 20 of 20, with 3 cross-lane restores |

Three things differ from the other models:

- The DSV4 cache keeps one stream per sequence and splits every ubatch by
  sequence, so lanes do not share a ubatch. The replay lets the cache place
  the cells.
- With long prompts, Skippy moves lanes through state checkpoints: it saves
  one sequence and restores it into another. The replay follows each
  restore back to the sequence that saved it.
- Stages number lanes independently. Sequence 2 of stage 0 was sequence 3
  of stage 1. Exports are matched to imports by position and row hash. The
  boundary activation is the hyper-connection state: 4 × 4,096 values per
  token.

**Per-sequence KV streams.** Two-stage Qwen3-4B Q8_0 split on the CPU with
4 lanes, stage 1 with a stream per sequence. Four concurrent audits passed.
The attention rows of stage 1 opened from their streams, with block errors
less than 3e-8.

**Storage and hashing.**

- Without replay, the node stores the residual stream for every layer and
  token: 576 KiB per token for Qwen3-8B in f32, or 1.7 GiB for a
  3,072-token turn.
- BLAKE3 hashes 1.8 GB in 0.35 s on one core and 0.09 s on 8 cores. The
  prefill of the same turn takes 1.4 s on the RTX 5060 Ti.
- On CUDA with GPU hashing, Qwen3-4B copies about 3 KiB of digests per
  token per 18-layer stage. A copy of the values is 0.8 MiB.
- Per position, a stage keeps `2 · layers + 1` hashes: about 1.2 KB for 18
  layers.

### Serving cost history

Each row is a different build. Do not compare a row with the current build
without a new measurement.

| Build | Hardware and model | Decode | Prefill |
|---|---|---|---|
| Milestone 0: every matmul column stored | RTX 5060 Ti | 1.8x slower | 9x to 10x slower |
| Milestone 0 | RTX 5090 | 2.3x slower | 9x to 10x slower |
| Values copied on the decoding thread, one layer per challenge | RTX 5060 Ti, two-stage Qwen3-4B Q8_0, median of five runs | 71.7 → 69.5 tok/s (3% slower) | 2292 → 1661 tok/s (1.38x slower) |
| Values copied off the decoding thread, 0.8 MiB per token per stage | RTX 5060 Ti (PCIe 4.0 x4) | Not recorded | 1.27x slower |
| Same | RTX 5090 (PCIe 5.0 x16) | Not recorded | 1.13x slower |
| GPU hashing, digests copied | RTX 5060 Ti, both stages, 1,004-token prompt after a decode | 70.2 → 70.8 tok/s | 1648 → 1562 tok/s (1.06x slower) |
| Same | RTX 5090, both stages | 149.5 → 146.2 tok/s | 2090 → 1977 tok/s (1.06x slower) |

The GPU-hashing rows are the median of five runs, with receipts off and
then on in one session.

**Pinned cells.** Back-to-back 256-token Qwen3-4B requests on an RTX 5090,
all bound: decode ran at 117.5 tok/s, against 119.5 tok/s with receipts
off (1.7% slower). When every turn was pinned for 30 seconds, the cost was
4.5%. Unbound requests ran as fast as with receipts off. So did the second
turn of a conversation that continues the first.

**Recurrent state commits.** Two-stage Qwen3.5-9B Q4_K_M split, node on
CUDA (RTX 5060 Ti): the state commits made decode about 4% slower (median
55.5 against 57.9 tokens/s over seven 256-token turns each).

### Opening time history

**One layer per challenge, replay from the first position.** Two-stage
Qwen3-4B Q8_0 split on an RTX 5060 Ti. The challenge named the last 16
positions, which was the worst case for that build:

| Conversation (prompt + generated) | Serving | Open layer 5 (stage 0) | Open layer 30 (stage 1) |
|---|---|---|---|
| 86 + 256 | 3.9 s | 0.67 s | 1.27 s |
| 88 + 1024 | 14.3 s | 2.02 s | 4.30 s |
| 2859 + 32 | 2.1 s | 0.39 s | 0.66 s |
| 2857 + 512 | 9.3 s | 1.32 s | 2.76 s |

The current build does not replay from the first position, so this table
does not describe it.

**Replay of the opened ubatches only.** Two-stage Qwen3-4B split:

| Turn | Hardware | Opening time |
|---|---|---|
| 15,000 fresh tokens | RTX 5060 Ti | 1.7 s (5.5 to 8 s before this change) |
| 31,000 fresh tokens | RTX 5090 | 1.9 s |
| Turn from a cached prefix | Not recorded | About 1 s |

### Client checks

CUDA on the RTX 5090, prototyped in the lab harness. "Rel" is the relative
L2 error per token. "Max" is the largest single element, relative to the
RMS of the column.

**Glue.** The client rebuilds each matrix input from committed values in
f64: the output of the previous layer, matrix outputs, attention outputs
and weights.

| Rebuilt value | Qwen3-8B | Qwen3-30B-A3B |
|---|---|---|
| Matrix inputs (norm, SwiGLU, attention output) | rel ≤ 1.5e-7, max ≤ 1.3e-5 | rel ≤ 1.3e-7 |
| Layer output `x_{l+1}` (incl. MoE combine) | rel ≤ 7e-8 | rel ≤ 1.1e-7 |
| V rows in the cache | bit-identical | bit-identical |
| K rows in the cache (norm, RoPE, f16) | 2 to 10% of values off by 1 f16 ulp, rel ≤ 1.3e-4 | rel ≤ 1.2e-4 |
| Q | rel ≤ 1.4e-4 | rel ≤ 1.8e-4 |

- Row checks on rebuilt inputs match row checks on the GPU's own inputs.
  For Qwen3-8B prefill, p99 `e` is 2.4e-7 for both. The maximum is 7e-4 for
  both, from activation-rounding ties.
- The RoPE error increases with position, because the angle is computed in
  f32. At 16k positions, K reaches rel 4e-4. Exact RoPE is nearer to the
  GPU than an emulation of the GPU's float formula, which uses fast-math
  sine.
- Through `skippy serve` on CUDA, Q at positions 1,600 to 2,350 was off by
  1.2e-4 to 1.8e-4. A flat tolerance of 1e-4 failed these values. The
  shipped tolerance increases with position (see
  [RECEIPTS.md](RECEIPTS.md#parameters)).
- Through `skippy serve`, glue errors were at most 3e-6 with the node on a
  GPU and the auditor on the CPU. Honest row errors were 1e-4 to 4e-4.

**Attention.** The honest error is GPU flash attention against exact f64
attention on the committed Q, K and V. The cheat error is a cheated output
against honest exact attention:

| Model, context | Honest prefill (max) | Honest decode (max) | Window 1024 (min) | KV q8_0 (min) |
|---|---|---|---|---|
| Qwen3-8B, 3k | 2.4e-3 | 7.6e-7 | not run | not run |
| Qwen3-8B, 16k | 5.7e-3 | 7.2e-7 | 1.4e-2 | 5.1e-3 |
| Qwen3-8B, q8_0 KV cache | 4.4e-3 (16k) | 3.2e-4 at 4k with Q as q8_1 (p99 6e-7) | 1.4e-2 | not run |
| Gemma 4 E2B, 16k (sliding window, heads 256/512) | 9.0e-3 | 6.5e-4 | 6.3e-2 | 1.3e-3 |
| Qwen3.5-9B, 16k (hybrid) | 6.8e-3 | 3.6e-7 | 1.5e-2 | 1.8e-3 |

- Decode is tight on every model. With a q8_0 KV cache, the CUDA kernel
  rounds Q to q8_1.
- The prefill error increases about as the square root of the context. The
  combine check alone cannot separate a mild cheat (a q8_0 KV cache) from
  honest prefill. The recomputed blocks do the detection, and truncation
  fails both.
- End to end on a two-stage Qwen3-4B Q8_0 split (RTX 5090, turns of fewer
  than 100 positions), the partials of eight heads added about 3 KB to a
  9.7 MB opening. The second request returned the rows of the drawn blocks,
  about 1 MB, in 0.01 s. Honest partials combined to the committed output
  within 4e-3. The recomputed blocks were within 5e-8 of the node's.
- Gated attention output, Qwen3.5 layer 3, CPU: combine error 6e-4.
- Qwen3-4B with a q8_0 KV cache, CPU, layers 0 and 20: combine errors less
  than 2e-7 when the check rounds q to q8_0 as the kernel does.
- A 16k turn opens 256 blocks per head. The rows of two full blocks per
  pick are about 2 MB on Qwen3-4B. This is an estimate and was not measured
  end to end on an earlier build.

**Recurrent mixer (Qwen3.5-9B Gated DeltaNet), lab f64 model.** The lab
recomputed the state update and outputs in f64 from the committed q, k, v,
gate, beta and incoming state. The honest output error is at most 2e-5 in
prefill and 1.6e-7 in decode. The state error is at most 6.2e-6. A state
kept in f16 shows about 1e-4 (median), bf16 about 1e-3, and a zeroed state
0.25 or more. In the lab model, a tolerance near 5e-5 separates them.

**Recurrent mixer, shipped check.** The shipped check runs the layer's own
ggml ops and uses a tolerance of 1e-3, because a backend can sum the scan
in other chunks.

| Split | Node | Region | Result |
|---|---|---|---|
| Two-stage Qwen3.5-9B Q4_K_M | CPU | 1, 6 and 16 positions on layers 0 to 2 | Error 0 for outputs and both states, about 0.1 s each |
| Same | CUDA (RTX 5060 Ti), auditor on the CPU | 64-position prefill regions | Within 2.6e-5 (outputs) and 1.9e-6 (states) |
| Same | CUDA (RTX 5060 Ti), auditor on the CPU | Decode regions | Within 5e-7 |

A 64-position region adds about 12 MB to the opening of the stage. A
one-position region adds about 6 MB. Almost all of it is the two states at
the ends of the region.

**MoE routing.** On Qwen3-30B-A3B, the top-k of the client's router logits
matched the experts used for every checked token, in prefill and decode.
The smallest top-k gap was 6.7e-4, against router logit errors near 1e-7.

**Sparse attention (DSA), DeepSeek-V4-Flash.** The model scores the
compressed KV blocks with a lightning indexer (64 heads × 128, ReLU,
per-head weights), and attention reads the top 512. Top-k selects only
after 2,048 tokens, because the blocks are compressed 4×. On the
long-prompt split run above, the lab recomputed the indexer scores in f64
from the GPU's q, k and weights. It compared the GPU's selection with the
exact top 512, across all 21 indexer layers:

| | Rows (token × layer) | Result |
|---|---|---|
| Honest score error, relative to the row's score spread | 290k | p99 ≤ 9.6e-5, max 2.3e-3 |
| Honest selections that differ from the exact top 512 | 204k (104M selections) | 7,818, all within 1e-3 of the spread from the 512th score (a band that holds 0.7% of positions) |
| Cheat: indexer with half its heads | 107k | 99.8% of rows have a selection outside the 1e-3 band (9% of selections); 97% at 1e-2 |
| q and k rounded to bf16 | 107k | 3 rows outside 1e-3, none at 1e-2: within tolerance |

A full check needs all of the token's indexer keys in that layer:
1,750 × 128 values at 7k tokens. The eval callback that reads the indexer
splits the graph and changes rounding, so these numbers come from a
separate pass, not the bit-exact replay.

A DSA opening is not built. The lab proposed this acceptance rule. The
selection has exactly `top_k` entries, from the GGUF. Every selected
position scores at least `τ - band`, and every unselected position at most
`τ + band`. `τ` is the exact 512th score, and the band is 1e-2 of the score
spread of the row, which is more than the honest maximum. A smaller opening
gives the selected indices and `τ`. The client then checks 16 selected and
16 unselected positions, which catches the half-heads cheat on most rows.
Each sampled key must also chain to the committed residuals through the
compressor of the indexer.

**Output head and sampling.**

- Logits rows pass the row check like other matrices (`e` ≤ 1.1e-7).
- Lab, Qwen3-8B with a q6_K head on CUDA: the honest difference between
  `r · y` and `(rᵀW) · x` is at most 3.7e-4 logits. Four vectors make a
  128 KiB key, built in one pass over `output.weight` (480 MiB). The top-two
  logit gap was as small as 7.4e-4.
- Shipped head key, Qwen3-4B Q8_0 on CUDA: the key file is 107 KiB. The
  honest key error is at most 1.5e-7 of the norm of the logits. The head
  check decreased from about 6 s and 405 MiB of cached matrix to about
  10 ms.
- `std::uniform_real_distribution<double>` over `mt19937` gave identical
  draws with GCC 15 libstdc++ and Apple clang 21 libc++. The shipped check
  does not replay the draw.
- Head rounding flips, Qwen3-4B. At about one position in 150, a value of
  the normed residual is within a few ulps of a q8 step boundary. The
  node's logits are then about 4e-4 of their norm from the reference. The
  auditor lists the values that can move: fewer than one per position with
  q8, and about ten with f16.

### Auditor cost history

**One layer of 16 positions per opening.** Release build, auditor on the CPU
of the machine that served:

| Model | Layer load | Glue, first evaluation | Glue, per position after | Row checks |
|---|---|---|---|---|
| Qwen3-4B Q8_0 | 0.5 to 1.3 s | 76 to 117 ms | 0.04 to 0.27 ms | 21 to 461 ms |
| granite-3.1-3b-a800m Q8_0 (MoE) | 0.9 s | 99 to 161 ms | 0.03 to 0.27 ms | 0.6 to 1.0 s |

The first evaluation creates the glue context and builds the graph. Later
evaluations of the same layer use them again, and the per-position number
is the fastest of five. The row checks include a full recomputation of
granite's router (40 rows) and the row reads from the GGUF, so they change
with the page cache. One audit cost about 1 to 2 seconds of CPU time.

The arithmetic of the row checks is 16 dot products per matrix per token,
which is less than a millisecond. The fixed costs above are much larger
than this arithmetic. The current audit time is in
[Client audit time](#client-audit-time).

### End-to-end runs

| Build | Setup | Case | Result |
|---|---|---|---|
| Milestone 0 | Two-stage Qwen3-4B, one RTX 5060 Ti | Serve Q8_0, claim Q8_0 | Pass on layers of both stages, max error 1e-4 to 4e-4 |
| Milestone 0 | Same | Serve Q6_K on both stages, claim Q8_0 | Fail on every matrix, error 7e-3 to 5e-2 |
| One layer of 16 positions | Two-stage Qwen3-4B, RTX 5060 Ti and CPU | Serve Q8_0, claim Q8_0 | Pass on layers 0, 5, 17, 18, 30 and 35. Row errors 1e-4 to 3e-4 |
| One layer of 16 positions | Same | Serve Q6_K, claim Q8_0 | Fail on every matrix, error 1e-2 to 1e-1. Merkle paths, replay and glue pass |
| One layer of 16 positions | granite-3.1-3b-a800m (MoE) | Serve Q8_0 | Pass on layers 0, 5, 20 and 31. Expert ids and selections match the CPU exactly |
| One layer of 16 positions | Same | Serve Q6_K, claim Q8_0 | Fail on every quantized matrix. The f32 router passes |
| Four layers, bindings across layers | Two-stage Qwen3-4B, CPU | Serve Q8_0, claim Q8_0 | Pass |
| Four layers, bindings across layers | Same | Serve Q6_K, claim Q8_0 | Fail on every weight role and on the input of layer 0 (error 1.9e-2). Merkle paths, boundaries, replay and glue pass |
| Native MTP | Two-stage Qwen3-30B-A3B with an MTP head, CPU | Turns with accepted drafts, and turns that rejected every draft | Open and pass |
| Split GGUF | Qwen3-4B Q8_0 in three shards, two-stage split, CPU | Audit and recheck with `repo@revision/file` | Pass |
| Mesh | Two-node Qwen3-4B split, one RTX 5060 Ti | Challenges of the layer on the other peer | Pass |
| Mesh | Two nodes on one machine, Qwen3-4B, debug build | Honest turns | Pass in about 8 s each |
| Mesh | Same | Peer serves Q6_K, receipts name the Q8_0 file | Fail on every weight role, the input of layer 0 and the output head. The peer was blocked before the next request |
| Mesh | Two-node split, empty cache | `skippy recheck-receipt --evidence FILE` | 12.7 s |
