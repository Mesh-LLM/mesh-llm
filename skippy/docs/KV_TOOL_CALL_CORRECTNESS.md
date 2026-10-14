# KV cache and tool-call correctness

An exact prefix-cache restore must restore the prefix token IDs as well as KV
and recurrent tensors. Native sampling uses absolute token-history indices to
distinguish prompt tokens from generated tokens. Missing history delays grammar
activation by the number of cached tokens and also changes repetition penalties.

Issue [#2408](https://github.com/Mesh-LLM/mesh-llm/issues/2408) exposed this with a
351-token prompt and a 256-token cache hit: the lazy tool grammar never activated
during the 92-token completion. The model emitted Python's `True` where the tool
schema required JSON's `true`. Full-state payloads and disabling speculative
decoding did not avoid the problem.

The exact-cache transaction now completes its tensor import by restoring token
history. History restoration rebuilds sampling state using the current request's
grammar configuration. A failure rolls back the cache restore. Resident-prefix
restoration already supplies the token IDs.

This defect is not specific to Qwen or recurrent architectures. The shared native
sampler consumes token history for dense transformers and hybrid/recurrent models
alike. The fix covers serialized radix hits and the L2/L3 restore path, for both
full-state and KV/recurrent payloads. A model that happens to emit valid JSON
without effective grammar enforcement can hide the defect. These findings do not
establish corruption of the cached model tensors.

Streaming must also reject an incomplete terminal tool call. Partial parser
snapshots can change while a call is being generated; a terminal parse must match
the emitted name and argument bytes and contain a complete JSON object. A model
stop that violates this contract produces an inference error rather than an
ordinary successful `stop`. Explicit output-budget exhaustion retains `length`.

## Regression coverage

The native `skippy_sampling_history` test uses generated model fixtures and real
full-state and KV/recurrent exports. It checks cold/restored sampling, eager and
lazy grammar progression, repeated session reuse, restored generated tokens,
frequency/presence penalties, and rejected history-length mismatches. Frontend
tests exercise boolean truncation, missing calls, renamed calls, parser revisions,
valid terminal suffixes, and explicit output-budget exhaustion.
The synthetic native fixture is a dense Gemma model; live Metal validation uses
Qwen3.8-27B. Sustained replays should additionally cover a dense transformer and a
hybrid model. Representative coverage is not certification of every model family.

For sustained replay, use the typed Rust harness with
`--require-complete-output`, together with cache/context qualification:

```sh
cargo xtool automation replay-matrix run \
  --repo "$PWD" --ref "candidate=$COMMIT" --backend cuda \
  --model "$MODEL_FILE" --expected-model-sha256 "$MODEL_SHA256" \
  --dataset-file "$DATASET_FILE" --reader "$TRAJECTORY_READER" \
  --replay-mode all --passes 2 --sessions-per-concurrency 8 \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --minimum-worker-waves 2 --minimum-context-tokens 131072 \
  --minimum-session-prompt-tokens 32768 --min-isl 32768 --max-isl 131072 \
  --min-turns 5 --min-cache-pct 50 --require-recurrent-restores \
  --require-complete-output --output "$EVIDENCE_DIR"
```

The strict gate rejects missing completion evidence, missing finish reasons,
`length`, malformed or incomplete tool arguments, and inconsistent tool-call
finish reasons. Transport errors and missing `[DONE]` already fail requests.
Request records retain `output_problems`; cell summaries retain
`output_incomplete_requests`. This detects structural truncation, not semantic
incompleteness in otherwise valid prose. Context qualification additionally
requires measured later-turn cache reuse; recurrent qualification is appropriate
only for a model with recurrent memory.
