# Workload oracle evidence fixture contract

These fixtures were authored from the three legacy sources and their existing
unit tests before implementing the candidate. No candidate output supplied an
expected value. The fixture artifacts are literal bytes: model `abc`, candidate
empty, oracle `hello`, projector `hello\n`. SHA-256 expectations are fixed known
digests, independent of the candidate hasher. These are evidence-policy fixtures,
not models, runnable comparators, or PCM audio.

The isolated correction passed 40 active Rust tests, including the original 32,
and an explicit differential/file-API run against unchanged Python commands.
That run retained 178 observations, including 72 PCM rows through both writer
and verifier, exact successful writer bytes, error-order cases, and one-byte
artifact tampering. See `WORKLOAD_ORACLE_CORRECTION_RECEIPT.md` at the candidate
root for source hashes, commands, and compatibility limits. The original receipt
and source review remain historical pre-correction records.

Interpreter-backed differential tests have been removed. Current tests execute
the Rust evidence writer and verifier and assert domain behavior and consumed
output contracts. The differential observations above are historical evidence.

## Behavior census

- Writer: terminal `-smoke` suffix only; line splitting and trimming; last nonempty
  class-specific comparator-pass line; arbitrary writer class and supplied model
  digest remain accepted; executable basename and artifact bytes are recorded.
- Verifier: embedding/rerank/ocr/speech_recognition use `llama-server`,
  encoder_decoder uses `llama-completion`, speech_synthesis uses `llama-tts`.
  Verify all eleven identity fields against caller-supplied local inputs. Hash
  model, projector when supplied, candidate and oracle independently. Check the
  actual oracle basename as well as the evidence's recorded basename.
- Projector: required only by the verifier for ocr, speech_synthesis and
  speech_recognition; optional but digest-bound for every other class. Absent
  projector evidence is equivalent to null only when no projector is supplied.
- Comparator: exact class-specific prefix required; writer strips log lines but
  verifier does not strip the recorded comparison. Neither executes a comparator.
- TTS: writer additionally requires a passing result bound to the supplied patch;
  both paths require complete PCM metrics. Positive integer fields reject bool,
  float, null and string, without imposing a u64/i128 ceiling. RMS includes both
  endpoints of [0.0, 0.02], cosine both endpoints of [0.9995, 1.0]. Out-of-bounds,
  NaN and either infinity fail. Integer 0/1 and negative floating zero remain
  legal where within bounds. Preserve additional metrics; do not impose PCM
  schema policy on non-TTS evidence.
- JSON: object requirement, malformed/UTF-8/BOM rejection, duplicate-key
  last-value semantics, extra-field acceptance and sorted ASCII JSON output.
  Strings must contain valid Unicode scalars; lone surrogates are rejected.
  DEL is emitted as `\u007f`. Decoded keys sort by codepoint.
- Files: read failures fail; writer does not create output parents, and no
  validation rejection truncates an existing output. No executable permissions
  are required because executables are hashed as files, not launched.

JSON parsing uses the default serde JSON boundary, including its nesting and
number limits. Tests cover duplicate replacement, malformed input and complete
document rejection. Workload evidence CLI adapters are covered by
`migration_workload_oracle_cli`; this does not certify a workflow caller cutover.

Hashes bind local bytes only. They do not authenticate a remote producer, prove
that a comparator ran, or prove that supplied PCM metrics came from real audio.
