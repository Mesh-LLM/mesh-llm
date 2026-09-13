# Non-Chat Model Workloads

Mesh LLM can serve several llama.cpp model classes whose primary contract is
not causal chat generation. The public OpenAI-compatible endpoint stays on the
normal mesh port (`http://127.0.0.1:9337/v1`), while the loaded model advertises
its runtime-probed workload class for safe routing.

## Supported surfaces

| Model class | HTTP surface | Current execution boundary |
|---|---|---|
| Embedding / encoder-only | `POST /v1/embeddings` | Unsplit local full model |
| Cross-encoder rerank | `POST /v1/rerank` | Unsplit local full model |
| Encoder-decoder | `POST /v1/completions`, `POST /v1/chat/completions` | Unsplit local full model |
| OCR | `POST /v1/chat/completions` or `POST /v1/responses` with an image input | Causal trunk and projector colocated on one node |
| Speech synthesis | `POST /v1/audio/speech` | Unsplit local full model and projector |
| Speech recognition / translation | `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` | Causal trunk and audio projector colocated on one node |

Embedding, rerank, encoder-decoder, and speech-synthesis workloads do not
silently enter the stage-split generation path. A filtered stage model returns
a structured `unsupported` error. OCR and speech recognition use causal trunks,
but their projector remains local to the trunk; they are not claims of a
distributed projector implementation.

## Mixed-version routing

A model name can be advertised by several nodes running different versions.
Model discovery accepts any compatible advertisement, but each serving target
must independently advertise the workload required by the endpoint. The host
router and passive-client proxy exclude incompatible targets before context
ranking, cache affinity, reservation spreading, and retries. One current peer
does not grant its endpoints to a legacy peer serving the same model name.

Absent workload metadata remains compatible with ordinary generation requests,
not with embedding, rerank, speech synthesis, or audio-upload endpoints. Audio
uploads additionally require runtime-verified audio support in the same target's
model descriptor. Unsupported targets are not restored by an availability
fallback or a cached automatic model choice.

## Embeddings

The request accepts a string, an array of strings, a token array, or an array of
token arrays. `encoding_format` may be `float` or `base64`. Base64 values encode
little-endian `f32` values. The runtime honors the pooling type recorded in the
GGUF and returns L2-normalized vectors.

```python
from openai import OpenAI

client = OpenAI(api_key="mesh", base_url="http://127.0.0.1:9337/v1")
response = client.embeddings.create(
    model="nomic-embed-text-v1.5-Q8_0",
    input=["search_query: distributed inference", "search_document: pooled GPUs"],
    encoding_format="float",
)
print(len(response.data[0].embedding))
```

The optional `dimensions` field is accepted only when it equals the model's
native output width. Dimensionality reduction is not performed implicitly.

## Reranking

`/v1/rerank` is the common cross-encoder companion surface. Documents may be
strings or objects containing a string `text` field.

```bash
curl -sS http://127.0.0.1:9337/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jina-reranker-v1-turbo-en",
    "query": "distributed model inference",
    "documents": ["one GPU", "several GPUs connected over a mesh"],
    "top_n": 2,
    "return_documents": true
  }'
```

Results are ordered by descending relevance score. Each result retains the
zero-based index of the original document.

## Encoder-decoder generation and OCR

Encoder-decoder GGUFs use the existing completion and chat response shapes.
Their encoder pass and decoder loop are executed by the local Skippy runtime;
tool calls are rejected because the current encoder-decoder path does not
implement that contract.

OCR remains a multimodal text-generation request. Send the image using the
existing `image_url`/`input_image` content-part contract. Automatic routing
requires a model that advertises both a causal workload and vision capability.

## Audio

Speech synthesis returns raw binary data with the matching response content
type. The current native Qwen3-TTS path supports `wav` (`audio/wav`) and `pcm`
(`audio/pcm`). Other OpenAI response formats are parsed and rejected with a
structured unsupported error instead of returning bytes under the wrong media
type. Set `response_format` explicitly because the OpenAI request default is
`mp3`, which this native path does not encode.

```bash
curl -sS http://127.0.0.1:9337/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-TTS-12Hz-1.7B-Base-Q8_0",
    "input": "The mesh is ready.",
    "voice": "default",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Transcription and translation accept `multipart/form-data` with a required
`model` field and one `file` field. Uploads are bounded at 64 MiB. The supported
response formats are `json` and `text`.

```bash
curl -sS http://127.0.0.1:9337/v1/audio/transcriptions \
  -F model=ultravox-v0_5-llama-3_2-1b \
  -F file=@recording.wav \
  -F response_format=json
```

Automatic audio routing filters for a runtime-verified audio capability. An
explicitly selected incompatible model is rejected by the local backend rather
than silently discarding the audio.

## Class-specific workload certification

Every row in `ci/llama-canary/family-certified.json` records an explicit
`class`. The six checked-in non-chat rows use the certified `workload-oracle`
profile with separate class-specific smoke and local-monolithic oracle lanes;
they do not inherit the causal-generation `full` profile or its three split
handoff lanes. The checked-in representatives cover
embedding, rerank, encoder-decoder, OCR, speech synthesis, and speech recognition
with immutable Hugging Face revisions and artifact digests.

Run a focused local smoke with the model files already present:

```bash
scripts/skippy-workload-certify.sh \
  --class embedding \
  --lane embedding-smoke \
  --model-path /path/to/model.gguf \
  --model-id local-embedding \
  --work-dir target/workload-certification
```

Projector-backed classes additionally require `--projector-path`. Each lane
checks local real-model behavior and exercises its HTTP endpoint through the
OpenAI frontend. The embedding lane also uses the official Python OpenAI SDK
when that optional package is installed; its mandatory HTTP assertions do not
depend on the SDK being present.

An isolated invocation without an oracle remains a smoke check only: it verifies
local execution, the HTTP response contract, and class-specific coarse
assertions, but cannot satisfy the checked-in certified profile. Filtered
staged workloads also fail closed where unsupported. The mandatory family
battery oracle uses test-only llama.cpp executables built from the same pinned
patch queue; those executables are never packaged as Mesh-LLM serving backends:

```bash
LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_LINK_MODE=static \
LLAMA_STAGE_BUILD_DIR=/path/to/candidate-static-build \
just llama-build

LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_LINK_MODE=static \
LLAMA_STAGE_BUILD_DIR=/path/to/isolated-oracle-build \
LLAMA_STAGE_WORKLOAD_ORACLE=ON \
just llama-build

LLAMA_STAGE_BACKEND=cpu \
LLAMA_STAGE_LINK_MODE=static \
LLAMA_STAGE_BUILD_DIR=/path/to/candidate-static-build \
scripts/skippy-workload-certify.sh \
  --class embedding \
  --lane embedding-smoke \
  --model-path /path/to/model.gguf \
  --model-id local-embedding \
  --work-dir target/workload-certification \
  --oracle-server /path/to/isolated-oracle-build/bin/llama-server \
  --require-oracle
```

The runner checks each oracle executable's current pinned-source build stamp
and requires CPU-only execution, including Metal disabled on macOS. The
monolithic reference uses `--no-repack` to match the staged runtime's default
model-loading configuration; omitting that flag can create a numerical
mismatch unrelated to the workload implementation. Use
`--oracle-server` for embedding, rerank, OCR, or speech recognition;
`--oracle-completion` with `bin/llama-completion` for encoder-decoder; and
`--oracle-tts` with `bin/llama-tts` for speech synthesis. The family battery
accepts the corresponding `SKIPPY_WORKLOAD_ORACLE_SERVER`,
`SKIPPY_WORKLOAD_ORACLE_COMPLETION`, and `SKIPPY_WORKLOAD_ORACLE_TTS`
environment variables; a certified family fails closed if its class-appropriate
executable is absent. The direct completion CLI is intentional: the pinned
`llama-server` completion endpoint does not produce the correct FLAN-T5 text
for this fixture, while the pinned monolithic `llama-completion` encoder pass
and decoder loop do.

The per-class oracle gates are:

| Class | Independent comparison | Important limit |
|---|---|---|
| Embedding | Identical batch and each individual input; same vector width, maximum coordinate error `1e-4`, minimum cosine `0.99999` | Parity does not measure retrieval quality |
| Rerank | Identical query/documents; maximum score error `1e-4` and identical ordering | Parity does not measure ranking quality |
| Encoder-decoder | Same prompt and greedy seed; identical text after whitespace normalization | Pinned direct monolithic completion CLI, not the server endpoint |
| OCR | Same generated `MESH 42` PNG and prompt; normalized text matches and contains the independently known fixture label | One synthetic image does not certify broad OCR accuracy |
| Speech recognition | Same WAV and prompt; normalized text matches | The generic smoke WAV has no checked-in transcript label, so this is execution parity, not transcription accuracy |
| Speech synthesis | Same prompt, seed, top-k/top-p, and frame cap in deterministic in-process Skippy and monolithic `llama-tts`; PCM format/length match, relative RMS error at most `2%`, waveform cosine at least `0.9995` | Public HTTP speech sampling is stochastic; PCM parity does not establish intelligibility |

The family battery records an oracle pass only when the class-specific
comparison completes and a run-local evidence file matches the pinned model,
projector, candidate executable, oracle executable, and patch SHA. A smoke
pass cannot substitute for that evidence. The original provisional
`workload-smoke` profile remains available for future rows that have not yet
passed an independent oracle. Certification here is a bounded equivalence
claim for these six pinned artifacts and fixtures, not broad model-quality
or split-serving certification.
