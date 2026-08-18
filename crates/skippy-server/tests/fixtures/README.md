# Streaming tool-call parser fixtures

`qwen35_partial_tool_calls.txt` records what
`common_chat_parse(prefix, is_partial = true)` returns for growing byte-prefixes
of a `qwen35` tool-call generation. It is the input to the streaming tool-call
tests in `src/frontend/generation/tool_call_stream.rs` and
`src/frontend/tests/chat_stream_deltas.rs`, so those tests are driven by real
parser output rather than a hand-written guess at it.

`record_partial_tool_calls.cpp` is the generator. It calls the parser through the
same shape `skippy_parse_chat_response_json` uses (see
`third_party/llama.cpp/patches/0010-Add-Skippy-tokenization-and-stage-chat.patch`):
`common_chat_parse(text, is_partial, params)` followed by
`common_chat_msgs_to_json_oaicompat`.

## Regenerating

Regenerate after a `third_party/llama.cpp` pin bump or any patch-queue change
that touches chat parsing, and commit the result with that change. From a
prepared native checkout:

```sh
scripts/prepare-llama.sh pinned
cd .deps/llama.cpp
B=../llama-build/build-stage-abi-static-metal   # or your platform's build dir
c++ -std=c++17 -O1 -I common -I include -I ggml/include -I vendor -I src \
  ../../crates/skippy-server/tests/fixtures/record_partial_tool_calls.cpp \
  $B/common/libllama-common.a $B/common/libllama-common-base.a \
  $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a \
  $B/ggml/src/ggml-metal/libggml-metal.a $B/ggml/src/ggml-blas/libggml-blas.a \
  $B/ggml/src/libggml-base.a \
  -framework Accelerate -framework Metal -framework Foundation \
  -framework MetalKit -framework CoreFoundation -o /tmp/record_partial_tool_calls
/tmp/record_partial_tool_calls models/templates/Qwen3.5-4B.jinja \
  > ../../crates/skippy-server/tests/fixtures/qwen35_partial_tool_calls.txt
```

The generator emits the fixture file's exact line format, so a regenerated file
should differ only where parser behaviour changed. A `THROW` marker in place of a
`tool_calls` array records a prefix the parser rejected. The readers in
`tool_call_stream.rs` and `chat_stream_deltas.rs` do not special-case it: they
deserialize the payload as JSON directly and fail on a `THROW` line rather than
skipping it, so a new one appearing turns into a failing test — a behaviour
change worth reading before committing.
