# Upstream opportunities

The ledger for work that belongs in llama.cpp rather than in this façade.
"Never modify llama.cpp from here" (AGENTS.md) means discoveries need a home
that survives the PR that found them — this is that home. The patch queue
(`skippy/llama_cpp/patches`) is where an upstream-shaped change lands once it
is accepted into our runtime; this file tracks everything before that.

Rules:

- Append an entry when you flag something upstream-shaped while porting,
  benchmarking, or fuzzing. Record the *evidence* (missing symbol, measured
  cost, sanitizer trace), not just the wish.
- Cross-reference the `benchmarks/cases.txt` row when one exists
  (`blocked-upstream` acceptance).
- Update Status when something is filed, fixed upstream, or folded into the
  patch queue. Do not delete entries — closed entries are the record of what
  we already asked for.
- When a finding is security-shaped, say plainly what was proven versus what
  is suspected, and keep the repro bytes or the method.

## Verified findings against the pinned upstream

### gguf.cpp validates tensor dimensions by overflowing them first

Status: **live in our pin** (llama.cpp `7c1b680d5`, build `b11313`) — known
upstream as CVE-2026-33298 / GHSA-96jg-mvhq-q7q7, with the gguf-layer residue
tracked in ggml-org/llama.cpp#27203.

What was proven by execution (2026-10-19 sweep, method below):

- `gguf_init_from_reader` (`ggml/src/gguf.cpp` ~line 701) guards tensor
  representability with `ggml_nelements(...) > 0 && (INT64_MAX/ne[1] <= ne[0]
  || ...)`. `ggml_nelements` (`ggml.c:1289`) multiplies all four dims as
  `int64_t` — signed overflow, UB, and it happens *inside* the check that is
  supposed to catch it.
- Dims `{2^32, 2^32, 2^16, 2^16}` (product 2^96) wrap to 0: the `> 0`
  precondition is false, the whole guard is skipped, and the file **parses**,
  yielding a tensor that claims ~7.9e21 GiB of F32 data with zero-length
  computed strides. UBSan fires at `ggml.c:1289` on the way in.
- The model-load path is protected by a second layer
  (`llama_model_loader::get_tensor` compares file dims against
  architecture-expected shapes), which is presumably why the advisory scopes
  the residual to consumers of raw `gguf_init_from_file`.

Our side: `read_tensors` rejects extents past `INT64_MAX` and extent products
that wrap (`tests/inventory_parser_test.cpp`,
`RejectsDimensionsWhoseProductWrapsToZero`), so the planner never sees the
shape. Action item: when the pin moves, re-run the sweep below and confirm the
gguf layer rejects the wrap corpus; if it still parses, keep our guard and
consider re-reporting to #27203 with the wrap-to-zero repro.

## API gaps that force façade-side duplication

### No public accessor for tensor source locations

The inventory parser (`include/llama/model_inventory.h`) exists because
`skippy_stage_planner_create_v1` needs an exact tensor closure — names,
absolute file offsets, stored lengths — *before* any model exists, and the
public C ABI exposes none of it (`tensor_source_locations` is internal to the
loader). A `skippy_model_tensor_source_locations_v1`-shaped entry point (or
equivalent public gguf accessors that survive the loader boundary) would let
`model_inventory.cpp` shrink to a thin call and retire our parser, its caps,
and its fuzz corpus for planning purposes.

Evidence: the parser is ~200 lines of untrusted-input handling with its own
defect history (uncapped reserves, unbounded array recursion, missing stream
checks — all fixed, all ASan-covered). The planner probe is the only witness
that file-derived values match native expectations.

Status: open. Blocked-upstream in spirit; no manifest row because no
benchmark case exists for "parse the header" in isolation.

### No prepared activation-export token (patch 0009 native half)

`PreparedExport` removes our per-hop allocation and descriptor rebuild, but
llama still maps parts internally per hop. Closing the last gap needs an ABI
entry point (`skippy_session_prepare_export_token_v1`-shaped) plus per-family
work in the patch queue.

Manifest: `benchmarks/cases.txt` row `0009 B stage_boundary blocked-upstream`.

### No owned request recipe for encoder stages (patch 0003)

BERT's `attention.mask.0` has no way to be staged as an owned request input,
so encoder models cannot take the two-stage path yet.

Manifest: row `0003 B stage_open_encoder blocked-upstream`.

## Performance worth asking llama for

### Native validate_chain re-derives guarded plans per call

Measured: native ~10.9 ms per validation vs ~0.18 µs for our cached plan
validation (~60,000x) — the native path rebuilds guarded plans on every call
where our `PlanStore` RCU cache serves an immutable snapshot. The cache shape
(copy-on-write index, atomic snapshot publish) is the upstream-worthy idea.

Manifest: class C ceiling row `stage_validate`.

### Per-decode fixed cost in the Metal path

~0.83 ms of fixed dispatch overhead per decode call regardless of work size.
Seen in the stage-chain profiling; removal needs backend-internal batching of
the dispatch, not façade work.

### Small matrices under-fill the GPU

384-wide matrices reach ~20–29 % of peak on a 40-core GPU. An embedding/rerank
workload is permanently narrow; worth an upstream look at narrow-matrix tiling
on Metal.

## Hardening observations (non-vulnerabilities)

- **O(n²) duplicate detection** in `gguf.cpp`: duplicate KV keys and duplicate
  tensor names are checked with an inner loop over all previous entries. Counts
  are bounded only by file size, so a metadata-heavy GGUF of a few hundred MB
  costs billions of string comparisons — quadratic CPU from untrusted input,
  not memory unsafety. A hash set fixes the class.
- **No in-tree fuzz corpus for `gguf_init_from_file`** that we could find. Ours
  (below) is cheap to regenerate and caught a live CVE class on first contact.

## Verification log

2026-10-19, pin `7c1b680d5` (b11313), out-of-tree build in `/tmp` so the
checkout stays clean:

```bash
cmake -S .deps/llama.cpp -B /tmp/llama-gguf-asan \
  -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -g' \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -g'
cmake --build /tmp/llama-gguf-asan --target ggml -j8
```

Driver: one `gguf_init_from_file` per file, `no_alloc = true`, free on
success, report parsed/rejected; any ASan/UBSan report is the finding.
Corpus (95 files, generator kept in /tmp/gguf-fuzz): every-byte truncations of
a valid one-tensor GGUF, implausible string/array/tensor/metadata counts,
unknown metadata and tensor types, 64-deep array nesting, duplicate names and
keys, negative and zero dims, hostile alignment values, and the wrap-to-zero
dim products above.

Result: zero crashes, zero ASan reports, all cases rejected with diagnostics
except the wrap-to-zero family — which parses and trips UBSan at
`ggml.c:1289` (the entry above). Data-section truncations (metadata complete)
legitimately parse because `no_alloc` never reads data.
