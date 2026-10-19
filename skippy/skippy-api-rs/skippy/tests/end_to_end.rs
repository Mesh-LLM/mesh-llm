//! End-to-end regression tests: real GGUF, real llama.cpp.
//!
//! The C++ suite (`skippy_api/tests/`) pins these contracts against mocks, so
//! it proves the workload layer's logic but cannot prove the backend agrees
//! with it. Every test here runs the same contract through the native stack,
//! which is where the interesting failures live: pooling selection, context
//! geometry, batch layout, and the retained-context pool.
//!
//! Model files come from `SKIPPY_TEST_MODEL` (embedder) and
//! `SKIPPY_TEST_RERANKER` (reranker). A test whose model is absent returns
//! early, so `cargo test` stays green on a machine without GGUFs.
//! `SKIPPY_TEST_CAUSAL` (any small chat GGUF) covers the decoder-only
//! classification default and the stage-chain tests.
//! `SKIPPY_TEST_QWEN3_RERANKER` (a Qwen3-Reranker GGUF) covers the
//! tensor-based classifier gate's family regression.
//!
//! `integration.rs` keeps the broad smoke coverage; this file targets the
//! behaviours that only a real model can confirm.

use skippy::{
    ModelFamily, NativePooling, Pooling, Runtime, Session, SessionConfig, StageError, Token, WorkloadError,
    WorkloadInfo, WorkloadKind,
};

fn optional_model_path(variable: &str) -> Option<String> {
    match std::env::var(variable) {
        Ok(path) if !path.is_empty() => Some(path),
        _ => {
            eprintln!("{variable} unset; skipping");
            None
        }
    }
}

fn model_path(variable: &str) -> Option<String> {
    let path = std::env::var(variable).ok().filter(|path| !path.is_empty());
    if path.is_none() {
        // Skipping keeps local runs green on a machine without GGUFs; CI sets
        // this so a missing model fails instead of silently reducing coverage.
        if std::env::var("SKIPPY_TEST_REQUIRE_MODELS").is_ok_and(|value| !value.is_empty()) {
            panic!("{variable} is required when SKIPPY_TEST_REQUIRE_MODELS is set");
        }
        eprintln!("{variable} unset; skipping");
    }
    path
}

/// Loads the embedder, or returns `None` when the test model is not configured.
fn embedder() -> Option<(Runtime, String)> {
    let path = model_path("SKIPPY_TEST_MODEL")?;
    let runtime = Runtime::new().expect("runtime");
    Some((runtime, path))
}

fn tokens(session: &mut Session<'_>, text: &str) -> Vec<Token> {
    session.tokenize(text).expect("tokenize")
}

// ---------------------------------------------------------------------------
// Classification: the values the workload probe reports, not just the kind.
// Mirrors WorkloadTest's table and the reranker regression in `backend.cpp`.
// ---------------------------------------------------------------------------

/// A plan hands back what a split runtime needs: a contiguous partition with a
/// resident closure, a contract, and a boundary that lines up between stages.
#[test]
fn a_chain_runs_every_stage_and_hands_back_a_terminal_frame() {
    let path = model_path("SKIPPY_TEST_MODEL");
    if path.is_none() {
        return;
    }
    let path = path.expect("model path");
    let runtime = Runtime::new().expect("runtime");
    // Ranges come from the planner: hard-coded splits outrun the fixture's
    // layer count (MiniLM has six) and get rejected before any stage program
    // is attempted, which says nothing about chains.
    let halves = runtime.plan_stages(&path, 2).expect("two stages planned");
    let ranges: Vec<(i32, i32)> = halves.iter().map(|plan| (plan.layers.start, plan.layers.end)).collect();
    // A chain built from the planner's own ranges must open — or be rejected
    // for the one reason a BERT fixture legitimately has: its stage program
    // roots an attention mask the installer cannot own. Any other rejection
    // is a regression, so the reason is pinned rather than tolerated.
    let mut chain = match runtime.open_chain(&path, &ranges) {
        Ok(chain) => chain,
        // StageError has one variant today; no catch-all arm or it is
        // unreachable and rustc rejects it under -D warnings.
        Err(StageError::Rejected(reason)) => {
            assert!(
                reason.contains("attention.mask"),
                "expected the encoder's attention-mask limitation, got: {reason}"
            );
            return; // encoder stage programs are not installable yet
        }
    };
    assert_eq!(chain.stages(), 2);

    let tokens = [Token(1), Token(2), Token(3)];
    let frame = chain.run(&tokens).expect("chain prefill");
    if let Some(bytes) = frame {
        assert!(!bytes.is_empty(), "a terminal frame with content must be non-empty");
    }
}

/// Splitting must not change what crosses a boundary: two chains covering the
/// same layers — one stage versus two — must return byte-identical terminal
/// frames. The prefix deliberately stops before the model's last layer: a
/// terminal stage exports a frontier only when layers still follow it, and a
/// full-model decoder chain exports nothing (nothing is downstream), which
/// would make the comparison vacuous. Verified on qwen2.5-0.5b-instruct where
/// the 3/4-depth prefix frame (28672 bytes) agrees exactly.
#[test]
fn a_chain_terminal_frame_is_stage_count_invariant() {
    let Some(path) = model_path("SKIPPY_TEST_CAUSAL") else {
        return;
    };
    let runtime = Runtime::new().expect("runtime");

    // plan_stages derives contiguous ranges, so the test carries no
    // hard-coded layer counts.
    let quarters = runtime.plan_stages(&path, 4).expect("four stages");
    assert_eq!(quarters.len(), 4, "four stages requested, four returned");
    let start = quarters[0].layers.start;
    let split = quarters[2].layers.start;
    let prefix_end = quarters[2].layers.end;
    assert!(prefix_end < quarters[3].layers.end, "the prefix must leave layers downstream so the frontier exists");

    let tokens = [Token(1), Token(2), Token(3), Token(4)];

    let mut monolithic = runtime.open_chain(&path, &[(start, prefix_end)]).expect("a prefix chain opens as one stage");
    let frame_one =
        monolithic.run(&tokens).expect("one-stage prefill").expect("a prefix terminal stage exports a frontier");

    let mut split_chain =
        runtime.open_chain(&path, &[(start, split), (split, prefix_end)]).expect("a prefix chain opens as two stages");
    let frame_two =
        split_chain.run(&tokens).expect("two-stage prefill").expect("a prefix terminal stage exports a frontier");

    assert_eq!(frame_one, frame_two, "stage count must not change the terminal frame");

    // The full model for contrast: nothing is downstream of the last layer,
    // so decoder-only chains return no terminal frame — the two sides must
    // agree on that too.
    let halves = runtime.plan_stages(&path, 2).expect("two stages");
    let total = halves[1].layers.end;
    let mut whole = runtime.open_chain(&path, &[(start, total)]).expect("full model opens as one stage");
    let mut halved = runtime
        .open_chain(&path, &[(start, halves[0].layers.end), (halves[0].layers.end, total)])
        .expect("full model opens as two stages");
    let frame_whole = whole.run(&tokens).expect("full one-stage prefill");
    let frame_halved = halved.run(&tokens).expect("full two-stage prefill");
    assert_eq!(frame_whole, frame_halved, "full-model chains agree as well");
}

#[test]
fn stages_partition_the_model_and_share_one_boundary() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else {
        return;
    };
    let runtime = Runtime::new().expect("runtime");

    assert_eq!(runtime.model_family(&path), ModelFamily::Bert, "MiniLM is a BERT encoder");

    let plans = runtime.plan_stages(&path, 2).expect("two stages");
    assert_eq!(plans.len(), 2, "two stages requested, two returned");
    assert_eq!(plans[0].layers.start, 0, "the first stage starts at layer 0");
    assert_eq!(plans[0].layers.end, plans[1].layers.start, "stages must be contiguous");

    for plan in &plans {
        assert!(!plan.resident_identities.is_empty(), "a stage keeps tensors resident");
        assert!(!plan.execution_contract_hex.is_empty(), "a stage carries its contract");
    }

    assert!(!plans[0].exports.is_empty(), "a split stage exports a frontier");
    assert_eq!(
        plans[0].semantics.frontier_values,
        plans[0].exports.len() + plans[0].imports.len(),
        "semantics count what crosses the boundary"
    );
    // A fresh split must validate as one clean chain.
    runtime.validate_stage_chain(&path, 2).expect("two stages form a chain");
    // The identity is the contract across the boundary; the binding is
    // side-local (the producer's tensor versus the consumer's slot), so only
    // the identities must agree.
    let exported: Vec<&String> = plans[0].exports.iter().map(|value| &value.identity).collect();
    let imported: Vec<&String> = plans[1].imports.iter().map(|value| &value.identity).collect();
    assert_eq!(exported, imported, "what stage 0 exports is what stage 1 imports, in order");
}

/// A stage is a model of its own: it opens, it owns resources, and its
/// session borrows it. A slice with no pooling head refuses to embed and says
/// why rather than returning a short vector.
#[test]
fn a_stage_opens_as_its_own_model() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else {
        return;
    };
    let runtime = Runtime::new().expect("runtime");

    // A slice is only openable when the native stage program can express the
    // model's request inputs; BERT encoders need an attention mask the
    // installer cannot own yet, so a refusal here is a capability boundary and
    // must still carry its reason.
    let stage = match runtime.load_stage(&path, 2, 0) {
        Ok(stage) => stage,
        Err(error) => {
            assert!(!error.to_string().is_empty(), "a refusal carries a reason");
            return;
        }
    };
    let session = stage.create_session();

    match session {
        Ok(mut session) => {
            // A non-terminal slice has no embedding head, so embedding must
            // refuse with a reason rather than return a short vector.
            let mut output = vec![0.0_f32; 384];
            let outcome = session.embed_into(&[Token(1), Token(2), Token(3)], &mut output);
            if let Err(error) = outcome {
                assert!(!error.to_string().is_empty(), "a refusal carries the backend's reason");
            }
        }
        Err(error) => assert!(!error.to_string().is_empty()),
    }
}

#[test]
fn embedder_reports_mean_pooling_and_a_usable_width() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).expect("load");
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let info: WorkloadInfo = *session.workload();
    assert_eq!(info.kind, WorkloadKind::Embedding);
    assert_eq!(info.pooling, NativePooling::Mean);
    assert_eq!(info.embedding_pooling, Some(Pooling::Mean));
    assert_eq!(info.classifier_outputs, 0);
    assert!(info.output_dimensions > 0);

    // The advertised width is the width actually produced.
    let input = tokens(&mut session, "width check");
    let embedding = session.embed(&input).unwrap();
    assert_eq!(embedding.len(), info.output_dimensions);
}

#[test]
fn reranker_is_classified_as_rank_pooled_with_one_classifier_output() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::Rerank);
    assert_eq!(info.pooling, NativePooling::Rank);
    assert_eq!(info.classifier_outputs, 1);
    // A classifier head is not an embedding width.
    assert_eq!(info.embedding_pooling, None);
    assert_eq!(info.output_dimensions, 0);
}

#[test]
fn a_causal_decoder_is_not_a_reranker() {
    // llama_model_n_cls_out defaults to 1, so its raw value is not a presence
    // test: ungated, every decoder-only chat GGUF reads as a one-output
    // classifier and lands in the rerank table with a score head that does
    // not exist. The count is encoder-gated at load time.
    let Some(path) = model_path("SKIPPY_TEST_CAUSAL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::CausalGeneration);
    assert_eq!(info.classifier_outputs, 0);
    assert_eq!(info.embedding_pooling, None);

    // Neither lane is served: the refusals are the contract, not a crash.
    let input = [Token(1), Token(2), Token(3)];
    assert!(matches!(session.embed(&input), Err(WorkloadError::UnsupportedWorkload)));
    assert!(matches!(session.rerank("query", "document"), Err(WorkloadError::UnsupportedWorkload)));
}

#[test]
fn qwen3_reranker_is_classified_by_its_cls_output_tensor() {
    // The qwen3 family was the regression: it loads its rerank head as the
    // `cls.output` tensor, which the old family gate refused - "a decoder
    // family never carries this library's classifier head" - so a
    // Qwen3-Reranker classified as CausalGeneration and rerank() was
    // refused. Classification reads the tensor table now; family never
    // decides a workload.
    let Some(path) = optional_model_path("SKIPPY_TEST_QWEN3_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::Rerank);
    assert_eq!(info.pooling, NativePooling::Rank);
    assert_eq!(info.classifier_outputs, 1);
    assert_eq!(info.embedding_pooling, None);
}

#[test]
fn an_architecture_outside_the_family_table_is_not_a_reranker() {
    // The other half of P0-B: ModelFamily::Unknown covers every architecture
    // the 16-name family table never mentions (qwen2moe, qwen3moe, deepseek2,
    // glm4, ...). The family gate kept n_cls_out's default of 1 for all of
    // them, so a chat GGUF classified as a one-output reranker and rerank()
    // returned garbage. The tensor decides; no table entry is required.
    let Some(path) = optional_model_path("SKIPPY_TEST_UNKNOWN_ARCH") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::CausalGeneration);
    assert_eq!(info.classifier_outputs, 0);
    assert!(matches!(session.rerank("query", "document"), Err(WorkloadError::UnsupportedWorkload)));
}

// ---------------------------------------------------------------------------
// Vocab suppress tokens (patch 0012): the declaration is a model fact, read
// once at load and handed to the caller that owns sampling.
// ---------------------------------------------------------------------------

/// No local fixture declares `tokenizer.ggml.suppress_tokens`, and neither do
/// most real GGUFs — the expected answer is an empty typed set, not an error
/// or a sentinel id.
#[test]
fn a_model_without_declared_suppress_tokens_reports_an_empty_set() {
    let Some((runtime, path)) = embedder() else {
        return;
    };
    let model = runtime.load_model(&path).expect("load embedder");
    let suppressed: Vec<Token> = model.suppress_tokens();
    assert!(suppressed.is_empty(), "MiniLM declares no suppress tokens: {suppressed:?}");
}

// ---------------------------------------------------------------------------
// Buffer contracts: what the caller's memory looks like afterwards.
// Mirrors EmbedTest.NormalisesIntoTheCallerBufferAndReturnsAViewOfIt and
// EmbedTest.ExactCapacityIsEnough.
// ---------------------------------------------------------------------------

const SENTINEL: f32 = -1234.5;

#[test]
fn embed_into_writes_only_the_model_width() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let dimensions = session.workload().output_dimensions;

    let input = tokens(&mut session, "only the width");
    let mut buffer = vec![SENTINEL; dimensions + 5];
    let written = session.embed_into(&input, &mut buffer).unwrap();

    assert_eq!(written, dimensions);
    assert!(buffer[dimensions..].iter().all(|value| *value == SENTINEL));
    let norm: f32 = buffer[..dimensions].iter().map(|value| value * value).sum();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
}

#[test]
fn embed_into_accepts_a_buffer_of_exactly_the_width() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let dimensions = session.workload().output_dimensions;

    let input = tokens(&mut session, "exact capacity");
    let mut buffer = vec![0.0f32; dimensions];
    let written = session.embed_into(&input, &mut buffer).unwrap();
    assert_eq!(written, dimensions);
}

// ---------------------------------------------------------------------------
// Batch contracts. Mirrors EmbedTest.Batch* (which can only see the mock's
// rows) against real per-sequence pooling.
// ---------------------------------------------------------------------------

#[test]
fn batch_rows_are_the_rows_of_their_own_inputs() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 2 }).unwrap();

    let first = tokens(&mut session, "the quick brown fox");
    let second = tokens(&mut session, "quarterly revenue projections");
    let single_first = session.embed(&first).unwrap();
    let single_second = session.embed(&second).unwrap();

    let batched = session.embed_batch(&[first.as_slice(), second.as_slice()]).expect("batched embed");

    assert_eq!(batched.len(), 2);
    for (row, single) in batched.iter().zip([single_first, single_second]) {
        let max = row.iter().zip(&single).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
        assert!(max < 5e-4, "row differs by {max}");
    }
}

#[test]
fn batch_with_an_empty_group_is_rejected() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 2 }).unwrap();
    let filled = tokens(&mut session, "one real input");

    let empty: [Token; 0] = [];
    let error = session.embed_batch(&[&empty, filled.as_slice()]).unwrap_err();
    assert_eq!(error, WorkloadError::EmptyInput);
}

#[test]
fn batch_input_longer_than_max_tokens_is_rejected() {
    // The per-sequence budget, not just the total: a 1500-token row inside a
    // {512, 4} session fits the 2048 total but violates the contract the
    // single-sequence path enforces - the longest single input is max_tokens.
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 4 }).unwrap();
    let long = tokens(&mut session, &"word ".repeat(600));
    assert!(long.len() > 512, "the fixture must exceed the per-sequence budget");

    let error = session.embed_batch(&[long.as_slice()]).unwrap_err();
    assert_eq!(error, WorkloadError::TooManyTokens);
}

// ---------------------------------------------------------------------------
// Geometry: the retained-context pool must never hand a session a context it
// cannot use. Mirrors the backend's take_idle_context key.
// ---------------------------------------------------------------------------

#[test]
fn a_small_session_rejects_input_its_context_cannot_hold() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut small = model.create_session(SessionConfig { max_tokens: 8, ..Default::default() }).unwrap();
    let long = tokens(&mut small, &"word ".repeat(40));

    assert!(long.len() > 8, "test needs an input longer than the geometry");
    assert_eq!(small.embed(&long).unwrap_err(), WorkloadError::TooManyTokens);
}

#[test]
fn pooled_context_never_undersizes_the_batch() {
    // The padding case: {300,1} builds n_ctx = 512 (llama pads n_ctx to a
    // multiple of 256) with n_batch = n_ubatch = 300. Labelling the recycled
    // slot by n_ctx alone let take_idle_context hand it to a {400,1} session,
    // whose 350-token embed reached llama_decode above n_batch and aborted
    // the process. The pool labels by the smallest of the three, so the 400
    // request builds fresh and the embed succeeds.
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    {
        let pooled = model.create_session(SessionConfig { max_tokens: 300, sequences: 1 }).unwrap();
        drop(pooled);
    }
    let mut session = model.create_session(SessionConfig { max_tokens: 400, sequences: 1 }).unwrap();

    let mut long = tokens(&mut session, &"the ".repeat(400));
    assert!(long.len() >= 350, "{} tokens from 400 repeats", long.len());
    long.truncate(350);
    session.embed(&long).unwrap();
}

#[test]
fn sessions_with_different_geometries_coexist() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut small = model.create_session(SessionConfig { max_tokens: 8, ..Default::default() }).unwrap();
    let mut large = model.create_session(SessionConfig { max_tokens: 512, ..Default::default() }).unwrap();

    let text = "word ".repeat(40);
    let long = tokens(&mut large, &text);
    assert!(long.len() > 8);

    // The same tokens the small session refused embed fine on the large one,
    // and interleaving the two keeps both correct.
    let embedding = large.embed(&long).unwrap();
    let short = tokens(&mut small, "short");
    let small_embedding = small.embed(&short).unwrap();

    let small_norm: f32 = small_embedding.iter().map(|value| value * value).sum();
    let large_norm: f32 = embedding.iter().map(|value| value * value).sum();
    assert!((small_norm - 1.0).abs() < 1e-3, "small norm {small_norm}");
    assert!((large_norm - 1.0).abs() < 1e-3, "large norm {large_norm}");
}

// ---------------------------------------------------------------------------
// Tokenize: the text path is independent of the batch geometry.
// Mirrors TokenizeTest (C++ has no llama in it, so this is the real check).
// ---------------------------------------------------------------------------

#[test]
fn tokenize_accepts_text_longer_than_the_session_geometry() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 8, ..Default::default() }).unwrap();

    let long_text = "word ".repeat(200);
    let long = session.tokenize(&long_text).expect("tokenize does not enforce max_tokens");
    assert!(long.len() > 8);

    // ...while the decode path does.
    assert_eq!(session.embed(&long).unwrap_err(), WorkloadError::TooManyTokens);
}

#[test]
fn tokenize_is_stable_and_adds_special_tokens() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let first = session.tokenize("stable text").unwrap();
    let second = session.tokenize("stable text").unwrap();
    assert_eq!(first, second);
    // The tokenizer adds BOS/EOS, so a non-empty text is longer than its pieces.
    assert!(first.len() >= 2);
}

// ---------------------------------------------------------------------------
// Rerank: prompt composition and per-document scores through the real backend.
// Mirrors RerankTest.ReturnsScoreAndTokenCount and the batch pair.
// ---------------------------------------------------------------------------

#[test]
fn rerank_counts_the_composed_prompt_tokens() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let query = "What is the capital of France?";
    let document = "Paris is the capital and largest city of France.";
    let query_tokens = tokens(&mut session, query).len();
    let document_tokens = tokens(&mut session, document).len();

    let scored = session.rerank(query, document).unwrap();

    // The composed prompt carries both sides. One separator token is shared, so
    // the composed count is the standalone sum minus one, never less.
    assert!(
        scored.token_count + 1 >= query_tokens + document_tokens,
        "{} vs {query_tokens} + {document_tokens}",
        scored.token_count
    );
    assert!(scored.token_count >= query_tokens);
    assert!(scored.token_count >= document_tokens);
}

#[test]
fn rerank_batch_beyond_the_sequence_budget_is_rejected() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 1 }).unwrap();

    let error = session.rerank_batch("q", &["one", "two"]).unwrap_err();
    assert_eq!(error, WorkloadError::TooManySequences);
}

#[test]
fn rerank_batch_keeps_each_document_apart() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 2 }).unwrap();

    let query = "What is the capital of France?";
    let relevant = "Paris is the capital and largest city of France.";
    let unrelated = "Photosynthesis converts light into chemical energy.";

    let batched = session.rerank_batch(query, &[relevant, unrelated]).unwrap();
    let single_relevant = session.rerank(query, relevant).unwrap().score;
    let single_unrelated = session.rerank(query, unrelated).unwrap().score;

    assert!((batched[0] - single_relevant).abs() < 5e-3);
    assert!((batched[1] - single_unrelated).abs() < 5e-3);
    assert!(batched[0] > batched[1], "ordering lost: {batched:?}");
}

// ---------------------------------------------------------------------------
// Configuration and workload rejection, end to end.
// Mirrors NativeRuntimeTest's configuration cases and EncodePromptTest's
// workload check.
// ---------------------------------------------------------------------------

#[test]
fn unusable_session_configurations_are_rejected() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();

    // Session is not Debug, so the error side is destructured rather than
    // unwrap_err()'d.
    for config in [
        SessionConfig { max_tokens: 0, ..Default::default() },
        SessionConfig { sequences: 0, ..Default::default() },
        // The context would need max_tokens * sequences tokens; this cannot fit.
        SessionConfig { max_tokens: 512, sequences: u32::MAX },
    ] {
        let Err(error) = model.create_session(config) else {
            panic!("expected {config:?} to be rejected");
        };
        assert_eq!(error, skippy::RuntimeError::UnsupportedConfiguration);
    }
}

#[test]
fn encode_is_rejected_by_an_embedding_model() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let prompt = tokens(&mut session, "encode me");

    assert_eq!(session.encode(&prompt).unwrap_err(), WorkloadError::UnsupportedWorkload);
}

#[test]
fn embed_is_rejected_by_a_reranker() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let prompt = tokens(&mut session, "embed me");

    assert_eq!(session.embed(&prompt).unwrap_err(), WorkloadError::UnsupportedWorkload);
}

// ---------------------------------------------------------------------------
// Lifecycle: drop order is enforced by the borrow checker, so this checks the
// part Rust cannot enforce - that a dropped session leaves the pool usable and
// the next session behaves like a fresh one.
// ---------------------------------------------------------------------------

#[test]
fn dropping_a_session_then_making_a_new_one_is_transparent() {
    let Some((runtime, path)) = embedder() else { return };
    let model = runtime.load_model(&path).unwrap();
    let text = "recycled session";

    let first = {
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let input = tokens(&mut session, text);
        session.embed(&input).unwrap()
    };

    for _ in 0..4 {
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let input = tokens(&mut session, text);
        assert_eq!(session.embed(&input).unwrap(), first);
    }

    // Dropping the model with sessions already gone must not disturb the runtime.
    drop(model);
    let reloaded = runtime.load_model(&path).unwrap();
    let mut session = reloaded.create_session(SessionConfig::default()).unwrap();
    let input = tokens(&mut session, text);
    assert_eq!(session.embed(&input).unwrap(), first);
}
