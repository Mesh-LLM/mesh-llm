//! Exercises the full stack against real GGUF files. The C++ side has no
//! llama-dependent tests by design; this is where that coverage lives.
//!
//!   SKIPPY_TEST_MODEL     a BERT-family embedder GGUF (e.g. all-MiniLM-L6-v2;
//!                         end_to_end asserts the Bert family, so nomic/e5
//!                         GGUFs fail there and are not supported fixtures)
//!   SKIPPY_TEST_RERANKER  optional; a reranker with one classifier output
//!
//! Tests that need a file skip when its variable is unset or empty.

mod common;

use common::model_env;
use skippy::{Pooling, Runtime, RuntimeError, SessionConfig, Token, WorkloadError, WorkloadKind};

fn model_path(variable: &str) -> Option<String> {
    let path = model_env(variable);
    if path.is_none() {
        // Skipping keeps local runs green on a machine without GGUFs; CI sets
        // this so a missing model fails instead of silently reducing coverage.
        if model_env("SKIPPY_TEST_REQUIRE_MODELS").is_some() {
            panic!("{variable} is required when SKIPPY_TEST_REQUIRE_MODELS is set");
        }
        eprintln!("{variable} unset; skipping");
    }
    path
}

fn norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn missing_file_is_load_failed() {
    let runtime = Runtime::new().unwrap();
    let error = runtime.load_model("/nonexistent/model.gguf").err();
    // The payload is readable through the public Result: a native load
    // failure names the file it could not load, and a file the parser
    // rejects carries the parser's reason in this same field.
    match error {
        Some(RuntimeError::LoadFailed { message }) => {
            assert!(message.contains("/nonexistent/model.gguf"), "message was: {message}");
        }
        other => panic!("expected LoadFailed, got {other:?}"),
    }
}

#[test]
fn zero_max_tokens_is_unsupported_configuration() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();

    let error = model.create_session(SessionConfig { max_tokens: 0, ..Default::default() }).err();

    assert_eq!(error, Some(RuntimeError::UnsupportedConfiguration));
}

#[test]
fn embedding_model_is_classified_and_embeds_to_unit_length() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::Embedding);
    assert!(info.embedding_pooling.is_some(), "{info:?}");
    assert!(info.output_dimensions > 0, "{info:?}");

    let tokens = session.tokenize("The quick brown fox jumps over the lazy dog.").unwrap();
    assert!(!tokens.is_empty());

    let embedding = session.embed(&tokens).unwrap();
    assert_eq!(embedding.len(), info.output_dimensions);
    assert!((norm(&embedding) - 1.0).abs() < 1e-3, "norm {}", norm(&embedding));
}

#[test]
fn embedding_is_deterministic_and_text_sensitive() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let first = session.tokenize("a cat sat on the mat").unwrap();
    let second = session.tokenize("quarterly revenue projections").unwrap();

    let a = session.embed(&first).unwrap();
    let b = session.embed(&first).unwrap();
    let c = session.embed(&second).unwrap();

    assert_eq!(a, b);
    assert!(dot(&a, &c) < 0.999, "unrelated texts should not coincide: {}", dot(&a, &c));
}

#[test]
fn tokenize_reuses_the_callers_allocation() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let mut tokens: Vec<Token> = Vec::with_capacity(1024);
    let before = tokens.as_ptr();
    session.tokenize_into("short", &mut tokens).unwrap();
    session.tokenize_into("a somewhat longer piece of text", &mut tokens).unwrap();

    assert_eq!(tokens.as_ptr(), before);
    assert!(tokens.capacity() >= 1024);
}

#[test]
fn empty_input_is_rejected_before_the_backend() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    assert_eq!(session.embed(&[]).err(), Some(WorkloadError::EmptyInput));
    assert_eq!(session.encode(&[]).err(), Some(WorkloadError::EmptyInput));
}

#[test]
fn input_longer_than_max_tokens_is_rejected() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 8, ..Default::default() }).unwrap();

    let tokens = session.tokenize("one two three four five six seven eight nine ten eleven twelve").unwrap();
    assert!(tokens.len() > 8, "need more than 8 tokens, got {}", tokens.len());

    assert_eq!(session.embed(&tokens).err(), Some(WorkloadError::TooManyTokens));
}

#[test]
fn output_too_small_reports_the_required_width() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let width = session.workload().output_dimensions;

    let tokens = session.tokenize("hello").unwrap();
    let mut output = [0.0f32; 1];

    assert_eq!(session.embed_into(&tokens, &mut output).err(), Some(WorkloadError::OutputTooSmall { required: width }));
}

#[test]
fn embedding_model_rejects_other_workloads() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let tokens = session.tokenize("hello").unwrap();

    assert_eq!(session.rerank("q", "d").err(), Some(WorkloadError::UnsupportedWorkload));
    assert_eq!(session.encode(&tokens).err(), Some(WorkloadError::UnsupportedWorkload));
}

#[test]
fn several_sessions_share_a_model_and_release_in_any_order() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();

    let mut first = model.create_session(SessionConfig::default()).unwrap();
    let mut second = model.create_session(SessionConfig::default()).unwrap();
    let tokens = first.tokenize("shared").unwrap();
    let a = first.embed(&tokens).unwrap();
    drop(first);
    let b = second.embed(&tokens).unwrap();

    assert_eq!(a, b);
}

#[test]
fn sessions_are_usable_from_another_thread() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();
    let tokens = session.tokenize("threads").unwrap();
    let here = session.embed(&tokens).unwrap();

    let there = std::thread::scope(|scope| scope.spawn(|| session.embed(&tokens).unwrap()).join().unwrap());

    assert_eq!(here, there);
}

#[test]
fn batched_embeddings_match_single_embeddings() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();

    let texts = ["the quick brown fox", "quarterly revenue projections", "a cat sat on the mat"];
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 4 }).unwrap();

    let groups: Vec<Vec<Token>> = texts.iter().map(|text| session.tokenize(text).unwrap().to_vec()).collect();
    let batched = session.embed_batch(&groups.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
    assert_eq!(batched.len(), texts.len());
    assert_eq!(batched[0].len(), session.workload().output_dimensions);

    // Batched and single decodes take different kernel paths, so agreement is
    // numerical rather than bit-exact (measured ~1e-4 on a unit vector). A
    // pooling mistake would be orders of magnitude larger.
    for (index, group) in groups.iter().enumerate() {
        let single = session.embed(group).unwrap();
        let max = batched[index].iter().zip(&single).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
        assert!(max < 5e-4, "row {index}: max difference {max}");
    }
}

#[test]
fn batched_rerank_matches_single_scores() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 2 }).unwrap();

    let query = "What is the capital of France?";
    let documents =
        ["Paris is the capital and largest city of France.", "Photosynthesis converts light into chemical energy."];

    let batched = session.rerank_batch(query, &documents).unwrap();
    assert_eq!(batched.len(), documents.len());
    for (index, document) in documents.iter().enumerate() {
        let single = session.rerank(query, document).unwrap().score;
        assert!((batched[index] - single).abs() < 5e-3, "doc {index}: {} vs {single}", batched[index]);
    }
    assert!(batched[0] > batched[1], "{batched:?}");
}

#[test]
fn batch_beyond_the_session_sequence_budget_is_rejected() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig { max_tokens: 512, sequences: 1 }).unwrap();

    let tokens = session.tokenize("hello").unwrap().to_vec();
    let groups = [tokens.as_slice(), tokens.as_slice()];

    assert_eq!(session.embed_batch(&groups).err(), Some(WorkloadError::TooManySequences));
}

#[test]
fn released_sessions_do_not_leak_previous_state() {
    let Some(path) = model_path("SKIPPY_TEST_MODEL") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();

    let (first_text, second_text) = ("alpha beta gamma", "quarterly revenue projections");
    let (first, second) = {
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let a_tokens = session.tokenize(first_text).unwrap().to_vec();
        let a = session.embed(&a_tokens).unwrap();
        let b_tokens = session.tokenize(second_text).unwrap().to_vec();
        let b = session.embed(&b_tokens).unwrap();
        (a, b)
    };

    // A released session's context is retained and handed to the next session.
    // Neither the previous prompt nor its output may survive that handover.
    for _ in 0..8 {
        let mut session = model.create_session(SessionConfig::default()).unwrap();
        let a_tokens = session.tokenize(first_text).unwrap().to_vec();
        let a = session.embed(&a_tokens).unwrap();
        let b_tokens = session.tokenize(second_text).unwrap().to_vec();
        let b = session.embed(&b_tokens).unwrap();
        assert_eq!(a, first, "reused context changed the first text's embedding");
        assert_eq!(b, second, "reused context changed the second text's embedding");
    }
}

#[test]
fn reranker_scores_relevant_document_higher() {
    let Some(path) = model_path("SKIPPY_TEST_RERANKER") else { return };
    let runtime = Runtime::new().unwrap();
    let model = runtime.load_model(&path).unwrap();
    let mut session = model.create_session(SessionConfig::default()).unwrap();

    let info = *session.workload();
    assert_eq!(info.kind, WorkloadKind::Rerank);
    assert_eq!(info.classifier_outputs, 1);
    assert_eq!(info.embedding_pooling, None::<Pooling>);

    let query = "What is the capital of France?";
    let relevant = session.rerank(query, "Paris is the capital and largest city of France.").unwrap();
    let irrelevant = session.rerank(query, "Photosynthesis converts light into chemical energy.").unwrap();

    assert!(relevant.token_count > 0);
    assert!(relevant.score > irrelevant.score, "{relevant:?} vs {irrelevant:?}");
    assert_eq!(session.embed(&[Token(1)]).err(), Some(WorkloadError::UnsupportedWorkload));
}
