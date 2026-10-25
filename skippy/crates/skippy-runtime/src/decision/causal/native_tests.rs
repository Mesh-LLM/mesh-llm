//! Model-backed proof against the existing sampled-token logprob reader.
//! Run explicitly with SKIPPY_DECISION_MODEL and --ignored; an absent model
//! is an error, never a passing certification.

use super::super::DecisionValue;
use super::*;
use crate::{
    IterationBatchPhase, IterationBatchRequest, LogitBias, RuntimeConfig, SamplingConfig,
    StageSession,
};
use anyhow::Context;

/// Open the real-model fixture with full-precision KV state.
fn open_model() -> Result<StageModel> {
    open_model_with_cache(crate::config::GGML_TYPE_F32)
}

/// Open a CPU causal model with explicit cache precision and batching parameters.
fn open_model_with_cache(cache_type: u32) -> Result<StageModel> {
    open_model_on_device(cache_type, "CPU")
}

/// Select a real backend explicitly so GPU readback tests cannot silently use CPU.
fn open_model_on_device(cache_type: u32, device: &str) -> Result<StageModel> {
    crate::test_support::require_native_runtime();
    let devices = crate::backend_devices()?;
    ensure!(
        devices.iter().any(|found| found.name == device),
        "required decision test device {device:?} is unavailable; available devices: {devices:?}"
    );
    let path = std::env::var_os("SKIPPY_DECISION_MODEL")
        .ok_or_else(|| anyhow::anyhow!("SKIPPY_DECISION_MODEL is required"))?;
    let microbatch = std::env::var("SKIPPY_DECISION_TEST_MICROBATCH")
        .map_or(Ok(1024), |value| value.parse::<u32>())?;
    let layers = crate::ModelInfo::open(&path)?
        .tensors()?
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .ok_or_else(|| anyhow::anyhow!("model has no transformer layers"))?
        + 1;
    StageModel::open(
        path,
        &RuntimeConfig {
            layer_end: layers,
            // Isolate score/cache correctness from reduced-precision KV and
            // batch-shape arithmetic. HTTP fixtures exercise normal F16 KV.
            flash_attn_type: crate::FlashAttentionType::Disabled,
            cache_type_k: cache_type,
            cache_type_v: cache_type,
            ctx_size: 4096,
            lane_count: 8,
            n_batch: Some(1024),
            n_ubatch: Some(microbatch),
            n_threads: Some(4),
            n_threads_batch: Some(4),
            kv_unified: Some(true),
            selected_backend_device: Some(device.into()),
            n_gpu_layers: if device == "CPU" { 0 } else { 999 },
            kv_offload: Some(device != "CPU"),
            op_offload: Some(device != "CPU"),
            ..RuntimeConfig::default()
        },
    )
}

/// Alternating prompts must not inherit the previous decode's host-logit buffer.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL and optionally SKIPPY_DECISION_TEST_DEVICE"]
fn causal_readout_reused_host_rows_match_synchronized_reference() -> Result<()> {
    let device = std::env::var("SKIPPY_DECISION_TEST_DEVICE").unwrap_or_else(|_| "CPU".into());
    let model = open_model_on_device(crate::config::GGML_TYPE_F32, &device)?;
    let prompts = [
        model.tokenize("The capital of France is", true)?,
        model.tokenize("The capital of Japan is", true)?,
    ];
    let selected = model.tokenize(" Paris", false)?;
    ensure!(!selected.is_empty());
    let candidates = vec![vec![selected[0]]];
    let expected = prompts
        .iter()
        .map(|prompt| reference_score(&model, prompt, &candidates[0]))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        (expected[0] - expected[1]).abs() > 1e-3,
        "fixture must distinguish stale prompt scores on this model"
    );
    for iteration in 0..64 {
        let index = iteration % prompts.len();
        let actual = read(&model, &prompts[index], &candidates)?[0];
        ensure!(
            (f64::from(actual) - expected[index]).abs() < 5e-4,
            "{device} iteration {iteration}: expected {}, got {actual}",
            expected[index]
        );
    }
    Ok(())
}

/// Construct independent single-token and multi-token questions over shared state.
fn request() -> DecisionRequest {
    DecisionRequest { state: DecisionValue::String("The customer requested a refund for an item delivered yesterday. Their question concerns billing, not shipping.".repeat(4)), seed: [0; 32], questions: vec![
        DecisionQuestion { key: "department".into(), instructions: Some(DecisionValue::String("Choose a department.".into())), kind: DecisionQuestionKind::Choice { options: vec![("B".into(), DecisionValue::String("billing".into())), ("S".into(), DecisionValue::String("shipping".into()))] } },
        DecisionQuestion { key: "action".into(), instructions: Some(DecisionValue::String("Choose the next action.".into())), kind: DecisionQuestionKind::Choice { options: vec![("contact billing support".into(), DecisionValue::String("billing".into())), ("track shipment".into(), DecisionValue::String("shipping".into()))] } },
    ] }
}

/// Reader planning preserves model admission, tokenizer limits and model lifetime.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real, pinned causal GGUF"]
fn causal_decision_reader_preserves_plan_and_outlives_model_handle() -> Result<()> {
    let model = open_model()?;
    let expected = DecisionPlan::for_model(&model, &request())?;
    let reader = model.reader();
    assert_eq!(
        reader.supports_causal_readout(),
        model.supports_causal_readout()
    );
    drop(model);
    assert_eq!(DecisionPlan::for_reader(&reader, &request())?, expected);
    let mut oversized = request();
    oversized.state = DecisionValue::String(" word".repeat(MAX_PROMPT_TOKENS + 1));
    let error = DecisionPlan::for_reader(&reader, &oversized).unwrap_err();
    assert!(error.to_string().contains("token limit"), "{error}");
    Ok(())
}

/// Score a candidate with constrained sampled-token logprobs as an independent oracle.
fn reference_score(model: &StageModel, prompt: &[i32], candidate: &[i32]) -> Result<f64> {
    let mut session = model.create_session()?;
    let mut score = 0.0;
    for (index, &token) in candidate.iter().enumerate() {
        let sampling = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            top_logprobs: Some(0),
            logit_bias: vec![LogitBias {
                token_id: token,
                bias: 1000.0,
            }],
            ..SamplingConfig::default()
        };
        let input = if index == 0 {
            prompt
        } else {
            &candidate[index - 1..index]
        };
        let chosen = if index == 0 {
            session
                .prefill_chunk_frame_sampled(input, Some(&sampling), None, 0)?
                .0
        } else {
            session.decode_step_sampled(input[0], Some(&sampling))?
        };
        ensure!(
            chosen == token,
            "constrained reference selected an unexpected token"
        );
        let records = session.take_token_logprobs()?;
        ensure!(
            records.len() == 1 && records[0].token == token,
            "missing independent logprob evidence"
        );
        score += f64::from(records[0].logprob);
    }
    Ok(score)
}

/// Read selected first-token scores without sampling or advancing a generated token.
fn read(model: &StageModel, prompt: &[i32], candidates: &[Vec<i32>]) -> Result<Vec<f32>> {
    let mut session = model.create_session()?;
    let selected = candidates
        .iter()
        .map(|candidate| candidate[0])
        .collect::<Vec<_>>();
    let (output, scores) = StageSession::iteration_batch_scored(
        &mut [IterationBatchRequest {
            session: &mut session,
            token_ids: prompt,
            positions: &[],
            sampling: None,
            input: None,
            sample_last: false,
            phase: IterationBatchPhase::Prefill,
        }],
        &[&selected],
    )?;
    ensure!(
        output.samples.is_empty() && session.take_token_logprobs()?.is_empty(),
        "readout must not sample"
    );
    ensure!(
        session.token_count() == prompt.len() as u64,
        "readout must not advance a generation token"
    );
    Ok(scores[0].clone())
}

/// Native readouts match the independent oracle across batching and prefix reuse.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL: a real, pinned causal GGUF"]
fn causal_readout_matches_constrained_reference_and_warm_batched_prefixes() -> Result<()> {
    let model = open_model()?;
    let plan = DecisionPlan::for_model(&model, &request())?;
    ensure!(plan.questions[0].method == ReadoutMethod::SelectedTokenLogits);
    ensure!(plan.questions[1].method == ReadoutMethod::CandidateLogLikelihood);
    verify_batched_prefix(&model, &plan)?;
    for question in &plan.questions {
        let scores = read(&model, &question.prompt, &question.candidates)?;
        for (candidate, score) in question.candidates.iter().zip(scores) {
            let reference = reference_score(&model, &question.prompt, &candidate[..1])?;
            ensure!(
                (reference - f64::from(score)).abs() < 2e-4,
                "prefill readout differs from reference: {reference} vs {score}"
            );
        }
    }
    verify_teacher_forcing(&model, &plan.questions[1])?;
    verify_readout_after_sampling(&model, &plan.questions[0])?;
    Ok(())
}

/// Prior sampled generation must not contaminate a later unsampled readout.
fn verify_readout_after_sampling(model: &StageModel, question: &QuestionPlan) -> Result<()> {
    let mut session = model.create_session()?;
    let sampling = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let first = StageSession::iteration_batch_sampled(&mut [IterationBatchRequest {
        session: &mut session,
        token_ids: &question.prompt,
        positions: &[],
        sampling: Some(&sampling),
        input: None,
        sample_last: true,
        phase: IterationBatchPhase::Prefill,
    }])?;
    let chosen = first
        .samples
        .first()
        .context("missing generation sample")?
        .predicted_token;
    session.take_token_logprobs()?;
    let selected = question.candidates.iter().map(|c| c[0]).collect::<Vec<_>>();
    let (output, scores) = StageSession::iteration_batch_scored(
        &mut [IterationBatchRequest {
            session: &mut session,
            token_ids: &[chosen],
            positions: &[],
            sampling: None,
            input: None,
            sample_last: false,
            phase: IterationBatchPhase::Prefill,
        }],
        &[&selected],
    )?;
    ensure!(output.samples.is_empty() && session.take_token_logprobs()?.is_empty());
    ensure!(session.token_count() == question.prompt.len() as u64 + 1);
    drop(session);
    let reference = open_model()?;
    for (&token, &score) in selected.iter().zip(&scores[0]) {
        // Preserve the prefill/decode boundary: evaluating the entire prefix
        // as one batch selects different quantized matrix kernels.
        let mut independent = reference.create_session()?;
        prefill(&mut independent, &question.prompt)?;
        let sampling = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            top_logprobs: Some(0),
            logit_bias: vec![LogitBias {
                token_id: token,
                bias: 1000.0,
            }],
            ..SamplingConfig::default()
        };
        ensure!(independent.decode_step_sampled(chosen, Some(&sampling))? == token);
        let records = independent.take_token_logprobs()?;
        ensure!(records.len() == 1 && records[0].token == token);
        let expected = f64::from(records[0].logprob);
        ensure!(
            (expected - f64::from(score)).abs() < 2e-4,
            "reused sampler changed raw readout: {expected} vs {score}"
        );
    }
    Ok(())
}

/// Teacher-forced candidate scores equal the sum of independent conditional logprobs.
fn verify_teacher_forcing(model: &StageModel, question: &QuestionPlan) -> Result<()> {
    for candidate in &question.candidates {
        let mut session = model.create_session()?;
        let mut total = 0.0;
        for (index, token) in candidate.iter().enumerate() {
            let input = if index == 0 {
                question.prompt.as_slice()
            } else {
                &candidate[index - 1..index]
            };
            let (output, scores) = StageSession::iteration_batch_scored(
                &mut [IterationBatchRequest {
                    session: &mut session,
                    token_ids: input,
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: false,
                    phase: IterationBatchPhase::Prefill,
                }],
                &[std::slice::from_ref(token)],
            )?;
            ensure!(output.samples.is_empty());
            total += f64::from(scores[0][0]);
        }
        drop(session);
        let reference = reference_score(model, &question.prompt, candidate)?;
        ensure!(
            (reference - total).abs() < 5e-4,
            "teacher-forced sum differs from reference: {reference} vs {total}, candidate={candidate:?}"
        );
    }
    Ok(())
}

/// Warm shared prefixes preserve each batched question's independent scores.
fn verify_batched_prefix(model: &StageModel, plan: &DecisionPlan) -> Result<()> {
    // Native execution and verification lanes occupy the first 2 * lane_count IDs.
    const PREFIX_SEQUENCE: i32 = 16;
    let prefix = &plan.questions[0].prompt[..plan.shared_prefix_tokens];
    let mut source = model.create_session()?;
    prefill(&mut source, prefix)?;
    source.save_prefix(PREFIX_SEQUENCE, prefix.len() as u64)?;
    let mut sessions = plan
        .questions
        .iter()
        .map(|_| model.create_session_from_resident_prefix(PREFIX_SEQUENCE, prefix))
        .collect::<Result<Vec<_>>>()?;
    let selections = plan
        .questions
        .iter()
        .map(|q| q.candidates.iter().map(|c| c[0]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut requests = sessions
        .iter_mut()
        .zip(&plan.questions)
        .map(|(session, question)| IterationBatchRequest {
            session,
            token_ids: &question.prompt[prefix.len()..],
            positions: &[],
            sampling: None,
            input: None,
            sample_last: false,
            phase: IterationBatchPhase::Prefill,
        })
        .collect::<Vec<_>>();
    let (output, scores) = StageSession::iteration_batch_scored(
        &mut requests,
        &selections.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )?;
    ensure!(output.samples.is_empty() && scores.len() == 2);
    drop(requests);
    drop(sessions);
    source.drop_sequence(PREFIX_SEQUENCE)?;
    drop(source);
    // Preserve batch shape and chunk boundaries to isolate KV reuse from
    // different BLAS / quantized matmul arithmetic at different batch sizes.
    let reference_model = open_model()?;
    let mut references = plan
        .questions
        .iter()
        .map(|_| {
            let mut session = reference_model.create_session()?;
            prefill(&mut session, prefix)?;
            Ok(session)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut reference_requests = references
        .iter_mut()
        .zip(&plan.questions)
        .map(|(session, question)| IterationBatchRequest {
            session,
            token_ids: &question.prompt[prefix.len()..],
            positions: &[],
            sampling: None,
            input: None,
            sample_last: false,
            phase: IterationBatchPhase::Prefill,
        })
        .collect::<Vec<_>>();
    let (_, cold) = StageSession::iteration_batch_scored(
        &mut reference_requests,
        &selections.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )?;
    for (warm, cold) in scores.iter().flatten().zip(cold.iter().flatten()) {
        ensure!(
            (*warm - cold).abs() < 2e-4,
            "warm batch differs from recomputed prefix: {warm} vs {cold}; all warm={scores:?}"
        );
    }
    Ok(())
}

/// Prefill a session without sampling a token for subsequent score comparisons.
fn prefill(session: &mut StageSession, tokens: &[i32]) -> Result<()> {
    let output = StageSession::iteration_batch_sampled(&mut [IterationBatchRequest {
        session,
        token_ids: tokens,
        positions: &[],
        sampling: None,
        input: None,
        sample_last: false,
        phase: IterationBatchPhase::Prefill,
    }])?;
    ensure!(output.samples.is_empty());
    Ok(())
}

/// Readout scores track the native oracle across prefill chunk shapes.
#[test]
#[ignore = "requires SKIPPY_DECISION_MODEL: compares native arithmetic across chunk shapes"]
fn causal_readout_tracks_native_reference_across_prefill_shapes() -> Result<()> {
    let model = open_model_with_cache(crate::config::GGML_TYPE_F16)?;
    let plan = DecisionPlan::for_model(&model, &request())?;
    let mut evidence = Vec::new();
    for question in &plan.questions {
        for chunk_size in [1, 32, 128, 1024] {
            let chunks = question.prompt.chunks(chunk_size).collect::<Vec<_>>();
            let (last, prefix) = chunks.split_last().context("nonempty prompt")?;
            let selected = question.candidates.iter().map(|c| c[0]).collect::<Vec<_>>();
            let mut session = model.create_session()?;
            for chunk in prefix {
                prefill(&mut session, chunk)?;
            }
            let (_, scores) = StageSession::iteration_batch_scored(
                &mut [IterationBatchRequest {
                    session: &mut session,
                    token_ids: last,
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: false,
                    phase: IterationBatchPhase::Prefill,
                }],
                &[&selected],
            )?;
            drop(session);
            for (&token, &score) in selected.iter().zip(&scores[0]) {
                let mut reference = model.create_session()?;
                for chunk in prefix {
                    prefill(&mut reference, chunk)?;
                }
                let sampling = SamplingConfig {
                    enabled: true,
                    temperature: 0.0,
                    top_logprobs: Some(0),
                    logit_bias: vec![LogitBias {
                        token_id: token,
                        bias: 1000.0,
                    }],
                    ..SamplingConfig::default()
                };
                ensure!(
                    reference
                        .prefill_chunk_frame_sampled(last, Some(&sampling), None, 0)?
                        .0
                        == token
                );
                let records = reference.take_token_logprobs()?;
                ensure!(records.len() == 1 && records[0].token == token);
                let expected = records[0].logprob;
                ensure!(
                    (score - expected).abs() < 2e-4,
                    "chunk={chunk_size}: selected readout {score} differs from native reference {expected}"
                );
                evidence.push(
                    serde_json::json!({"question": question.key, "chunk_size": chunk_size,
                    "token": token, "readout": score, "native_reference": expected}),
                );
            }
        }
    }
    if let Some(path) = std::env::var_os("SKIPPY_DECISION_NUMERICS_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence)?)?;
    }
    Ok(())
}
