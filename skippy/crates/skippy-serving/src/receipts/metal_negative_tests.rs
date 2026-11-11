//! Opt-in native Q8-positive / Q6-execution-as-Q8-claim control.
//!
//! Requires same-model, single-file GGUFs in SKIPPY_METAL_CONTROL_Q8 and
//! SKIPPY_METAL_CONTROL_Q6, and a NEW directory SKIPPY_METAL_CONTROL_OUTPUT.
//! Dynamic builds also need MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR. Run this
//! ignored test alone, serially, on a qualified Metal device. Nothing is
//! downloaded. Evidence stays in the output directory; secret head keys do not.
//!
//! This deliberately constructs a recorder inside a test, below production
//! admission. It never changes `start`, a config path, or a quantization label
//! to make Q6 admissible. This is a native arithmetic control, not CLI/split
//! qualification. The positive must pass before the negative is attempted.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use skippy_receipt_audit::native_model::NativeClaimedModel;
use skippy_receipt_audit::outcome::{Check, Outcome};
use skippy_receipt_audit::output::AuditOutput;
use skippy_receipt_audit::recheck::recheck_evidence;
use skippy_receipt_audit::turn_audit::{
    AuditOptions, ClaimedSetup, OpenChallenge, Opened, OpenedRows, ReceivedTurn, TurnAudit,
    audit_turn, claimed_layer_count,
};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::audit::BackendFamily;
use skippy_receipts::evidence::Evidence;
use skippy_receipts::head::key::HeadKey;
use skippy_receipts::recorder::KvStreams;
use skippy_receipts::sampler_request::RequestedSampling;
use skippy_receipts::turn;
use skippy_receipts::vrf;
use skippy_receipts::weights::{LocalFile, WeightFile};
use skippy_receipts::wire::{Challenge, RECEIPT_VERSION, Receipt};
use skippy_runtime::{BackendDevice, SamplingConfig};

use super::{
    ReplyLimits, StageReceipts, TurnTokens, audited_backend, kv_cache, lock_runtime,
    metal_placement, metal_weights, native,
};
use crate::runtime_state::{RuntimeState, load_runtime};
use skippy_inference_api::ReceiptRequest;
use skippy_protocol::{LoadMode, StageConfig};

const SESSION: &str = "metal-numerical-control";

fn required_path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .with_context(|| format!("{name} is required for this ignored native test"))
}

fn weights(path: &Path) -> Result<WeightFile> {
    WeightFile::open(Box::new(LocalFile::open(path)?))
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let file = File::options().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

fn config(path: &Path, layers: u32) -> Result<StageConfig> {
    Ok(StageConfig {
        model_path: Some(path.to_str().context("GGUF path must be UTF-8")?.to_owned()),
        model_id: SESSION.into(),
        stage_id: "stage-0".into(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layers,
        load_mode: LoadMode::RuntimeSlice,
        ctx_size: 512,
        lane_count: 1,
        n_batch: Some(64),
        n_ubatch: Some(64),
        n_gpu_layers: -1,
        cache_type_k: "f16".into(),
        cache_type_v: "f16".into(),
        ..StageConfig::default()
    })
}

fn qualified_device(config: &StageConfig) -> Result<BackendDevice> {
    let devices = skippy_runtime::backend_devices()?;
    let device = metal_placement::device(config, &devices)
        .context("one unambiguous Metal device is required")?;
    ensure!(
        matches!(BackendFamily::of(&device.name), Some(BackendFamily::Metal)),
        "control must execute on Metal, not {}",
        device.name
    );
    for reason in [
        audited_backend::unaudited(&device.name, device.description.as_deref()),
        metal_placement::unaudited(config),
        kv_cache::unaudited(config),
    ] {
        ensure!(reason.is_none(), "unqualified control: {reason:?}");
    }
    Ok(device.clone())
}

/// Only test code can call this: production still checks the loaded Q6 file.
fn attach(
    runtime: &Arc<Mutex<RuntimeState>>,
    config: &StageConfig,
    backend: String,
    claimed: &str,
) -> Result<Arc<StageReceipts>> {
    let mut state = lock_runtime(runtime);
    let width = state
        .output_activation_boundary()
        .and_then(|boundary| boundary.primary_token_width("output").ok())
        .and_then(|width| usize::try_from(width).ok());
    let receipts = Arc::new(StageReceipts::new(
        0,
        config.layer_start..config.layer_end,
        backend,
        None,
        claimed.to_owned(),
        ed25519_dalek::SigningKey::from_bytes(&[71; 32]),
        ReplyLimits::for_width(width),
    ));
    receipts.set_replayer(Box::new(native::NativeReplayer::new(
        state.model.receipt_replayer(),
        Arc::downgrade(runtime),
        Arc::downgrade(&receipts),
    )));
    receipts.reports.set_native(state.model.receipt_flusher());
    let streams = state.model.receipt_replayer().kv_streams()?;
    receipts.reports.recorder().set_kv_streams(if streams > 1 {
        KvStreams::PerSequence
    } else {
        KvStreams::Shared
    });
    let recurrent = state
        .model
        .set_receipt_recorder(Arc::new(native::RecorderSink::new(&receipts)))?;
    receipts.reports.recorder().set_recurrent(recurrent);
    state.set_receipts(receipts.clone());
    // No HTTP listener or upkeep thread: this test owns the stage until audit ends.
    Ok(receipts)
}

fn compute_turn(
    runtime: &Arc<Mutex<RuntimeState>>,
    receipts: &StageReceipts,
    key: &vrf::SecretKey,
    id: &str,
) -> Result<Receipt> {
    let (session_turn, tokens) = {
        let mut state = lock_runtime(runtime);
        let prompt =
            "Explain why checking a numerical computation needs independent evidence. ".repeat(8);
        let prompt = state.model.tokenize(&prompt, true)?;
        ensure!(
            (17..256).contains(&prompt.len()),
            "unexpected prompt length"
        );
        let sampling = SamplingConfig::default(); // disabled sampling is greedy
        let mut token = state.prefill_chunked_sampled(SESSION, &prompt, Some(&sampling))?;
        for _ in 0..3 {
            token = state.decode_sampled(SESSION, token, Some(&sampling))?;
        }
        let turn = state.receipt_turn(SESSION).context("native receipt turn")?;
        let tokens = TurnTokens {
            generation_start: u32::try_from(prompt.len())?,
            final_token: Some(token),
            sampled: 4,
            sampler: turn::GREEDY.to_owned(),
        };
        (turn, tokens)
    };
    // Keep the runtime unlocked for the commit/open/replay lifecycle.
    receipts.finish_turn(
        id,
        SESSION,
        session_turn,
        tokens,
        Some(ReceiptRequest {
            client_key: key.public(),
            parent: None,
            seed: None,
        }),
    )
}

struct NativeNode(Arc<StageReceipts>);

impl OpenChallenge for NativeNode {
    fn open(&self, challenge: &Challenge) -> Result<Opened> {
        let started = Instant::now();
        let opening = self.0.open(challenge, None)?;
        let bytes = serde_json::to_vec(&opening)?;
        // Also records the opening digest for admission of the rows challenge.
        let signature = self.0.sign_opening(challenge, &bytes)?;
        Ok(Opened {
            opening,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn rows(&self, challenge: &BlockChallenge) -> Result<OpenedRows> {
        let started = Instant::now();
        let rows = self.0.rows(challenge, None)?;
        let bytes = serde_json::to_vec(&rows)?;
        let signature = self.0.sign_rows(challenge, &bytes)?;
        Ok(OpenedRows {
            rows,
            bytes,
            signature,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

/// Save even an unexpected result, before asserting anything about it.
fn save_audit(audit: TurnAudit, path: &Path) -> Result<(Box<AuditOutput>, Option<Box<Evidence>>)> {
    match audit {
        TurnAudit::Audited { output, evidence } => {
            write_json(path, &output)?;
            Ok((output, evidence))
        }
        TurnAudit::NoAudit(output) => {
            write_json(path, &output)?;
            bail!("no audit: {}", output.reason)
        }
        TurnAudit::Unanswered(output) => {
            write_json(path, &output)?;
            bail!("unanswered: {}", output.reason)
        }
    }
}

fn assert_verdict(output: &AuditOutput, negative: bool) -> Result<()> {
    ensure!(
        output.binding_error.is_none()
            && output.receipt_error.is_none()
            && output.opening_error.is_none()
            && output.request_errors.is_empty(),
        "control failed before arithmetic: {}",
        output.render_text()
    );
    ensure!(
        output.unchecked.is_empty(),
        "incomplete checks: {}",
        output.render_text()
    );
    if let Some(attention) = &output.attention {
        ensure!(
            attention.rows_error.is_none()
                && attention.rows_unanswered.is_none()
                && attention.rows_missing.is_none(),
            "invalid or incomplete attention evidence: {}",
            output.render_text()
        );
    }
    let expected = if negative {
        Outcome::Fail
    } else {
        Outcome::Pass
    };
    ensure!(output.outcome == expected, "{}", output.render_text());
    if negative {
        let matrix_failed = output
            .roles
            .iter()
            .any(|role| !role.pass && role.failing_tokens > 0);
        let logits_failed = output
            .head
            .as_ref()
            .and_then(Check::verdict)
            .is_some_and(|head| {
                head.positions
                    .iter()
                    .any(|position| position.logits_ok == Some(false))
            });
        ensure!(
            matrix_failed || logits_failed,
            "no numerical matrix-role/logit rejection: {}",
            output.render_text()
        );
    }
    Ok(())
}

fn run_control(q8: &Path, actual: &Path, out: &Path, negative: bool) -> Result<()> {
    let layers = claimed_layer_count(q8)?;
    ensure!(
        claimed_layer_count(actual)? == layers,
        "different executable layers"
    );
    let config = config(actual, layers)?;
    let device = qualified_device(&config)?;
    let refusal = metal_weights::unaudited(&config);
    ensure!(
        refusal.is_some() == negative,
        "unexpected production tensor admission: {refusal:?}"
    );
    // NamedTempFile is private (0600 on Unix), and removed on every exit path.
    let key_file = tempfile::NamedTempFile::new()?;
    let setup = ClaimedSetup {
        weights: weights(q8)?,
        model: Box::new(NativeClaimedModel::new(q8.to_owned())),
        layer_count: layers,
        head_key: key_file.path().to_owned(),
    };
    // Do expensive key preparation before the short live challenge window.
    // The audit reads this through its normal private key-cache loader.
    serde_json::to_writer(
        key_file.as_file(),
        &HeadKey::build(&setup.weights, rand::random())?,
    )?;
    let runtime = load_runtime(&config)?.context("native runtime did not load")?;
    let receipts = attach(
        &runtime,
        &config,
        device.name.clone(),
        q8.to_str().context("Q8 path")?,
    )?;
    write_json(
        &out.join("runtime.json"),
        &serde_json::json!({
            "backend": device.name,
            "device": device.description,
            "runtime": super::runtime_identity(),
            "production_tensor_refusal": refusal,
            "tensor_disable": std::env::var_os("GGML_METAL_TENSOR_DISABLE").map(|v| v.to_string_lossy().into_owned()),
            "scope": "single-stage native test; no production admission override"
        }),
    )?;
    let key = vrf::SecretKey::from_seed([93; 32]);
    let id = if negative { "q6-as-q8" } else { "q8-as-q8" };
    let receipt = compute_turn(&runtime, &receipts, &key, id)?;
    ensure!(
        receipt.version == 7 && receipt.version == RECEIPT_VERSION,
        "expected fresh v7 receipt"
    );
    turn::verify(&receipt, Some(&key.public()))?;
    let received = ReceivedTurn {
        receipt,
        key: &key,
        response: None,
        sampling: RequestedSampling::greedy(),
        asked_id: Some(id),
    };
    let audit = audit_turn(
        &setup,
        received,
        &NativeNode(receipts),
        &AuditOptions::default(),
    );
    let (output, evidence) = save_audit(audit, &out.join("audit.json"))?;
    let evidence = evidence.context("VRF-drawn signed evidence missing")?;
    write_json(&out.join("evidence.json"), &evidence)?;
    let layer_bytes = setup
        .weights
        .layer_bytes()
        .into_iter()
        .filter(|(layer, _)| *layer < layers)
        .collect();
    evidence
        .verify(&layer_bytes)
        .context("v7 evidence signatures and VRF proofs")?;
    let recheck = recheck_evidence(&evidence, &setup.weights, setup.model.as_ref(), layers);
    let (rechecked, _) = save_audit(recheck, &out.join("recheck.json"))?;
    assert_verdict(&output, negative)?;
    assert_verdict(&rechecked, negative)?;
    Ok(())
}

#[test]
#[ignore = "requires qualified Metal, native runtime, same-model Q8/Q6 GGUFs and a new output directory; run alone"]
fn metal_q8_positive_and_q6_as_q8_numerical_rejection() -> Result<()> {
    let q8 = required_path("SKIPPY_METAL_CONTROL_Q8")?.canonicalize()?;
    let q6 = required_path("SKIPPY_METAL_CONTROL_Q6")?.canonicalize()?;
    ensure!(q8 != q6, "Q8 and Q6 controls must be different files");
    let out = required_path("SKIPPY_METAL_CONTROL_OUTPUT")?;
    std::fs::create_dir(&out).context("output directory must be new; never overwrite evidence")?;
    let claimed = weights(&q8)?;
    let executed = weights(&q6)?;
    ensure!(
        executed
            .tensors
            .values()
            .any(|tensor| tensor.is_matrix() && tensor.ggml_type == 14),
        "Q6 control contains no Q6_K matrix"
    );
    ensure!(
        claimed.tensors.len() == executed.tensors.len(),
        "different tensor inventories"
    );
    for (name, tensor) in &claimed.tensors {
        ensure!(
            executed.tensor(name)?.dims == tensor.dims,
            "different shape for {name}"
        );
    }
    write_json(
        &out.join("artifacts.json"),
        &serde_json::json!({
            "q8": {"path": q8, "sha256": sha256(&q8)?},
            "q6": {"path": q6, "sha256": sha256(&q6)?},
            "receipt_version": RECEIPT_VERSION
        }),
    )?;
    skippy_runtime::test_support::require_native_runtime();
    for (name, actual, negative) in [("q8", q8.as_path(), false), ("q6", q6.as_path(), true)] {
        let dir = out.join(name);
        std::fs::create_dir(&dir)?;
        run_control(&q8, actual, &dir, negative).with_context(|| format!("{name} control"))?;
    }
    write_json(
        &out.join("completed.json"),
        &serde_json::json!({
            "q8_audit_and_recheck": "pass",
            "q6_as_q8_audit_and_recheck": "numerical rejection"
        }),
    )?;
    Ok(())
}
