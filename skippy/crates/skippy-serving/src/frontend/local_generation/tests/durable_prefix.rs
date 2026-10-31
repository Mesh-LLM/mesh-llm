//! A fresh runtime must restore cold-evicted native state from committed L3.
use super::*;
use sha2::{Digest, Sha256};
use skippy_cache::{L3CacheManager, StoreLimits};
use skippy_protocol::MessageBase;
use std::fmt::Write as _;
use std::{path::Path, time::Instant};

fn backend_with_disk(root: &Path, disk_bytes: u64) -> Result<StageOpenAiBackend> {
    let (mut backend, _) = recurrent_test_backend(
        "native-durable-prefix",
        "native-durable-prefix",
        512,
        64,
        8,
        512,
    )?;
    let manager = L3CacheManager::acquire(root, StoreLimits::new(disk_bytes, 0))?;
    backend.kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &backend.config,
        crate::runtime_state::loaded_model_state_kind(Some(&backend.runtime)),
        crate::runtime_state::loaded_memory_cache_capabilities(Some(&backend.runtime)),
        Some(manager),
        None,
    )?
    .map(Arc::new);
    Ok(backend)
}

fn identity(
    backend: &StageOpenAiBackend,
    tokens: &[i32],
) -> crate::kv_integration::PrefillKvIdentity {
    let config = &backend.config;
    let base = MessageBase {
        schema_version: 1,
        run_id: config.run_id.clone(),
        request_id: "durable-request".into(),
        session_id: "source".into(),
        stage_id: config.stage_id.clone(),
        stage_index: config.stage_index,
        topology_id: config.topology_id.clone(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: None,
    };
    backend
        .kv
        .as_ref()
        .expect("native KV enabled")
        .prefill_identity(config, &base, 0, tokens)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn native_cold_prefix_survives_runtime_restart() -> Result<()> {
    let root = tempfile::tempdir()?;
    let expected = capture_and_commit(root.path())?;
    require_restored_prefix(root.path(), &expected)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn longer_durable_checkpoint_restores_despite_a_short_native_prefix() -> Result<()> {
    let root = tempfile::tempdir()?;
    let case = checkpoint_preference_case(root.path())?;
    let CheckpointPreferenceCase {
        backend,
        tokens,
        attention,
        recurrent,
        ..
    } = case;
    let kv = backend.kv.as_ref().unwrap();
    let long = identity(&backend, &tokens);
    let mut rt = backend.runtime.lock().unwrap();
    let location = kv
        .l3
        .as_ref()
        .unwrap()
        .locate_longest(&long.namespace, &tokens, 64)?
        .unwrap();
    assert_eq!(location.token_count, 64);
    let located_desc: skippy_runtime::RuntimeKvPageDesc =
        serde_json::from_str(location.kv_desc_json.as_deref().unwrap())?;
    located_desc.validate_payload(location.kv_bytes as usize)?;
    let candidates = [identity(&backend, &tokens[..16]), long];
    let restored = kv
        .restore_exact_state(&mut rt, "target", &candidates)?
        .unwrap_or_else(|| {
            panic!("a short RAM hit must not hide the longer L3 checkpoint: {location:?}")
        });
    assert_eq!(restored.source, "l3");
    assert_eq!(restored.token_count, tokens.len());
    assert_eq!(
        rt.export_kv_page("target", 0, tokens.len() as u64)?.payload,
        attention
    );
    assert_eq!(rt.export_recurrent_state("target")?, recurrent);
    Ok(())
}

struct CheckpointPreferenceCase {
    backend: StageOpenAiBackend,
    tokens: Vec<i32>,
    attention: Vec<u8>,
    recurrent: Vec<u8>,
    descriptor: skippy_runtime::RuntimeKvPageDesc,
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn failed_long_l3_import_rolls_back_then_reuses_the_short_native_owner() -> Result<()> {
    let root = tempfile::tempdir()?;
    let case = checkpoint_preference_case(root.path())?;
    let backend = case.backend;
    let kv = backend.kv.as_ref().unwrap();
    let long = identity(&backend, &case.tokens);
    // Correct attention metadata and bytes force the failure into recurrent import.
    kv.l3.as_ref().unwrap().spill(
        &long.namespace,
        &case.tokens,
        &skippy_cache::ExactStatePayload::kv_recurrent(case.attention, vec![0]),
        Some(serde_json::to_string(&case.descriptor)?),
        None,
    )?;
    {
        let mut rt = backend.runtime.lock().unwrap();
        assert!(
            kv.restore_exact_state(
                &mut rt,
                "bad-import-probe",
                &[identity(&backend, &case.tokens[..16]), long.clone()]
            )
            .is_err(),
            "fixture must reach a native import error, not an early metadata miss"
        );
        assert_eq!(rt.session_token_count("bad-import-probe"), Some(0));
        assert_eq!(rt.canonical_session_position("bad-import-probe")?, 0);
        rt.drop_session_timed("bad-import-probe")?;
        assert!(
            rt.canonical_session_position("bad-import-probe").is_err(),
            "an absent or discarded lane must not satisfy the clean retry gate"
        );
    }
    let mut prompt = case.tokens;
    prompt.push(11);
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("l3-failure-fallback"),
        true,
        None,
    );
    let sampling = SamplingConfig {
        enabled: false,
        ..Default::default()
    };
    let stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &prompt,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &SpeculativeDecodeConfig::default(),
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &ids,
        },
        |_| Ok(TokenControl::Continue),
    )?;
    assert_eq!(
        stats.cached_prompt_tokens, 32,
        "a failed longer durable restore must not discard the resident fallback"
    );
    assert_eq!(kv.radix.lock().unwrap().stats().resident_pinned_tokens, 0);
    Ok(())
}

fn checkpoint_preference_case(root: &Path) -> Result<CheckpointPreferenceCase> {
    let backend = backend_with_disk(root, 1 << 30)?;
    let kv = backend.kv.as_ref().unwrap();
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8].repeat(8);
    let long = identity(&backend, &tokens);
    let (attention, recurrent, descriptor) = {
        let mut rt = backend.runtime.lock().unwrap();
        rt.prefill("long-source", &tokens[..16])?;
        let shorter = rt.export_kv_page("long-source", 0, 16)?;
        let shorter_recurrent = rt.export_recurrent_state("long-source")?;
        kv.l3.as_ref().unwrap().spill(
            &long.namespace,
            &tokens[..16],
            &skippy_cache::ExactStatePayload::kv_recurrent(shorter.payload, shorter_recurrent),
            Some(serde_json::to_string(&shorter.desc)?),
            None,
        )?;
        rt.prefill("long-source", &tokens[16..])?;
        let page = rt.export_kv_page("long-source", 0, tokens.len() as u64)?;
        let attention = page.payload;
        let descriptor = page.desc;
        let recurrent = rt.export_recurrent_state("long-source")?;
        kv.record_resident_prefix(&mut rt, "long-source", &long, &tokens)?
            .unwrap();
        rt.drop_session_timed("long-source")?;
        rt.ensure_session_active("eviction")?;
        assert_eq!(
            kv.evict_resident_prefix_for_tokens(&mut rt, "eviction", u64::MAX)?
                .evicted_entries,
            1
        );
        rt.drop_session_timed("eviction")?;
        (attention, recurrent, descriptor)
    };
    let l3 = kv.l3.as_ref().unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while !l3
        .locate_longest(&long.namespace, &tokens, 64)?
        .is_some_and(|location| location.token_count == tokens.len() as u64)
    {
        anyhow::ensure!(
            Instant::now() < deadline,
            "long checkpoint was not committed"
        );
        thread::sleep(Duration::from_millis(2));
    }
    let short = identity(&backend, &tokens[..32]);
    let mut rt = backend.runtime.lock().unwrap();
    rt.prefill("short-source", &tokens[..32])?;
    kv.record_resident_prefix(&mut rt, "short-source", &short, &tokens[..32])?
        .unwrap();
    rt.drop_session_timed("short-source")?;
    assert_eq!(kv.probe_resident_prefix(&long).unwrap().token_count, 32);
    drop(rt);
    Ok(CheckpointPreferenceCase {
        backend,
        tokens,
        attention,
        recurrent,
        descriptor,
    })
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct PrefixReference {
    seed_process_id: u32,
    attention_sha256: String,
    recurrent_sha256: String,
    predictions: Vec<i32>,
}

fn digest(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut text, "{byte:02x}").unwrap();
    }
    text
}

fn capture_and_commit(root: &Path) -> Result<PrefixReference> {
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8].repeat(8);
    let inputs = [11, 12, 13, 14];
    let reference = {
        let backend = backend_with_disk(root, 1 << 30)?;
        let kv = backend.kv.as_ref().expect("disk cache enabled");
        let stored = identity(&backend, &tokens);
        let (attention, recurrent, output) = {
            let mut runtime = backend.runtime.lock().unwrap();
            runtime.prefill("source", &tokens)?;
            let attention = runtime
                .export_kv_page("source", 0, tokens.len() as u64)?
                .payload;
            let recurrent = runtime.export_recurrent_state("source")?;
            assert!(!attention.is_empty());
            assert!(!recurrent.is_empty());
            let record = kv
                .record_resident_prefix(&mut runtime, "source", &stored, &tokens)?
                .expect("native checkpoint captured");
            assert!(record.stored);
            let output = inputs
                .iter()
                .map(|&token| runtime.decode("source", token))
                .collect::<Result<Vec<_>>>()?;
            runtime.drop_session_timed("source")?;
            runtime.ensure_session_active("eviction")?;
            let evicted =
                kv.evict_resident_prefix_for_tokens(&mut runtime, "eviction", u64::MAX)?;
            assert_eq!(evicted.evicted_entries, 1);
            (attention, recurrent, output)
        };
        let l3 = kv.l3.as_ref().expect("persistent tier enabled");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(location) = l3.locate_longest(&stored.namespace, &tokens, 1)? {
                assert_eq!(location.token_count, tokens.len() as u64);
                break;
            }
            anyhow::ensure!(
                Instant::now() < deadline,
                "cold offload did not commit to L3"
            );
            thread::sleep(Duration::from_millis(2));
        }
        PrefixReference {
            seed_process_id: std::process::id(),
            attention_sha256: digest(&attention),
            recurrent_sha256: digest(&recurrent),
            predictions: output,
        }
    }; // Drop the model, runtime, cache and joined writer before reopening L3.
    Ok(reference)
}

fn require_restored_prefix(root: &Path, expected: &PrefixReference) -> Result<()> {
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8].repeat(8);
    let inputs = [11, 12, 13, 14];
    let backend = backend_with_disk(root, 1 << 30)?;
    let stored = identity(&backend, &tokens);
    let kv = backend
        .kv
        .as_ref()
        .expect("disk cache enabled after restart");
    let mut runtime = backend.runtime.lock().unwrap();
    let restored = kv
        .restore_exact_state(&mut runtime, "restarted", &[stored])?
        .expect("fresh runtime must restore from persistent cache");
    assert_eq!(restored.source, "l3");
    assert_eq!(restored.token_count, tokens.len());
    assert_eq!(
        runtime.canonical_session_position("restarted")?,
        tokens.len() as u64
    );
    assert_eq!(
        digest(
            &runtime
                .export_kv_page("restarted", 0, tokens.len() as u64)?
                .payload
        ),
        expected.attention_sha256,
        "attention bytes changed after L3 restore"
    );
    assert_eq!(
        digest(&runtime.export_recurrent_state("restarted")?),
        expected.recurrent_sha256,
        "recurrent bytes changed after L3 restore"
    );
    let actual = inputs
        .iter()
        .map(|&token| runtime.decode("restarted", token))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        actual, expected.predictions,
        "fixed-input continuation changed after L3 restore"
    );
    Ok(())
}

/// Invoked twice by the typed validator, which owns separate child processes.
#[test]
#[ignore = "requires the validator's private phase/root and pinned model environment"]
fn native_cold_prefix_process_stage() -> Result<()> {
    let root = std::path::PathBuf::from(
        std::env::var_os("SKIPPY_L3_TEST_ROOT")
            .ok_or_else(|| anyhow::anyhow!("SKIPPY_L3_TEST_ROOT is required"))?,
    );
    anyhow::ensure!(
        root.is_absolute() && root.is_dir(),
        "private L3 test root must exist"
    );
    let l3 = root.join("l3");
    let reference = root.join("reference.json");
    match std::env::var("SKIPPY_L3_TEST_PHASE")?.as_str() {
        "seed" => {
            anyhow::ensure!(
                !l3.exists() && !reference.exists(),
                "seed requires fresh L3 evidence"
            );
            let captured = capture_and_commit(&l3)?;
            std::fs::write(reference, serde_json::to_vec(&captured)?)?;
        }
        "restore" => {
            anyhow::ensure!(l3.is_dir(), "persisted L3 directory missing");
            anyhow::ensure!(
                std::fs::metadata(&reference)?.len() <= 4096,
                "L3 reference exceeds bound"
            );
            let expected: PrefixReference = serde_json::from_slice(&std::fs::read(reference)?)?;
            anyhow::ensure!(
                expected.seed_process_id != std::process::id(),
                "L3 restore requires a fresh process"
            );
            anyhow::ensure!(
                expected.predictions.len() == 4,
                "L3 reference lacks continuation predictions"
            );
            require_restored_prefix(&l3, &expected)?;
            std::fs::write(
                root.join("restart-receipt.json"),
                serde_json::to_vec(&serde_json::json!({
                    "status":"passed", "source":"l3", "seed_process_id":expected.seed_process_id,
                    "restore_process_id":std::process::id(), "state_integrity":"sha256_exact",
                    "fixed_input_predictions":expected.predictions
                }))?,
            )?;
        }
        _ => bail!("L3 process phase must be seed or restore"),
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn native_cold_offload_refusal_keeps_inference_usable() -> Result<()> {
    let root = tempfile::tempdir()?;
    let tokens = [1, 2, 3, 4, 5, 6, 7, 8].repeat(8);
    let inputs = [11, 12, 13, 14];
    let stored = {
        // One byte cannot admit either attention or recurrent checkpoint data.
        let backend = backend_with_disk(root.path(), 1)?;
        let kv = backend.kv.as_ref().expect("refusing disk tier enabled");
        let stored = identity(&backend, &tokens);
        let mut runtime = backend.runtime.lock().unwrap();
        runtime.prefill("source", &tokens)?;
        assert!(
            kv.record_resident_prefix(&mut runtime, "source", &stored, &tokens)?
                .is_some()
        );
        let expected = inputs
            .iter()
            .map(|&token| runtime.decode("source", token))
            .collect::<Result<Vec<_>>>()?;
        runtime.drop_session_timed("source")?;
        runtime.ensure_session_active("reuse")?;
        assert_eq!(
            kv.evict_resident_prefix_for_tokens(&mut runtime, "reuse", u64::MAX)?
                .evicted_entries,
            1
        );
        assert_eq!(
            runtime.memory_used_cells("reuse")?,
            0,
            "refused offload must not pin attention cells"
        );
        runtime.prefill("reuse", &tokens)?;
        let actual = inputs
            .iter()
            .map(|&token| runtime.decode("reuse", token))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            actual, expected,
            "cold recomputation changed after refused offload"
        );
        stored
    }; // Join the writer, including the refused write, before checking persistence.
    let backend = backend_with_disk(root.path(), 1)?;
    let kv = backend.kv.as_ref().expect("persistent tier enabled");
    assert!(
        kv.l3
            .as_ref()
            .unwrap()
            .locate_longest(&stored.namespace, &tokens, 1)?
            .is_none()
    );
    Ok(())
}
