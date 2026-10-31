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
