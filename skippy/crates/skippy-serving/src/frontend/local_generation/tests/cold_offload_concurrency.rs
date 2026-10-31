//! Real large native export, paused disk I/O, concurrent decode and reload integrity.
//! Run with the sibling fixtures' explicit model/runtime environment and an
//! optimized host, preserving the same 30-second gate:
//! ```text
//! just with-lld cargo test --release -p skippy-serving \
//!   --features dynamic-native-runtime --lib large_native_cold_offload_ \
//!   -- --ignored --nocapture
//! ```
//! Debug runs remain useful timing diagnostics; an unoptimized large spill can
//! exceed this production-like deadline. Optional proof files use
//! `SKIPPY_RECURRENT_CACHE_TEST_ARTIFACT_DIR`.
use super::recurrent_fixture::recurrent_test_backend_with_lanes;
use super::*;
use sha2::{Digest, Sha256};
use skippy_cache::{L3CacheManager, StoreLimits};
use std::{fmt::Write as _, path::Path, sync::mpsc, time::Instant};

const PREFIX_TOKENS: usize = 23_497;
const PINNED_TOKENS: usize = 512;

struct DiskPause {
    pause: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<skippy_inference_api::InferenceResult<Duration>>>,
}

impl Drop for DiskPause {
    fn drop(&mut self) {
        // Release before joining even on a failed assertion/poll. Backend Drop
        // joins the disk worker and must never see an abandoned test pause.
        self.pause.store(false, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct PinnedCheckpoint {
    kv: Arc<KvStageIntegration>,
    namespace: String,
    tokens: Vec<i32>,
}

impl Drop for PinnedCheckpoint {
    fn drop(&mut self) {
        self.kv
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .release_resident(&self.namespace, &self.tokens);
    }
}

#[derive(serde::Serialize)]
struct LargeOffloadProof {
    prefix_tokens: usize,
    attention_bytes: usize,
    recurrent_bytes: usize,
    attention_sha256: String,
    recurrent_sha256: String,
    // Includes size queries, synchronous D2H/materialization and owner deletion.
    // This is a scheduler/runtime critical section, NOT disk-writing time.
    eviction_runtime_ms: f64,
    concurrent_decode_wait_ms: f64,
    baseline_decode_ms: f64,
    decode_completed_while_disk_paused: bool,
}

fn backend(root: &Path) -> Result<StageOpenAiBackend> {
    let (mut backend, _) = recurrent_test_backend_with_lanes(
        "large-native-cold-offload",
        "large-native-cold-offload",
        32_768,
        512,
        1,
        65_536,
        2,
    )?;
    let manager = L3CacheManager::acquire(root, StoreLimits::new(4 << 30, 0))?;
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
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        Some("large-native-prefix"),
        true,
        None,
    );
    let base = backend.local_kv_message_base("source", &ids);
    backend
        .kv
        .as_ref()
        .unwrap()
        .prefill_identity(&backend.config, &base, 0, tokens)
}

fn digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut output, "{byte:02x}").unwrap();
    }
    output
}

fn capture_large_owner(
    backend: &StageOpenAiBackend,
    tokens: &[i32],
) -> Result<(LargeOffloadProof, PinnedCheckpoint)> {
    let kv = backend.kv.as_ref().unwrap();
    let pinned = identity(backend, &tokens[..PINNED_TOKENS]);
    let large = identity(backend, tokens);
    let mut runtime = backend.runtime.lock().unwrap();
    runtime.prefill("source", &tokens[..PINNED_TOKENS])?;
    kv.record_resident_prefix(&mut runtime, "source", &pinned, &tokens[..PINNED_TOKENS])?
        .expect("small checkpoint captured");
    let acquired = kv
        .radix
        .lock()
        .unwrap()
        .acquire_resident_checkpoint(&pinned.namespace, &pinned.token_ids)
        .expect("small checkpoint pinned");
    let pin = PinnedCheckpoint {
        kv: kv.clone(),
        namespace: pinned.namespace,
        tokens: acquired.stored_tokens,
    };
    runtime.prefill("source", &tokens[PINNED_TOKENS..])?;
    let attention = runtime
        .export_kv_page("source", 0, tokens.len() as u64)?
        .payload;
    let recurrent = runtime.export_recurrent_state("source")?;
    anyhow::ensure!(
        attention.len() >= 700 * 1024 * 1024,
        "this gate needs a real large attention payload, got {} bytes",
        attention.len()
    );
    let proof = LargeOffloadProof {
        prefix_tokens: tokens.len(),
        attention_bytes: attention.len(),
        recurrent_bytes: recurrent.len(),
        attention_sha256: digest(&attention),
        recurrent_sha256: digest(&recurrent),
        eviction_runtime_ms: 0.0,
        concurrent_decode_wait_ms: 0.0,
        baseline_decode_ms: 0.0,
        decode_completed_while_disk_paused: false,
    };
    kv.record_resident_prefix(&mut runtime, "source", &large, tokens)?
        .expect("large native checkpoint captured");
    runtime.drop_session_timed("source")?;
    runtime.prefill("continuing-decode", &[21, 22, 23])?;
    runtime.ensure_session_active("eviction")?;
    Ok((proof, pin))
}

fn next_decode(backend: &StageOpenAiBackend, token: i32) -> Result<Duration> {
    let started = Instant::now();
    backend.iteration_scheduler.execute_iteration(
        "continuing-decode",
        &[token],
        &[],
        None,
        true,
        skippy_runtime::IterationBatchPhase::Decode,
    )?;
    Ok(started.elapsed())
}

fn offload_while_disk_is_paused(
    backend: &StageOpenAiBackend,
    tokens: &[i32],
    proof: &mut LargeOffloadProof,
) -> Result<()> {
    let kv = backend.kv.as_ref().unwrap();
    let stored = identity(backend, tokens);
    proof.baseline_decode_ms = next_decode(backend, 24)?.as_secs_f64() * 1000.0;
    kv.l3_spill_worker_pause.store(true, Ordering::Release);
    let mut paused = DiskPause {
        pause: kv.l3_spill_worker_pause.clone(),
        worker: None,
    };
    let scheduler = backend.iteration_scheduler.clone();
    let worker_kv = kv.clone();
    let (started_tx, started_rx) = mpsc::channel();
    paused.worker = Some(thread::spawn(move || {
        scheduler.execute_runtime("large-native-eviction", move |runtime| {
            let started = Instant::now();
            let _ = started_tx.send(());
            let eviction = worker_kv
                .evict_resident_prefix_for_tokens(runtime, "eviction", u64::MAX)
                .map_err(crate::frontend::util::openai_backend_error)?;
            assert_eq!(eviction.evicted_entries, 1, "the pinned owner must survive");
            assert_eq!(eviction.evicted_tokens, PREFIX_TOKENS as u64);
            Ok(started.elapsed())
        })
    }));
    started_rx.recv_timeout(Duration::from_secs(30))?;
    // Enqueue while eviction owns the runtime. Waiting for synchronous export
    // is expected and measured; progress DURING D2H is deliberately not asserted.
    proof.concurrent_decode_wait_ms = next_decode(backend, 25)?.as_secs_f64() * 1000.0;
    proof.eviction_runtime_ms = paused
        .worker
        .take()
        .unwrap()
        .join()
        .map_err(|_| anyhow::anyhow!("eviction worker panicked"))??
        .as_secs_f64()
        * 1000.0;
    let deadline = Instant::now() + Duration::from_secs(30);
    while kv.l3_spill_worker_received.load(Ordering::Acquire) == 0 {
        anyhow::ensure!(
            Instant::now() < deadline,
            "large spill never reached disk worker"
        );
        thread::sleep(Duration::from_millis(2));
    }
    assert!(paused.pause.load(Ordering::Acquire));
    assert_eq!(kv.l3_spill_worker_received.load(Ordering::Acquire), 1);
    assert!(
        kv.l3
            .as_ref()
            .unwrap()
            .locate_longest(&stored.namespace, tokens, 1)?
            .is_none()
    );
    assert_eq!(
        backend
            .runtime
            .lock()
            .unwrap()
            .canonical_session_position("continuing-decode")?,
        5
    );
    let stats = kv.radix.lock().unwrap().stats();
    assert_eq!(stats.resident_entries, 1);
    assert_eq!(stats.resident_pinned_tokens, PINNED_TOKENS as u64);
    assert_eq!(kv.inflight_records.lock().unwrap().len(), 1);
    proof.decode_completed_while_disk_paused = true;
    write_offload_artifact(
        "large-cold-offload-progress.json",
        &serde_json::json!({
            "phase": "decode_completed_disk_worker_still_paused",
            "proof": proof,
            "inflight_claims": kv.inflight_records.lock().unwrap().len(),
            "l3_status": kv.l3.as_ref().unwrap().status().map_err(|error| error.to_string()),
        }),
    )?;
    drop(paused); // Only now may disk hashing/writing start.
    wait_for_commit(kv, &stored.namespace, tokens)?;
    Ok(())
}

fn wait_for_commit(kv: &KvStageIntegration, namespace: &str, tokens: &[i32]) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let committed = kv
            .l3
            .as_ref()
            .unwrap()
            .locate_longest(namespace, tokens, 1)?
            .is_some_and(|entry| entry.token_count == tokens.len() as u64);
        if committed && kv.inflight_records.lock().unwrap().is_empty() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            let diagnostic = serde_json::json!({
                "phase": "commit_drain_deadline_failed",
                "committed_exact_prefix": committed,
                "inflight_claims": kv.inflight_records.lock().unwrap().len(),
                "disk_worker_received": kv.l3_spill_worker_received.load(Ordering::Acquire),
                "disk_worker_paused": kv.l3_spill_worker_pause.load(Ordering::Acquire),
                "l3_status": kv.l3.as_ref().unwrap().status().map_err(|error| error.to_string()),
            });
            write_offload_artifact("large-cold-offload-failure.json", &diagnostic)?;
            bail!("large spill did not commit/drain: {diagnostic}");
        }
        thread::sleep(Duration::from_millis(2));
    }
}

fn restore_after_restart(root: &Path, tokens: &[i32], proof: &LargeOffloadProof) -> Result<()> {
    let backend = backend(root)?;
    let kv = backend.kv.as_ref().unwrap();
    assert_eq!(kv.radix.lock().unwrap().stats().resident_entries, 0);
    let stored = identity(&backend, tokens);
    let mut runtime = backend.runtime.lock().unwrap();
    let restored = kv
        .restore_exact_state(&mut runtime, "restarted", &[stored])?
        .expect("fresh runtime must restore the committed large checkpoint");
    assert_eq!(restored.source, "l3");
    assert_eq!(restored.token_count, tokens.len());
    assert_eq!(
        runtime.canonical_session_position("restarted")?,
        tokens.len() as u64
    );
    let attention = runtime
        .export_kv_page("restarted", 0, tokens.len() as u64)?
        .payload;
    let recurrent = runtime.export_recurrent_state("restarted")?;
    assert_eq!(attention.len(), proof.attention_bytes);
    assert_eq!(recurrent.len(), proof.recurrent_bytes);
    assert_eq!(digest(&attention), proof.attention_sha256);
    assert_eq!(digest(&recurrent), proof.recurrent_sha256);
    assert!(kv.inflight_records.lock().unwrap().is_empty());
    assert_eq!(kv.radix.lock().unwrap().stats().resident_pinned_tokens, 0);
    Ok(())
}

fn write_offload_artifact(name: &str, value: &serde_json::Value) -> Result<()> {
    if let Some(directory) = std::env::var_os("SKIPPY_RECURRENT_CACHE_TEST_ARTIFACT_DIR") {
        std::fs::create_dir_all(&directory)?;
        std::fs::write(
            Path::new(&directory).join(name),
            serde_json::to_vec_pretty(value)?,
        )?;
    }
    Ok(())
}

fn run_large_offload_proof(root: &Path, tokens: &[i32]) -> Result<LargeOffloadProof> {
    let proof = {
        let backend = backend(root)?;
        let (mut proof, pin) = capture_large_owner(&backend, tokens)?;
        offload_while_disk_is_paused(&backend, tokens, &mut proof)?;
        drop(pin);
        assert_eq!(
            backend
                .kv
                .as_ref()
                .unwrap()
                .radix
                .lock()
                .unwrap()
                .stats()
                .resident_pinned_tokens,
            0
        );
        proof
    }; // Model, runtime, native owners and joined durable worker are gone.
    restore_after_restart(root, tokens, &proof)?;
    Ok(proof)
}

#[test]
#[ignore = "requires --release, 23497-token recurrent checkpoint >=700MiB attention and explicit model env"]
fn large_native_cold_offload_allows_decode_while_disk_is_paused_and_survives_restart() -> Result<()>
{
    let root = tempfile::tempdir()?;
    let tokens = (0..PREFIX_TOKENS)
        .map(|index| 32 + ((index * 17) % 4096) as i32)
        .collect::<Vec<_>>();
    let proof = match run_large_offload_proof(root.path(), &tokens) {
        Ok(proof) => proof,
        Err(error) => {
            return Err(error.context(format!(
                "large offload failure; retained L3 store {}",
                root.keep().display()
            )));
        }
    };
    write_offload_artifact(
        "large-cold-offload-proof.json",
        &serde_json::json!({
            "phase": "all_progress_persistence_restart_assertions_passed", "proof": proof,
        }),
    )?;
    Ok(())
}
