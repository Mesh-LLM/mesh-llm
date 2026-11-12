use super::*;
use crate::kv_integration::RadixResidentEntry;
use skippy_cache::{L3CacheManager, StoreLimits};
use skippy_protocol::{MessageBase, StageConfig, StageKvCacheConfig, StageKvCacheMode};
use skippy_runtime::{MemoryCacheCapabilities, ModelStateKind};
use std::path::Path;
#[cfg(unix)]
use {
    crate::frontend::iteration_scheduler::IterationScheduler,
    crate::telemetry::{Telemetry, TelemetryLevel},
    std::{sync::Mutex, time::Duration},
};

fn fixture(root: &Path) -> (KvStageIntegration, PrefillKvIdentity) {
    let manager = L3CacheManager::acquire(root, StoreLimits::new(1 << 20, 0)).unwrap();
    let config = StageConfig {
        layer_end: 1,
        source_model_sha256: Some("ab".repeat(32)),
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            min_tokens: 1,
            max_entries: 8,
            max_bytes: 0,
            l2_max_bytes: 1024,
            exact_max_bytes: None,
            codec: Default::default(),
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
            prefill_admission: Default::default(),
            retention: Default::default(),
        }),
        ..StageConfig::default()
    };
    let kv = KvStageIntegration::from_test_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        Some(MemoryCacheCapabilities {
            resident: true,
            resident_checkpoint: true,
            resident_attention: true,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .unwrap();
    let base: MessageBase = serde_json::from_value(serde_json::json!({
        "schema_version":1,"run_id":"run","request_id":"request","session_id":"session",
        "stage_id":"stage","stage_index":0,"topology_id":"topology"
    }))
    .unwrap();
    let identity = kv.prefill_identity(&config, &base, 0, &[1, 2, 3, 4, 5, 6, 7, 8]);
    (kv, identity)
}

fn spill(kv: &KvStageIntegration, identity: &PrefillKvIdentity) {
    let desc = super::super::tests::native_desc();
    kv.l3
        .as_ref()
        .unwrap()
        .spill(
            &identity.namespace,
            &identity.token_ids,
            &ExactStatePayload::kv_recurrent(vec![1; 256], vec![2; 16]),
            Some(serde_json::to_string(&desc).unwrap()),
            None,
        )
        .unwrap();
}

fn resident(kv: &KvStageIntegration, identity: &PrefillKvIdentity, count: usize, seq: i32) {
    kv.radix
        .lock()
        .unwrap()
        .insert_resident(
            &identity.namespace,
            &identity.token_ids[..count],
            16,
            RadixResidentEntry {
                page_id: format!("ram-{count}"),
                seq_id: seq,
                token_count: count as u64,
                recompute_cost: 1,
            },
        )
        .unwrap();
}

#[test]
fn committed_longer_l3_beats_short_ram_and_equal_ram_keeps_native_ownership() {
    let root = tempfile::tempdir().unwrap();
    let (kv, identity) = fixture(root.path());
    spill(&kv, &identity);
    let prepared = kv.prepare_exact_state_restore(std::slice::from_ref(&identity), false);
    for (count, seq, expected_empty) in [(4, 8, false), (8, 9, true)] {
        resident(&kv, &identity, count, seq);
        assert_eq!(
            kv.ordered_restore_candidates(&prepared).is_empty(),
            expected_empty
        );
    }
    assert_eq!(
        kv.radix.lock().unwrap().stats().resident_entries,
        2,
        "metadata selection must not evict the resident fallback"
    );
}

#[test]
fn prepared_miss_does_not_probe_a_newly_committed_disk_entry_under_runtime_lock() {
    let root = tempfile::tempdir().unwrap();
    let (mut kv, identity) = fixture(root.path());
    // Exercise the radix-miss fallback too, not only resident preference.
    kv.payload = StagePrefixCachePayload::KvRecurrent;
    let prepared = kv.prepare_exact_state_restore(std::slice::from_ref(&identity), false);
    assert!(prepared.locations[0].is_none());
    spill(&kv, &identity);
    let mut runtime = RuntimeState::new_modelless_for_test(1);
    assert!(
        kv.restore_prepared_exact_state(&mut runtime, "target", &prepared, None)
            .unwrap()
            .is_none(),
        "the completed miss must not discover/import the new disk entry"
    );
    assert_eq!(runtime.active_session_count(), 0);
}

#[cfg(unix)]
fn scheduler() -> IterationScheduler {
    let config = StageConfig::default();
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    IterationScheduler::new(runtime, &config, 8, true, None, telemetry).unwrap()
}

#[cfg(unix)]
fn paused_prefix_link(root: &Path) -> (std::path::PathBuf, Vec<u8>) {
    use std::os::unix::ffi::OsStrExt;
    let namespace = std::fs::read_dir(root.join("prefixes"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let path = std::fs::read_dir(namespace)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    // SAFETY: c_path is a valid NUL-terminated path in this test's private directory.
    assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    (path, bytes)
}

#[cfg(unix)]
fn wait_for_lookup_reader(path: &Path) -> std::fs::File {
    use std::os::unix::fs::OpenOptionsExt;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
        {
            Ok(writer) => return writer,
            Err(error) if error.raw_os_error() == Some(libc::ENXIO) => {
                assert!(
                    Instant::now() < deadline,
                    "lookup never opened its disk prefix link"
                );
                std::thread::yield_now();
            }
            Err(error) => panic!("open paused prefix link: {error}"),
        }
    }
}

#[test]
#[cfg(unix)]
fn stalled_disk_lookup_allows_unrelated_runtime_work_and_rechecks_growing_resident() {
    use std::{io::Write as _, sync::mpsc, thread};
    let root = tempfile::tempdir().unwrap();
    let (kv, identity) = fixture(root.path());
    spill(&kv, &identity);
    resident(&kv, &identity, 4, 8);
    let (path, bytes) = paused_prefix_link(root.path());
    let kv = Arc::new(kv);
    let request_kv = Arc::clone(&kv);
    let request_identity = identity.clone();
    let scheduler = scheduler();
    let request_scheduler = scheduler.clone();
    let lookup = thread::spawn(move || {
        let prepared = request_kv.prepare_exact_state_restore(&[request_identity], false);
        request_scheduler.execute_runtime("restore-after-disk-lookup", move |runtime| {
            let restored = request_kv
                .restore_prepared_exact_state(runtime, "target", &prepared, None)
                .map_err(|error| {
                    skippy_inference_api::InferenceError::backend(error.to_string())
                })?;
            Ok((prepared, restored.is_none()))
        })
    });
    // Writer-open succeeds only after locate_longest opens the actual FIFO.
    // Keep it empty: the real fs::read remains blocked, not a simulated sleep.
    let mut writer = wait_for_lookup_reader(&path);
    let (progress_tx, progress_rx) = mpsc::channel();
    let progress = thread::spawn(move || {
        let result = scheduler.execute_runtime("unrelated-during-disk-lookup", |runtime| {
            Ok(runtime.active_session_count())
        });
        progress_tx.send(result).unwrap();
    });
    let completed = progress_rx.recv_timeout(Duration::from_secs(30));
    resident(&kv, &identity, 8, 9);
    writer.write_all(&bytes).unwrap();
    drop(writer); // Release the disk reader even when the progress assertion fails.
    let (prepared, resident_won) = lookup.join().unwrap().unwrap();
    progress.join().unwrap();
    assert_eq!(
        completed.unwrap().unwrap(),
        0,
        "unrelated scheduler work must complete before disk lookup resumes"
    );
    assert!(
        kv.ordered_restore_candidates(&prepared).is_empty(),
        "the now-equal resident prefix must beat the earlier durable metadata"
    );
    assert!(
        resident_won,
        "the scheduled transaction must leave the native fallback intact"
    );
    assert_eq!(kv.radix.lock().unwrap().stats().resident_entries, 2);
}
