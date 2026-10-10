//! Snapshot export failures are observable even when best-effort callers ignore errors.
use std::sync::atomic::Ordering;

use anyhow::{Result, bail};
use skippy_cache::ExactStatePayload;

use super::KvStageIntegration;

fn require_snapshot_bytes(payload: &ExactStatePayload) -> Result<()> {
    if payload.byte_len() == 0 {
        bail!("snapshot exporter returned zero bytes; refusing to record missing model state");
    }
    Ok(())
}

impl KvStageIntegration {
    pub(super) fn validate_snapshot_export<T>(
        &self,
        exported: Result<(ExactStatePayload, T)>,
    ) -> Result<(ExactStatePayload, T)> {
        let result = exported.and_then(|(payload, extra)| {
            require_snapshot_bytes(&payload)?;
            Ok((payload, extra))
        });
        if result.is_err() {
            let failures = self
                .snapshot_export_failures
                .fetch_add(1, Ordering::Relaxed);
            // One warning per stage, every failure counted. Never include prompts,
            // session identifiers, or arbitrary backend error text in the warning.
            if failures == 0 {
                let _ = skippy_events::diagnostics::emit(
                    skippy_events::diagnostics::ServingDiagnostic::Warning {
                        message: "Skippy prefix snapshot export failed; no cache entry recorded".into(),
                        context: Some("inspect skippy.exact_cache.export_failures; native memory may not support the selected snapshot format".into()),
                    },
                );
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_byte_exports_are_errors_not_cache_entries() {
        assert!(
            require_snapshot_bytes(&ExactStatePayload::kv_recurrent(Vec::new(), Vec::new()))
                .is_err()
        );
        assert!(require_snapshot_bytes(&ExactStatePayload::full_state(Vec::new())).is_err());
        assert!(require_snapshot_bytes(&ExactStatePayload::full_state(vec![1])).is_ok());
    }
    #[test]
    fn rejected_exports_increment_counter_without_recording() {
        use skippy_protocol::{
            StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
        };
        let config = StageConfig {
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: StageKvCachePayload::FullState,
                max_entries: 4,
                max_bytes: 0,
                l2_max_bytes: 0,
                exact_max_bytes: None,
                codec: skippy_protocol::StageKvCacheCodec::Native,
                min_tokens: 1,
                shared_prefix_stride_tokens: 128,
                shared_prefix_record_limit: 2,
            }),
            ..Default::default()
        };
        let kv = KvStageIntegration::from_loaded_model(
            &config,
            Some(skippy_runtime::ModelStateKind::Hybrid),
            None,
            None,
        )
        .unwrap()
        .unwrap();
        for _ in 0..2 {
            assert!(
                kv.validate_snapshot_export(Ok((ExactStatePayload::full_state(Vec::new()), ())))
                    .is_err()
            );
        }
        assert!(
            kv.validate_snapshot_export::<()>(Err(anyhow::anyhow!("unsupported memory export")))
                .is_err()
        );
        assert_eq!(kv.snapshot_export_failures.load(Ordering::Relaxed), 3);
        assert_eq!(kv.exact_state_records_queued.load(Ordering::Relaxed), 0);
        assert_eq!(
            kv.attrs()
                .into_iter()
                .find(|(key, _)| *key == "skippy.exact_cache.export_failures")
                .unwrap()
                .1,
            serde_json::json!(3)
        );
    }
}
