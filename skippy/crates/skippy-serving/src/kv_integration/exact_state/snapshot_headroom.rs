//! Host-memory admission for exact-state snapshots. On unified-memory hosts
//! the GPU state lives in host RAM, so exporting a snapshot that does not fit
//! the remaining headroom kills the process instead of degrading the cache.

use std::sync::atomic::Ordering;

use crate::runtime_state::RuntimeState;

use super::super::{KvLifecycleEvent, KvStageIntegration, StagePrefixCachePayload};
use super::{is_native_kv_unavailable, is_recurrent_unavailable};

/// Worst-case host copies of one snapshot while it is recorded: the native
/// export buffer, plus the blob store's standalone block copies when the new
/// payload's blocks collide with blocks already stored.
pub(crate) const SNAPSHOT_TRANSIENT_COPIES: u64 = 2;

/// Share of total host RAM kept free after recording (1/16, about 4 GiB on a
/// 64 GiB host) for decode growth and the allocations the export itself does
/// not account for.
pub(crate) const HOST_RESERVE_DIVISOR: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HostMemory {
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SnapshotAdmission {
    Admit,
    Skip {
        required_bytes: u64,
        available_bytes: u64,
    },
}

/// Available host memory a snapshot of `snapshot_bytes` needs before it may
/// be exported: `snapshot * SNAPSHOT_TRANSIENT_COPIES + total / HOST_RESERVE_DIVISOR`.
pub(crate) fn required_headroom_bytes(snapshot_bytes: u64, total_bytes: u64) -> u64 {
    snapshot_bytes
        .saturating_mul(SNAPSHOT_TRANSIENT_COPIES)
        .saturating_add(total_bytes / HOST_RESERVE_DIVISOR)
}

/// Unknown host memory keeps recording, as before.
pub(crate) fn admit_snapshot(snapshot_bytes: u64, host: Option<HostMemory>) -> SnapshotAdmission {
    let Some(host) = host else {
        return SnapshotAdmission::Admit;
    };
    let required_bytes = required_headroom_bytes(snapshot_bytes, host.total_bytes);
    if host.available_bytes >= required_bytes {
        SnapshotAdmission::Admit
    } else {
        SnapshotAdmission::Skip {
            required_bytes,
            available_bytes: host.available_bytes,
        }
    }
}

fn current_host_memory() -> Option<HostMemory> {
    skippy_hardware_profile::model_capacity::host_memory_bytes().map(
        |(total_bytes, available_bytes)| HostMemory {
            total_bytes,
            available_bytes,
        },
    )
}

fn warn(message: &str, context: String) {
    let _ =
        skippy_events::diagnostics::emit(skippy_events::diagnostics::ServingDiagnostic::Warning {
            message: message.to_string(),
            context: Some(context),
        });
}

impl KvStageIntegration {
    /// Sizes the snapshot through the native size-only queries, then checks it
    /// against host memory read now, before any state is copied. A size probe
    /// that fails admits the record (the export reports its own error), and
    /// says so once per stage so a host without admission is noticeable.
    pub(super) fn snapshot_fits_host_memory(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        payload: StagePrefixCachePayload,
        token_count: u64,
    ) -> bool {
        let snapshot_bytes = match self.snapshot_size(runtime, session_id, payload, token_count) {
            Ok(snapshot_bytes) => snapshot_bytes,
            Err(_) => {
                if self
                    .snapshot_size_probe_failures
                    .fetch_add(1, Ordering::Relaxed)
                    == 0
                {
                    warn(
                        "Prefix-cache snapshot size probe failed; host-memory admission is off for this stage",
                        format!(
                            "stage payload={payload:?}; snapshots are recorded without a host-memory check"
                        ),
                    );
                }
                return true;
            }
        };
        let SnapshotAdmission::Skip {
            required_bytes,
            available_bytes,
        } = admit_snapshot(snapshot_bytes, current_host_memory())
        else {
            return true;
        };
        self.exact_state_records_dropped
            .fetch_add(1, Ordering::Relaxed);
        self.notify_kv_lifecycle(KvLifecycleEvent::ExactStateRecordFailed);
        if self.snapshot_headroom_skips.fetch_add(1, Ordering::Relaxed) == 0 {
            warn(
                "Skipping prefix-cache snapshot: it would not fit in host memory",
                format!(
                    "snapshot_bytes={snapshot_bytes} token_count={token_count} \
                     required_available_bytes={required_bytes} \
                     available_bytes={available_bytes}; further skips on this stage are \
                     counted in skippy.exact_cache.records_dropped"
                ),
            );
        }
        false
    }

    /// Mirrors the export: a missing native KV page or a dense model's absent
    /// recurrent memory contributes no bytes; any other probe error is returned.
    fn snapshot_size(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        payload: StagePrefixCachePayload,
        token_count: u64,
    ) -> anyhow::Result<u64> {
        match payload {
            StagePrefixCachePayload::FullState => runtime.full_state_size(session_id),
            StagePrefixCachePayload::KvRecurrent => {
                let kv = match runtime.kv_page_size(session_id, 0, token_count) {
                    Ok(bytes) => bytes,
                    Err(error) if is_native_kv_unavailable(&error) => 0,
                    Err(error) => return Err(error),
                };
                let recurrent = match runtime.recurrent_state_size(session_id) {
                    Ok(bytes) => bytes,
                    Err(error)
                        if self.dense_without_recurrent && is_recurrent_unavailable(&error) =>
                    {
                        0
                    }
                    Err(error) => return Err(error),
                };
                Ok(kv.saturating_add(recurrent))
            }
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    fn host_bytes(total_bytes: u64, available_bytes: u64) -> Option<HostMemory> {
        Some(HostMemory {
            total_bytes,
            available_bytes,
        })
    }

    fn host(total_gib: u64, available_gib: u64) -> Option<HostMemory> {
        host_bytes(total_gib * GIB, available_gib * GIB)
    }

    #[test]
    fn headroom_reserves_two_snapshot_copies_and_a_sixteenth_of_host_ram() {
        assert_eq!(required_headroom_bytes(10 * GIB, 64 * GIB), 24 * GIB);
        assert_eq!(required_headroom_bytes(0, 64 * GIB), 4 * GIB);
    }

    #[test]
    fn snapshot_is_admitted_exactly_at_the_headroom_boundary() {
        assert_eq!(
            admit_snapshot(10 * GIB, host(64, 24)),
            SnapshotAdmission::Admit
        );
    }

    #[test]
    fn snapshot_is_skipped_one_byte_below_the_headroom_boundary() {
        assert_eq!(
            admit_snapshot(10 * GIB, host_bytes(64 * GIB, 24 * GIB - 1)),
            SnapshotAdmission::Skip {
                required_bytes: 24 * GIB,
                available_bytes: 24 * GIB - 1,
            }
        );
    }

    // Measured on a 64 GB Orin: ~24 GiB free after a 178K-token hybrid
    // prefill, and the end-of-prefill snapshot alone grew RSS by 12-20 GiB.
    #[test]
    fn long_prompt_snapshot_on_a_unified_memory_host_is_skipped() {
        assert_eq!(
            admit_snapshot(12 * GIB, host(61, 24)),
            SnapshotAdmission::Skip {
                required_bytes: required_headroom_bytes(12 * GIB, 61 * GIB),
                available_bytes: 24 * GIB,
            }
        );
    }

    #[test]
    fn short_prompt_snapshots_and_large_hosts_still_record() {
        assert_eq!(
            admit_snapshot(256 * MIB, host(61, 13)),
            SnapshotAdmission::Admit
        );
        assert_eq!(
            admit_snapshot(12 * GIB, host(512, 400)),
            SnapshotAdmission::Admit
        );
    }

    #[test]
    fn unknown_host_memory_keeps_recording() {
        assert_eq!(admit_snapshot(40 * GIB, None), SnapshotAdmission::Admit);
    }
}
