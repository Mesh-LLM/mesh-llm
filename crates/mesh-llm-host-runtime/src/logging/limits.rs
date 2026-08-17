//! Live controls for the two logging settings whose schema permits dynamic apply.
//!
//! This module deliberately owns only retention and replay limits. Storage roots,
//! artifact capture, queue sizing, and every other logging setting remain
//! restart-required and must not mutate a running logging runtime.

use std::sync::{Mutex, RwLock};

/// The complete, coherent dynamic logging configuration seen by runtime readers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoggingDynamicLimits {
    pub retention_ttl_secs: u64,
    pub replay_capacity: usize,
}

impl LoggingDynamicLimits {
    pub fn from_config(config: &mesh_llm_config::LoggingConfig) -> Self {
        Self {
            retention_ttl_secs: config.retention_ttl_secs,
            replay_capacity: config.replay_capacity as usize,
        }
    }
}

/// A reader-friendly atomic projection of live logging limits.
///
/// A single lock protects both values: consumers either see the old pair or
/// the new pair, never a retention value from one apply with replay capacity
/// from another. The apply lock also serializes the paired bus mutation.
#[derive(Debug)]
pub struct DynamicLoggingLimits {
    snapshot: RwLock<LoggingDynamicLimits>,
    apply_lock: Mutex<()>,
}

impl DynamicLoggingLimits {
    pub fn new(initial: LoggingDynamicLimits) -> Self {
        Self {
            snapshot: RwLock::new(initial),
            apply_lock: Mutex::new(()),
        }
    }

    pub fn snapshot(&self) -> LoggingDynamicLimits {
        *self
            .snapshot
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Serialize a complete limit update with a caller-owned side effect.
    ///
    /// The side effect (replay capacity adjustment) runs before publication, so
    /// an error leaves the observable snapshot untouched. Callers must keep the
    /// side effect infallible or roll it back before returning an error.
    pub fn apply<E>(
        &self,
        next: LoggingDynamicLimits,
        update_replay: impl FnOnce(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let _apply = self
            .apply_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        update_replay(next.replay_capacity)?;
        *self
            .snapshot
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn concurrent_readers_observe_only_complete_limit_pairs() {
        let old = LoggingDynamicLimits {
            retention_ttl_secs: 3_600,
            replay_capacity: 128,
        };
        let new = LoggingDynamicLimits {
            retention_ttl_secs: 7_200,
            replay_capacity: 256,
        };
        let limits = Arc::new(DynamicLoggingLimits::new(old));
        let readers = (0..4)
            .map(|_| {
                let limits = Arc::clone(&limits);
                thread::spawn(move || {
                    for _ in 0..10_000 {
                        assert!(matches!(limits.snapshot(), value if value == old || value == new));
                    }
                })
            })
            .collect::<Vec<_>>();

        limits.apply(new, |_| Ok::<_, ()>(())).expect("apply");
        for reader in readers {
            reader.join().expect("reader thread");
        }
    }

    #[test]
    fn failed_side_effect_keeps_the_previous_pair_visible() {
        let old = LoggingDynamicLimits {
            retention_ttl_secs: 3_600,
            replay_capacity: 128,
        };
        let limits = DynamicLoggingLimits::new(old);
        let result = limits.apply(
            LoggingDynamicLimits {
                retention_ttl_secs: 7_200,
                replay_capacity: 256,
            },
            |_| Err::<(), _>("replay update failed"),
        );

        assert_eq!(result, Err("replay update failed"));
        assert_eq!(limits.snapshot(), old);
    }
}
