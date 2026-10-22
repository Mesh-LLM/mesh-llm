//! Attribute occupied native-call time only after the corresponding device work
//! has completed. This is service time, not a GPU kernel-duration measurement:
//! lock waits and gaps between calls are deliberately excluded.
use std::time::Duration;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompletedExecution {
    pub prompt_tokens: u64,
    pub decode_tokens: u64,
    pub elapsed: Duration,
}

impl CompletedExecution {
    fn append(&mut self, next: Self) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(next.prompt_tokens);
        self.decode_tokens = self.decode_tokens.saturating_add(next.decode_tokens);
        self.elapsed = self.elapsed.saturating_add(next.elapsed);
    }
}

#[derive(Default)]
pub(super) struct ExecutionAccounting {
    expected_epoch: Option<u64>,
    pending: CompletedExecution,
    contaminated: bool,
}

impl ExecutionAccounting {
    pub(super) fn finish(
        &mut self,
        before: Option<u64>,
        after: Option<u64>,
        work: CompletedExecution,
        success: bool,
    ) -> Option<CompletedExecution> {
        let (Some(before), Some(after)) = (before, after) else {
            self.invalidate();
            return None;
        };
        if self.expected_epoch != Some(before) {
            // A cache fence or another execution route may have consumed or
            // added work. Never carry its unknown cost into a pure-phase rate.
            self.pending = CompletedExecution::default();
            self.contaminated |= before & 1 != 0;
        }
        self.expected_epoch = Some(after);
        self.pending.append(work);
        self.contaminated |= !success;
        // A failure can leave partial work even when no successful submission
        // was recorded. Require a later successful completion before trusting
        // any timing again.
        if !success || after & 1 != 0 {
            return None;
        }
        let completed = std::mem::take(&mut self.pending);
        let contaminated = std::mem::take(&mut self.contaminated);
        (!contaminated).then_some(completed)
    }

    fn invalidate(&mut self) {
        self.expected_epoch = None;
        self.pending = CompletedExecution::default();
        self.contaminated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(prompt: u64, decode: u64, millis: u64) -> CompletedExecution {
        CompletedExecution {
            prompt_tokens: prompt,
            decode_tokens: decode,
            elapsed: Duration::from_millis(millis),
        }
    }

    #[test]
    fn deferred_prefill_is_not_charged_to_a_pure_decode_sample() {
        let mut accounting = ExecutionAccounting::default();
        assert_eq!(
            accounting.finish(Some(0), Some(3), work(512, 0, 42), true),
            None
        );
        let completed = accounting
            .finish(Some(3), Some(8), work(0, 1, 139), true)
            .unwrap();
        assert_eq!(completed, work(512, 1, 181));
        // The next genuinely independent decode is safe to learn from.
        assert_eq!(
            accounting.finish(Some(8), Some(12), work(0, 1, 35), true),
            Some(work(0, 1, 35))
        );
    }

    #[test]
    fn queued_prefill_chunks_complete_as_one_sample_without_queue_waits() {
        let mut accounting = ExecutionAccounting::default();
        assert_eq!(
            accounting.finish(Some(0), Some(3), work(512, 0, 5), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(3), Some(7), work(512, 0, 6), true),
            None
        );
        // No wall-clock timestamp spans the caller's queue/lock gaps.
        assert_eq!(
            accounting.finish(Some(7), Some(12), work(32, 0, 89), true),
            Some(work(1056, 0, 100))
        );
    }

    #[test]
    fn cache_fence_discards_old_work_before_learning_new_completed_work() {
        let mut accounting = ExecutionAccounting::default();
        accounting.finish(Some(0), Some(3), work(512, 0, 5), true);
        assert_eq!(
            accounting.finish(Some(6), Some(10), work(0, 4, 40), true),
            Some(work(0, 4, 40))
        );
    }

    #[test]
    fn unobserved_async_work_taints_until_completion() {
        let mut accounting = ExecutionAccounting::default();
        accounting.finish(Some(0), Some(3), work(512, 0, 5), true);
        assert_eq!(
            accounting.finish(Some(7), Some(11), work(512, 0, 8), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(11), Some(16), work(0, 1, 100), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(16), Some(20), work(0, 1, 30), true),
            Some(work(0, 1, 30))
        );
    }

    #[test]
    fn failed_partial_execution_never_trains_a_rate() {
        let mut accounting = ExecutionAccounting::default();
        assert_eq!(
            accounting.finish(Some(0), Some(3), work(512, 0, 10), false),
            None
        );
        assert_eq!(
            accounting.finish(Some(3), Some(8), work(0, 1, 100), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(8), Some(12), work(0, 1, 30), true),
            Some(work(0, 1, 30))
        );
    }

    #[test]
    fn failure_without_a_pending_marker_still_requires_a_successful_fence() {
        let mut accounting = ExecutionAccounting::default();
        assert_eq!(
            accounting.finish(Some(0), Some(0), work(512, 0, 10), false),
            None
        );
        assert_eq!(
            accounting.finish(Some(0), Some(4), work(0, 1, 100), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(4), Some(8), work(0, 1, 30), true),
            Some(work(0, 1, 30))
        );
    }

    #[test]
    fn missing_native_marker_does_not_fall_back_to_unsynchronized_timing() {
        let mut accounting = ExecutionAccounting::default();
        assert_eq!(accounting.finish(None, None, work(0, 1, 99), true), None);
        assert_eq!(
            accounting.finish(Some(4), Some(8), work(0, 1, 99), true),
            None
        );
        assert_eq!(
            accounting.finish(Some(8), Some(12), work(0, 1, 30), true),
            Some(work(0, 1, 30))
        );
    }
}
