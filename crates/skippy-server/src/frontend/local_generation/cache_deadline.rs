use std::time::Duration;

/// Conservative single-stream prefill floor used to bound KV restore and
/// prefill-record work. Real Apple-Silicon prefill runs 200-400 tok/s, so an
/// eighth of that leaves headroom for slower stages without letting a
/// legitimate prompt-sized prefill die on the admission timeout.
const PREFILL_WORK_TOKENS_PER_SEC: f64 = 64.0;

/// Minimum and maximum prompt-scaled prefill work budget.
const PREFILL_WORK_MIN: Duration = Duration::from_secs(60);
const PREFILL_WORK_MAX: Duration = Duration::from_secs(30 * 60);

/// Stall timeout for one request's KV-restore + prefill-record work.
///
/// `generation_admission_timeout` bounds how long a request may wait to be
/// admitted. Cache-aware work uses this duration only after the scheduler
/// starts the operation, and renews it whenever a prefill chunk completes.
/// Taking the larger of the configured admission window and the prompt-scaled
/// floor leaves enough time for one slow native step without turning total
/// request age into a false wedge signal.
pub(super) fn cache_operation_stall_timeout(
    admission_timeout: Duration,
    prompt_tokens: usize,
) -> Duration {
    let prompt_budget = Duration::from_secs_f64(prompt_tokens as f64 / PREFILL_WORK_TOKENS_PER_SEC)
        .clamp(PREFILL_WORK_MIN, PREFILL_WORK_MAX);
    admission_timeout.max(prompt_budget).min(PREFILL_WORK_MAX)
}

#[cfg(test)]
mod tests {
    use super::{PREFILL_WORK_MAX, cache_operation_stall_timeout};
    use std::time::Duration;

    #[test]
    fn cache_operation_stall_timeout_scales_with_prompt_and_stays_bounded() {
        let admission = Duration::from_secs(60);

        let small = cache_operation_stall_timeout(admission, 128);
        assert_eq!(small, Duration::from_secs(60));

        let large = cache_operation_stall_timeout(admission, 60_000);
        assert_eq!(large, Duration::from_secs_f64(60_000.0 / 64.0));

        let huge = cache_operation_stall_timeout(admission, usize::MAX);
        assert_eq!(huge, PREFILL_WORK_MAX);

        let oversized_admission = cache_operation_stall_timeout(Duration::from_secs(60 * 60), 128);
        assert_eq!(oversized_admission, PREFILL_WORK_MAX);
    }
}
