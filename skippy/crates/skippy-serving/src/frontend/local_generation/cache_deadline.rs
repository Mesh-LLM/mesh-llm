use std::time::{Duration, Instant};

/// Conservative single-stream prefill floor used to bound KV restore and
/// prefill-record work. Real Apple-Silicon prefill runs 200-400 tok/s, so an
/// eighth of that leaves headroom for slower stages without letting a
/// legitimate prompt-sized prefill die on the admission timeout.
const PREFILL_WORK_TOKENS_PER_SEC: f64 = 64.0;

/// Minimum and maximum prompt-scaled prefill work budget.
const PREFILL_WORK_MIN: Duration = Duration::from_secs(60);
const PREFILL_WORK_MAX: Duration = Duration::from_secs(30 * 60);

/// Deadline for one request's KV-restore + prefill-record work.
///
/// `generation_admission_timeout` bounds how long a request may wait to be
/// admitted (queue wait); it was never sized for the work itself. Cache-aware
/// operations run the whole prefill inside the scheduler worker, so a
/// prompt-sized prefill can legitimately take minutes on a large model. The
/// work deadline therefore adds a prompt-scaled budget on top of the admission
/// timeout while staying finite so a wedged runtime operation still fails
/// instead of holding a lane forever.
pub(super) fn cache_operation_deadline(
    admission_timeout: Duration,
    prompt_tokens: usize,
) -> Instant {
    let prompt_budget = Duration::from_secs_f64(prompt_tokens as f64 / PREFILL_WORK_TOKENS_PER_SEC)
        .clamp(PREFILL_WORK_MIN, PREFILL_WORK_MAX);
    let now = Instant::now();
    now.checked_add(admission_timeout.saturating_add(prompt_budget))
        .unwrap_or_else(|| now + PREFILL_WORK_MAX)
}

#[cfg(test)]
mod tests {
    use super::{PREFILL_WORK_MAX, cache_operation_deadline};
    use std::time::{Duration, Instant};

    #[test]
    fn cache_operation_deadline_scales_with_prompt_and_stays_bounded() {
        let admission = Duration::from_secs(60);

        // A small prompt still gets a full minute of work budget on top of the
        // admission wait, so short prefills never inherit the queue-timeout wall.
        let small = cache_operation_deadline(admission, 128);
        assert!(small.duration_since(Instant::now()) >= Duration::from_secs(119));

        // A 60k-token agentic-history prompt (the replay workload that failed at
        // 60s) earns admission + a prompt-scaled prefill budget, not 60s flat.
        let large = cache_operation_deadline(admission, 60_000);
        assert!(
            large.duration_since(Instant::now()) >= Duration::from_secs(60 + 60),
            "60k-token prompt must get more than the bare admission timeout"
        );

        // The work budget stays finite so a wedged operation still fails.
        let huge = cache_operation_deadline(admission, usize::MAX);
        assert!(
            huge.duration_since(Instant::now())
                <= admission + PREFILL_WORK_MAX + Duration::from_secs(5),
            "deadline must remain bounded"
        );
    }
}
