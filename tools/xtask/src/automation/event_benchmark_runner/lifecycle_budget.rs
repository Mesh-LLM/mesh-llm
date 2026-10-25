//! Shared early admission for retained host, measurement, and both owned-tree cleanup budgets.
use crate::command::DynResult;
use std::time::Duration;

pub(super) fn cell(
    readiness: Duration,
    request: Duration,
    shutdown: Duration,
) -> DynResult<(Duration, Duration)> {
    if readiness.is_zero()
        || request.is_zero()
        || shutdown.is_zero()
        || shutdown > Duration::from_secs(300)
    {
        return Err(
            "trial readiness/request must be positive and shutdown must be in 1..=300 seconds"
                .into(),
        );
    }
    // Each of two members reserves graceful stop, forced tree wait, and separate EOF drain.
    let cleanup = shutdown.checked_mul(6).ok_or("cleanup budget overflow")?;
    let execution = readiness
        .checked_mul(2)
        .and_then(|budget| {
            request
                .checked_mul(2)
                .and_then(|requests| budget.checked_add(requests))
        })
        .and_then(|budget| budget.checked_add(Duration::from_secs(1)))
        .ok_or("host and worker deadline overflow")?;
    execution
        .checked_add(cleanup)
        .ok_or("cell deadline overflow")?;
    Ok((execution, cleanup))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_readiness_windows_warmup_measurement_and_six_cleanup_windows_are_reserved() {
        let (execution, cleanup) = cell(
            Duration::from_secs(40),
            Duration::from_secs(10),
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(execution, Duration::from_secs(101));
        assert_eq!(cleanup, Duration::from_secs(6));
    }
    #[test]
    fn zero_oversized_shutdown_and_duration_overflow_are_rejected() {
        assert!(
            cell(
                Duration::from_secs(1),
                Duration::from_secs(1),
                Duration::ZERO
            )
            .is_err()
        );
        assert!(
            cell(
                Duration::from_secs(1),
                Duration::from_secs(1),
                Duration::from_secs(301)
            )
            .is_err()
        );
        assert!(
            cell(
                Duration::MAX,
                Duration::from_secs(1),
                Duration::from_secs(1)
            )
            .is_err()
        );
    }
}
