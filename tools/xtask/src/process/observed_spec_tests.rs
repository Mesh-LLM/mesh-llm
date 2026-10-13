use crate::process::*;
use std::time::Duration;

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(1),
        graceful_shutdown: Duration::from_millis(1),
        forced_shutdown: Duration::from_millis(1),
        retained_bytes_per_stream: 0,
        readiness: Readiness::ObservedLines {
            deadline: Duration::from_secs(1),
            matcher: |_| false,
        },
        completion: Completion::StopAfterReady,
    }
}

#[test]
fn migration_process_observed_spec_requires_stop_after_ready() {
    let mut limits = limits();
    limits.completion = Completion::Exit;
    assert!(matches!(limits.validate(), Err(Failure::InvalidSpec(_))));
}

#[test]
fn migration_process_observed_spec_bounds_deadline_by_execution() {
    for deadline in [
        Duration::ZERO,
        Duration::from_secs(1),
        Duration::from_secs(1) + Duration::from_nanos(1),
    ] {
        let mut limits = limits();
        limits.readiness = Readiness::ObservedLines {
            deadline,
            matcher: |_| false,
        };
        assert_eq!(
            limits.validate().is_ok(),
            deadline == Duration::from_secs(1)
        );
    }
}
