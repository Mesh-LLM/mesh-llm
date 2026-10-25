use super::*;
use std::time::Duration;

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_millis(500),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 128,
        readiness: Readiness::Line {
            stream: Stream::Stdout,
            bytes: b"READY".to_vec(),
            deadline: Duration::from_millis(100),
        },
        completion: Completion::Exit,
    }
}

#[test]
fn migration_process_readiness_just_before_deadline_admits_poll() {
    let limits = limits();
    let observed = admit(
        &limits,
        false,
        (true, Duration::from_millis(100) - Duration::from_nanos(1)),
    );
    assert_eq!(observed, Ok(true));
}

#[test]
fn migration_process_readiness_at_deadline_rejects_poll() {
    let limits = limits();
    let observed = admit(&limits, false, (true, Duration::from_millis(100)));
    assert_eq!(observed, Err(Outcome::ReadinessDeadline));
}

#[test]
fn migration_process_readiness_just_after_deadline_rejects_poll() {
    let limits = limits();
    let observed = admit(
        &limits,
        false,
        (true, Duration::from_millis(100) + Duration::from_nanos(1)),
    );
    assert_eq!(observed, Err(Outcome::ReadinessDeadline));
}

#[test]
fn migration_process_readiness_accepted_before_deadline_survives_it() {
    let limits = limits();
    let observed = admit(&limits, true, (false, Duration::from_millis(101)));
    assert_eq!(observed, Ok(true));
}
