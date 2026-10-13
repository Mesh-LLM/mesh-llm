use super::*;

struct CompletionOwner {
    daemon: Option<Launch>,
    command: Option<Launch>,
    policy: ExpectedExit,
    completed: bool,
}

impl Coordinator for CompletionOwner {
    type Rejection = ();
    fn line(&mut self, _: MemberId, line: ObservedLine<'_>) -> ProbeDecision<()> {
        if line.bytes == b"READY" {
            ProbeDecision::Candidate
        } else {
            ProbeDecision::Pending
        }
    }
    fn tick(&mut self, context: Context<'_>) -> Action<()> {
        if let Some(launch) = self.daemon.take() {
            return Action::Start(launch);
        }
        if context
            .members
            .iter()
            .any(|member| matches!(member.state, MemberState::Starting))
            && context.members.len() == 1
        {
            return Action::Pending;
        }
        if let Some(launch) = self.command.take() {
            return Action::StartExpected {
                launch,
                policy: self.policy.clone(),
            };
        }
        self.completed = context
            .members
            .iter()
            .any(|member| matches!(member.state, MemberState::ExpectedExit { status: 23, .. }));
        if self.completed {
            Action::Complete
        } else {
            Action::Pending
        }
    }
}

fn owner(
    daemon: &std::path::Path,
    command: &std::path::Path,
    statuses: &[i32],
    mode: &str,
    deadline: Duration,
) -> CompletionOwner {
    let mut expected = launch(command, MemberId::WorkerOne, mode);
    expected
        .spec
        .environment
        .insert("MIGRATION_PROCESS_MODE".into(), Value::Public(mode.into()));
    if mode == "retained-crash" {
        std::fs::write(command.join("crash-now"), b"crash").unwrap();
    }
    CompletionOwner {
        daemon: Some(launch(daemon, MemberId::Seed, "ready-hang")),
        command: Some(expected),
        policy: ExpectedExit::new(statuses, deadline).unwrap(),
        completed: false,
    }
}

#[test]
fn declared_nonzero_exit_completes_without_stopping_live_daemon_early() {
    let daemon = tempfile::tempdir().unwrap();
    let command = tempfile::tempdir().unwrap();
    let mut coordinator = owner(
        daemon.path(),
        command.path(),
        &[23],
        "retained-crash",
        Duration::from_secs(2),
    );
    let report = run(&mut coordinator, &limits(), &Cancellation::default()).unwrap();
    assert!(report.success());
    assert!(coordinator.completed);
    let completed = &report.members[1];
    assert!(matches!(completed.disposition, Disposition::ExpectedExit));
    let receipt = completed.completion.as_ref().unwrap();
    assert_eq!(receipt.status, Some(23));
    assert!(receipt.accepted(&completed.process));
    assert_stopped(daemon.path(), &["ready-hang"]);
}

#[test]
fn undeclared_exit_status_remains_fatal() {
    let daemon = tempfile::tempdir().unwrap();
    let command = tempfile::tempdir().unwrap();
    let mut coordinator = owner(
        daemon.path(),
        command.path(),
        &[0],
        "retained-crash",
        Duration::from_secs(2),
    );
    let report = run(&mut coordinator, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert!(!report.success());
    assert_eq!(
        report.members[1].completion.as_ref().unwrap().status,
        Some(23)
    );
    assert_stopped(daemon.path(), &["ready-hang"]);
}

#[test]
fn expected_exit_deadline_stays_active_without_readiness_admission() {
    let daemon = tempfile::tempdir().unwrap();
    let command = tempfile::tempdir().unwrap();
    let mut coordinator = owner(
        daemon.path(),
        command.path(),
        &[0],
        "ready-hang",
        Duration::from_millis(100),
    );
    let report = run(&mut coordinator, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::Deadline);
    assert!(!report.success());
    assert!(report.members[1].completion.is_some());
    assert_stopped(daemon.path(), &["ready-hang"]);
    assert_stopped(command.path(), &["ready-hang"]);
}

#[test]
fn expected_exit_policy_requires_statuses_and_finite_completion_budget() {
    assert!(ExpectedExit::new(&[], Duration::from_secs(1)).is_err());
    assert!(ExpectedExit::new(&[0], Duration::ZERO).is_err());
    assert!(ExpectedExit::new(&vec![0; 257], Duration::from_secs(1)).is_err());
}
