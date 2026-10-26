use crate::protocol::Behavior;
use crate::pty::Terminal;
use crate::support::{Case, Sentinel, assert_absent, ready};

#[test]
fn migration_lifecycle_interrupt_ctrl_c_before_ready_cleans_owned_state() {
    let case = Case::new(Behavior::Clean, vec![]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("startup.armed");

    terminal.ctrl_c();
    let status = terminal.finish();

    assert_cancelled(&case, status, "Cancelled");
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

#[test]
fn migration_lifecycle_interrupt_sigterm_before_ready() {
    let case = Case::new(Behavior::Clean, vec![]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("startup.armed");

    terminal.terminate();
    let status = terminal.finish();

    assert_cancelled(&case, status, "Cancelled");
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

#[test]
fn migration_lifecycle_interrupt_cleanup_only_readiness_stays_cancelled() {
    let case = Case::new(Behavior::LateReady, vec![]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("startup.armed");

    terminal.ctrl_c();
    let status = terminal.finish();

    let diagnostics = assert_cancelled(&case, status, "Cancelled");
    assert!(diagnostics.contains("stop=NotAdmitted"));
    assert!(diagnostics.contains("Client ready"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

#[test]
fn migration_lifecycle_interrupt_repeated_during_cancelled_cleanup() {
    let case = Case::new(Behavior::HeldCleanup, vec![]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("startup.armed");

    terminal.ctrl_c();
    terminal.wait_file("cleanup.armed");
    repeat_and_release(&case, &mut terminal);
    let status = terminal.finish();

    let diagnostics = assert_cancelled(&case, status, "Cancelled");
    assert!(diagnostics.contains("forced: false"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

#[test]
fn migration_lifecycle_interrupt_admitted_cleanup_preserves_receipt_but_fails_command() {
    let case = Case::new(Behavior::HeldCleanup, vec![ready()]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("cleanup.armed");

    repeat_and_release(&case, &mut terminal);
    let status = terminal.finish();

    let diagnostics = assert_cancelled(&case, status, "Ready");
    assert!(diagnostics.contains("stop=Admitted"));
    assert!(diagnostics.contains("RequestedAfterLiveObservation"));
    assert!(diagnostics.contains("forced: false"));
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

#[test]
fn migration_lifecycle_interrupt_live_descendant_is_removed() {
    descendant_case(Behavior::LiveDescendant, false);
}

#[test]
fn migration_lifecycle_interrupt_stubborn_tree_is_forced_after_repeated_interrupts() {
    descendant_case(Behavior::StubbornDescendant, true);
}

fn descendant_case(behavior: Behavior, forced: bool) {
    let case = Case::new(behavior, vec![]);
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start(&case);
    terminal.wait_file("startup.armed");
    let leaf = std::fs::read_to_string(case.native.join("leaf.pid"))
        .unwrap()
        .parse()
        .unwrap();

    terminal.ctrl_c();
    if forced {
        for _ in 0..5 {
            terminal.ctrl_c();
            terminal.terminate();
        }
    }
    let status = terminal.finish();

    let diagnostics = assert_cancelled(&case, status, "Cancelled");
    assert!(
        diagnostics.contains(&format!("forced: {forced}")),
        "{diagnostics}"
    );
    assert_absent(leaf);
    assert!(sentinel.0.try_wait().unwrap().is_none());
    terminal.disarm();
}

fn repeat_and_release(case: &Case, terminal: &mut Terminal<'_>) {
    for _ in 0..5 {
        terminal.ctrl_c();
        terminal.terminate();
    }
    std::fs::write(case.native.join("cleanup.release"), b"release").unwrap();
}

fn assert_cancelled(case: &Case, status: std::process::ExitStatus, outcome: &str) -> String {
    let diagnostics = std::fs::read_to_string(case.native.join("cli.stderr")).unwrap();
    assert_eq!(status.code(), Some(1), "{status:?}; {diagnostics}");
    assert!(diagnostics.contains("cancelled"), "{diagnostics}");
    assert!(
        diagnostics.contains(&format!("outcome={outcome}")),
        "{diagnostics}"
    );
    assert!(
        std::fs::read(case.native.join("cli.stdout"))
            .unwrap()
            .is_empty()
    );
    case.assert_removed();
    diagnostics
}
