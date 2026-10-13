use super::{
    cli::case,
    protocol::{Behavior, Plan},
};
use crate::{pty::Terminal, support::Sentinel};

fn interrupt(behavior: Behavior, admitted: bool, terminate: bool) {
    let case = case(&Plan {
        behavior,
        ..Plan::default()
    });
    let mut sentinel = Sentinel::new(&case);
    let mut terminal = Terminal::start_command(&case, "daemon-readiness");
    terminal.wait_file(if admitted {
        "cleanup.armed"
    } else {
        "models.armed"
    });
    if terminate {
        terminal.terminate();
    } else {
        terminal.ctrl_c();
    }
    if admitted {
        for _ in 0..5 {
            terminal.ctrl_c();
            terminal.terminate();
        }
        std::fs::write(case.native.join("cleanup.release"), b"release").unwrap();
    }
    let status = terminal.finish();
    let diagnostics = std::fs::read_to_string(case.native.join("cli.stderr")).unwrap();
    assert_eq!(status.code(), Some(1), "{diagnostics}");
    assert!(diagnostics.contains("cancelled"), "{diagnostics}");
    assert!(
        diagnostics.contains(if admitted {
            "ProbeAdmitted"
        } else {
            "NotAdmitted"
        }),
        "{diagnostics}"
    );
    assert!(
        std::fs::read(case.native.join("cli.stdout"))
            .unwrap()
            .is_empty()
    );
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
    terminal.disarm();
}

#[test]
fn d19_ctrl_c_during_http() {
    interrupt(Behavior::SlowBody, false, false);
}
#[test]
fn d19_sigterm_during_http() {
    interrupt(Behavior::SlowHeaders, false, true);
}
#[test]
fn d19_repeated_interrupt_after_admission() {
    interrupt(Behavior::HoldCleanup, true, false);
}
