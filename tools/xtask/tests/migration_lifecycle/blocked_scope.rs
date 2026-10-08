use super::super::Interrupt;
use super::{action, replace};
use std::sync::atomic::{AtomicBool, Ordering};

static DELIVERED: AtomicBool = AtomicBool::new(false);

extern "C" fn existing(_: libc::c_int) {
    DELIVERED.store(true, Ordering::SeqCst);
}

#[test]
fn migration_lifecycle_blocked_scope_preserves_mask_pending_and_dispositions() {
    match std::env::var("TASK20_BLOCKED_SCOPE_PROBE") {
        Ok(case) => probe(&case),
        Err(_) => {
            for case in ["int", "term", "pending-int", "pending-term"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "command_interrupt::platform::tests::migration_lifecycle_blocked_scope_preserves_mask_pending_and_dispositions",
                        "--nocapture",
                    ])
                    .env("TASK20_BLOCKED_SCOPE_PROBE", case)
                    .output()
                    .unwrap();
                let status = output.status;
                assert!(status.success(), "{case}: {status:?}");
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
                    "{output:?}"
                );
                eprintln!("{case}: {}", String::from_utf8_lossy(&output.stdout));
            }
        }
    }
}

fn probe(case: &str) {
    // SAFETY: this isolated subprocess has no unrelated work; SIGALRM bounds a hung test.
    unsafe { libc::alarm(12) };
    let (signal, pending) = match case {
        "int" => (libc::SIGINT, false),
        "term" => (libc::SIGTERM, false),
        "pending-int" => (libc::SIGINT, true),
        "pending-term" => (libc::SIGTERM, true),
        _ => panic!("invalid isolated probe"),
    };
    let mut mask = thread_mask();
    // SAFETY: initialized local signal set; SIGUSR1 is an unrelated valid signal.
    assert_eq!(unsafe { libc::sigaddset(&mut mask, libc::SIGUSR1) }, 0);
    // SAFETY: initialized local signal set and signal is SIGINT or SIGTERM.
    assert_eq!(unsafe { libc::sigaddset(&mut mask, signal) }, 0);
    // SAFETY: only the isolated probe thread changes its own mask, before raising its signal.
    assert_eq!(
        unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &mask, std::ptr::null_mut()) },
        0
    );
    let mut owned = action(signal).unwrap();
    owned.sa_sigaction = (existing as *const ()).expose_provenance();
    owned.sa_flags = libc::SA_RESTART;
    // SAFETY: sa_mask is initialized; this adds a valid unrelated signal.
    assert_eq!(
        unsafe { libc::sigaddset(&mut owned.sa_mask, libc::SIGUSR2) },
        0
    );
    replace(signal, &owned).unwrap();
    let before = [
        action(libc::SIGINT).unwrap(),
        action(libc::SIGTERM).unwrap(),
    ];
    if pending {
        // SAFETY: the calling thread blocks this signal and owns its static callback.
        assert_eq!(unsafe { libc::raise(signal) }, 0);
    }
    let queued = pending_signals();
    assert_eq!(member(&queued, signal), i32::from(pending));

    let result = Interrupt::install().map_err(crate::automation::client_readiness::Error::from);
    let mut observer = crate::automation::retained_session::tests::blocked_observer();
    assert!(matches!(
        crate::automation::retained_session::run(&mut observer, &crate::automation::retained_session::tests::test_limits()),
        Err(crate::automation::retained_session::Error::Interrupt(reason)) if matches!((signal, &reason),
            (libc::SIGINT, super::Reason::BlockedSigint) | (libc::SIGTERM, super::Reason::BlockedSigterm))
    ));

    assert!(
        matches!(result, Err(crate::automation::client_readiness::Error::Invalid(message)) if message.contains("blocked"))
    );
    crate::automation::shared_owner_tests::assert_second_consumer_refuses_blocked_signal(signal);
    let after_mask = thread_mask();
    let after_pending = pending_signals();
    assert_same_set(&mask, &after_mask);
    assert_same_set(&queued, &after_pending);
    for (signal, before) in [libc::SIGINT, libc::SIGTERM].into_iter().zip(before) {
        let after = action(signal).unwrap();
        assert_eq!(after.sa_sigaction, before.sa_sigaction);
        assert_eq!(after.sa_flags, before.sa_flags);
        assert_same_set(&after.sa_mask, &before.sa_mask);
    }
    assert!(!DELIVERED.load(Ordering::SeqCst));
    assert!(!super::super::OWNED.load(Ordering::SeqCst));
}

fn thread_mask() -> libc::sigset_t {
    let mut mask = std::mem::MaybeUninit::zeroed();
    // SAFETY: null set is query-only; output points to writable sigset_t storage.
    assert_eq!(
        unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), mask.as_mut_ptr()) },
        0
    );
    // SAFETY: successful pthread_sigmask initialized mask.
    unsafe { mask.assume_init() }
}

fn pending_signals() -> libc::sigset_t {
    let mut pending = std::mem::MaybeUninit::zeroed();
    // SAFETY: output points to writable sigset_t storage; sigpending does not consume signals.
    assert_eq!(unsafe { libc::sigpending(pending.as_mut_ptr()) }, 0);
    // SAFETY: successful sigpending initialized pending.
    unsafe { pending.assume_init() }
}

fn assert_same_set(before: &libc::sigset_t, after: &libc::sigset_t) {
    for signal in 1..=31 {
        assert_eq!(
            member(before, signal),
            member(after, signal),
            "signal {signal}"
        );
    }
}

fn member(set: &libc::sigset_t, signal: libc::c_int) -> libc::c_int {
    // SAFETY: set is initialized and signal is in the portable Unix signal range.
    unsafe { libc::sigismember(set, signal) }
}
