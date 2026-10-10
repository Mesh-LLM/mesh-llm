use super::*;
#[test]
#[ignore = "owned acquisition lifecycle peer; invoked only by these tests"]
fn owned_worker() {
    let directory = std::env::var("OWNED_FETCH_PEER").unwrap();
    std::fs::write(
        std::path::Path::new(&directory).join("pid"),
        std::process::id().to_string(),
    )
    .unwrap();
    let mode = std::env::var("OWNED_FETCH_MODE").unwrap();
    if mode == "overflow" {
        use std::io::Write as _;
        std::io::stdout()
            .lock()
            .write_all(&vec![b'x'; STDOUT_CAP as usize + 1])
            .unwrap();
    }
    #[cfg(unix)]
    if mode == "escape-group" {
        // SAFETY: this dedicated fixture moves only itself into its parent's
        // existing group, exercising direct-child fallback without signalling it.
        unsafe {
            assert_eq!(libc::setpgid(0, libc::getpgid(libc::getppid())), 0);
        }
    }
    if mode == "normal" {
        return;
    }
    // Exercise a live native thread independently of HTTP client behavior.
    let worker = std::thread::spawn(|| {
        loop {
            std::thread::park_timeout(Duration::from_secs(1));
        }
    });
    let _ = worker.join();
}
fn peer(mode: &str, directory: &std::path::Path) -> Command {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--ignored",
            "--exact",
            "layer_package_fetch::worker_lifecycle::tests::owned_worker",
            "--nocapture",
        ])
        .env("OWNED_FETCH_PEER", directory)
        .env("OWNED_FETCH_MODE", mode);
    command
}
#[cfg(unix)]
fn assert_reaped(directory: &std::path::Path) {
    let pid: libc::pid_t = std::fs::read_to_string(directory.join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    // SAFETY: zero signal is an existence probe for the fixture PID.
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}
#[test]
fn deadline_and_capture_refusal_reap_worker_with_native_thread() {
    #[cfg(unix)]
    let modes = ["hold", "overflow", "escape-group"].as_slice();
    #[cfg(not(unix))]
    let modes = ["hold", "overflow"].as_slice();
    for &mode in modes {
        let directory = tempfile::tempdir().unwrap();
        let start = Instant::now();
        let error = run(peer(mode, directory.path()), Duration::from_secs(2)).unwrap_err();
        assert!(start.elapsed() < Duration::from_secs(3));
        let text = format!("{error:#}");
        assert!(text.contains("reaped=true"), "{text}");
        assert!(
            text.contains(if mode == "overflow" {
                "capture exceeded"
            } else {
                "deadline expired"
            }),
            "{text}"
        );
        #[cfg(unix)]
        assert_reaped(directory.path());
    }
}
#[test]
fn normally_exited_worker_is_reaped_and_capture_is_read_after_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = run(peer("normal", directory.path()), Duration::from_secs(3)).unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("test result: ok"));
    #[cfg(unix)]
    assert_reaped(directory.path());
}
