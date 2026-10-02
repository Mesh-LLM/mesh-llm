use super::*;
fn args() -> Vec<String> {
    [
        "--base-url",
        "http://127.0.0.1:9337",
        "--model",
        "model",
        "--mode",
        "contract",
    ]
    .map(str::to_owned)
    .to_vec()
}
#[test]
fn actual_wrapper_endpoint_and_timeouts_are_admitted_without_readiness_inference() {
    for timeout in ["120", "600", "900", "0.1"] {
        let mut a = args();
        a.extend(["--timeout".into(), timeout.into()]);
        let o = Options::parse(&a).unwrap();
        assert_eq!(o.endpoint, "http://127.0.0.1:9337/systemone");
    }
    for timeout in ["0", "-1", "NaN", "inf", "3601"] {
        let mut a = args();
        a.extend(["--timeout".into(), timeout.into()]);
        assert!(Options::parse(&a).is_err());
    }
    for endpoint in [
        "https://127.0.0.1:9337",
        "/",
        "http://user@127.0.0.1:9337",
        "http://127.0.0.1:9337?secret=x",
        "http://127.0.0.1:9337#fragment",
    ] {
        let mut a = args();
        a[1] = endpoint.into();
        assert!(Options::parse(&a).is_err());
    }
}
#[test]
fn report_destination_refuses_special_files_before_opening_and_never_truncates_directory() {
    let scratch = tempfile::tempdir().unwrap();
    assert!(write_report(scratch.path(), &serde_json::json!({})).is_err());

    let file = scratch.path().join("old");
    std::fs::write(&file, "old").unwrap();
    #[cfg(unix)]
    {
        let link = scratch.path().join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(write_report(&link, &serde_json::json!({})).is_err());
        assert_eq!(std::fs::read_to_string(file).unwrap(), "old");
    }
}

#[test]
fn ipv6_dial_host_is_bare_but_http_authority_keeps_brackets_and_port() {
    let uri: hyper::Uri = "http://[::1]:9337/systemone".parse().unwrap();
    assert_eq!(transport::socket_host(&uri).unwrap(), "::1");
    assert_eq!(uri.authority().unwrap().as_str(), "[::1]:9337");
    let uri: hyper::Uri = "http://127.0.0.1:9337/systemone".parse().unwrap();
    assert_eq!(transport::socket_host(&uri).unwrap(), "127.0.0.1");
}

#[test]
fn http_runtime_shutdown_returns_before_uncancellable_blocking_work_finishes() {
    use std::sync::mpsc;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = runtime.spawn_blocking(move || {
        started_tx.send(()).unwrap();
        // Bounded stand-in for Tokio's blocking getaddrinfo task. The owner
        // cannot cancel it, and it cannot complete until after shutdown returns.
        let released = release_rx.recv_timeout(Duration::from_secs(2)).is_ok();
        let _ = finished_tx.send(released);
    });
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    shutdown_http_runtime(runtime);
    assert!(matches!(
        finished_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    release_tx.send(()).unwrap();
    assert!(finished_rx.recv_timeout(Duration::from_secs(1)).unwrap());
    drop(worker);
}

#[test]
fn explicit_invalid_ports_are_rejected_before_fallback_or_report_publication() {
    for endpoint in [
        "http://127.0.0.1:99999",
        "http://127.0.0.1:",
        "http://127.0.0.1:+80",
        "http://127.0.0.1:-1",
        "http://[::1]:99999",
        "http://[::1]:",
        "http://[invalid]:80",
    ] {
        let mut options = args();
        options[1] = endpoint.into();
        assert!(Options::parse(&options).is_err(), "{endpoint}");
    }
    for (endpoint, port) in [
        ("http://127.0.0.1", 80),
        ("http://127.0.0.1:9337", 9337),
        ("http://[::1]", 80),
        ("http://[::1]:9337", 9337),
    ] {
        let mut options = args();
        options[1] = endpoint.into();
        let options = Options::parse(&options).unwrap();
        let uri = options.endpoint.parse().unwrap();
        assert_eq!(transport::socket_port(&uri).unwrap(), port);
    }
}
