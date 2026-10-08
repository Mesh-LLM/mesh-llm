use super::*;
#[test]
fn sysfs_capture_has_explicit_unavailable_and_bounded_zone_reading() {
    let root = tempfile::tempdir().unwrap();
    assert_eq!(linux_thermal(root.path())["available"], false);
    for (name, text) in [
        ("thermal_zone0", "42000\n"),
        ("thermal_zone1", "bad"),
        ("thermal_zone2", "999999"),
        ("thermal_zone64", "12000"),
    ] {
        std::fs::create_dir(root.path().join(name)).unwrap();
        std::fs::write(root.path().join(name).join("temp"), text).unwrap();
    }
    let evidence = linux_thermal(root.path());
    assert_eq!(evidence["available"], true);
    assert_eq!(
        evidence["temperature_millidegrees_celsius"],
        json!({"thermal_zone0":42000})
    );
}
#[cfg(unix)]
fn executable(root: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = root.join("probe");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
#[cfg(unix)]
#[test]
fn version_probe_executes_only_version_and_captures_bounded_output() {
    let root = tempfile::tempdir().unwrap();
    let path = executable(
        root.path(),
        "[ \"$#\" = 1 ] && [ \"$1\" = --version ] || exit 9\nprintf 'fixture-version\\n'",
    );
    assert_eq!(
        version(&path, Duration::from_secs(2), &Cancellation::default())
            .unwrap()
            .as_deref(),
        Some("fixture-version")
    );
}
#[cfg(unix)]
#[test]
fn thermal_probe_uses_exact_arguments_and_failed_probe_is_unavailable() {
    let root = tempfile::tempdir().unwrap();
    let path = executable(
        root.path(),
        "[ \"$*\" = '-g therm' ] || exit 9\nprintf 'fixture thermal\\n'",
    );
    assert_eq!(
        darwin_thermal(&path, Duration::from_secs(2), &Cancellation::default()).unwrap()["raw"],
        "fixture thermal"
    );
    std::fs::write(&path, "#!/bin/sh\nexit 7\n").unwrap();
    assert_eq!(
        darwin_thermal(&path, Duration::from_secs(2), &Cancellation::default()).unwrap()["available"],
        false
    );
}
#[test]
fn expired_or_cancelled_probe_budget_never_launches_missing_executable() {
    let missing = Path::new("/missing/preflight/probe");
    assert!(
        version(missing, Duration::ZERO, &Cancellation::default())
            .unwrap()
            .is_none()
    );
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(
        version(missing, Duration::from_secs(2), &cancel)
            .unwrap()
            .is_none()
    );
}

#[cfg(unix)]
#[test]
fn truncated_stdout_or_stderr_cannot_claim_available_version_or_thermal_metadata() {
    let root = tempfile::tempdir().unwrap();
    for stream in ["", " >&2"] {
        let path = executable(
            root.path(),
            &format!(
                "/usr/bin/head -c 20000 /dev/zero | /usr/bin/tr '\\000' x{stream}\nprintf 'valid-prefix\\n'"
            ),
        );
        let version = version(&path, Duration::from_secs(2), &Cancellation::default());
        assert!(version.is_err() || version.unwrap().is_none());
        let thermal = darwin_thermal(&path, Duration::from_secs(2), &Cancellation::default());
        assert!(thermal.is_err() || thermal.unwrap()["available"] == false);
    }
}

#[cfg(unix)]
#[test]
fn suppressed_capture_is_unavailable_even_when_raw_bytes_were_retained() {
    let root = tempfile::tempdir().unwrap();
    let path = executable(root.path(), "printf 'secret=fixture\\n'");
    assert!(
        version(&path, Duration::from_secs(2), &Cancellation::default())
            .unwrap()
            .is_none()
    );
    assert_eq!(
        darwin_thermal(&path, Duration::from_secs(2), &Cancellation::default()).unwrap()["available"],
        false
    );
}

#[test]
fn probe_budget_reserves_grace_force_and_eof_drain_and_keeps_five_second_cap() {
    assert_eq!(probe_budget(Duration::from_millis(400)), Duration::ZERO);
    assert_eq!(probe_budget(Duration::from_millis(600)), Duration::ZERO);
    assert_eq!(
        probe_budget(Duration::from_millis(601)),
        Duration::from_millis(1)
    );
    assert_eq!(
        probe_budget(Duration::from_secs(10)),
        Duration::from_secs(5)
    );
}
