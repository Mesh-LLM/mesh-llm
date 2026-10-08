use super::*;
#[test]
fn current_roles_and_historical_flags_remain_distinct() {
    for role in [Role::Public, Role::BinaryWorker] {
        let original = vec![
            Value::Public(role.legacy().into()),
            Value::Public("--openai-bind-addr".into()),
            Value::Public("127.0.0.1:1".into()),
        ];
        let mut args = original.clone();
        Dialect::Legacy.arguments(role, &mut args).unwrap();
        assert!(matches!(&args[0], Value::Public(value) if value == role.legacy()));
        Dialect::Current.arguments(role, &mut args).unwrap();
        assert!(matches!(&args[0], Value::Public(value) if value == "serve"));
        assert_eq!(
            args.iter()
                .any(|value| matches!(value, Value::Public(v) if v == "--worker-only")),
            matches!(role, Role::BinaryWorker)
        );
        assert_eq!(
            args.iter()
                .any(|value| matches!(value, Value::Public(v) if v == "--stage-transport")),
            !matches!(role, Role::Public)
        );
    }
}
#[cfg(unix)]
#[test]
fn actual_executable_help_admission_rejects_drift_and_unsupported_roles() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join("same filename");
    for (script, current, succeeds) in [
        ("exit 0", true, true),
        ("[ \"$1\" = serve-openai ]", false, true),
        ("exit 1", false, false),
    ] {
        std::fs::write(
            &binary,
            format!("#!/bin/sh\n[ \"$2\" = --help ] || exit 90\n{script}\n"),
        )
        .unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let digest = crate::product::digest::file_sha256(&binary)
            .map_err(|error| error.error)
            .unwrap();
        let spec = ProcessSpec {
            executable: binary.clone(),
            arguments: vec![],
            cwd: root.path().into(),
            environment: Default::default(),
        };
        let result = admit(
            &spec,
            &digest,
            Role::Public,
            Instant::now() + Duration::from_secs(15),
            &Cancellation::default(),
        );
        assert_eq!(result.is_ok(), succeeds);
        if let Ok(dialect) = result {
            assert_eq!(matches!(dialect, Dialect::Current), current);
        }
        assert!(
            admit(
                &spec,
                &"0".repeat(64),
                Role::Public,
                Instant::now() + Duration::from_secs(15),
                &Cancellation::default()
            )
            .is_err()
        );
    }
}

#[cfg(unix)]
#[test]
fn actual_help_rejects_invalid_utf8_expiry_cancellation_and_output_overflow() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join("binary");
    for script in [
        "printf '\\377'; exit 0",
        "i=0; while [ $i -lt 65537 ]; do printf x; i=$((i+1)); done",
        "while :; do :; done",
    ] {
        std::fs::write(&binary, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        let digest = crate::product::digest::file_sha256(&binary)
            .map_err(|error| error.error)
            .unwrap();
        let spec = ProcessSpec {
            executable: binary.clone(),
            arguments: vec![],
            cwd: root.path().into(),
            environment: Default::default(),
        };
        assert!(
            admit(
                &spec,
                &digest,
                Role::Public,
                Instant::now() + Duration::from_secs(4),
                &Cancellation::default()
            )
            .is_err()
        );
        let cancel = Cancellation::default();
        cancel.cancel();
        assert!(
            admit(
                &spec,
                &digest,
                Role::Public,
                Instant::now() + Duration::from_secs(15),
                &cancel
            )
            .is_err()
        );
        assert!(
            admit(
                &spec,
                &digest,
                Role::Public,
                Instant::now(),
                &Cancellation::default()
            )
            .is_err()
        );
    }
}
