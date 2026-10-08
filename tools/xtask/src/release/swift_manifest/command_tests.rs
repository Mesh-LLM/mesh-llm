use super::command::run;

#[test]
fn migration_release_swift_manifest_command_checks_artifact_before_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("artifact");
    let manifest = directory.path().join("Package.swift");
    let args = [
        "verify".to_owned(),
        "tag".to_owned(),
        artifact.display().to_string(),
        manifest.display().to_string(),
        "opaque".to_owned(),
    ];
    let report = run(&args);
    assert_eq!(report.code, 1);
    assert_eq!(
        report.stderr,
        format!(
            "Swift release artifact does not exist: {}\n",
            artifact.display()
        )
    );
}

#[test]
fn migration_release_swift_manifest_command_rejects_directory_as_manifest() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("artifact");
    std::fs::write(&artifact, b"not a real zip").unwrap();
    let args = [
        "verify".to_owned(),
        "tag".to_owned(),
        artifact.display().to_string(),
        directory.path().display().to_string(),
        "opaque".to_owned(),
    ];
    let report = run(&args);
    assert_eq!(report.code, 1);
    assert_eq!(
        report.stderr,
        format!(
            "Package.swift does not exist: {}\n",
            directory.path().display()
        )
    );
}

#[test]
fn migration_release_swift_manifest_command_updates_without_native_execution() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("artifact");
    let manifest = directory.path().join("Package.swift");
    std::fs::write(&artifact, b"opaque adapter fixture").unwrap();
    std::fs::write(
        &manifest,
        include_bytes!("../../../tests/fixtures/release/swift_manifest/template.swift"),
    )
    .unwrap();
    let args = [
        "update".to_owned(),
        "not-a-version".to_owned(),
        artifact.display().to_string(),
        manifest.display().to_string(),
        "opaque-checksum".to_owned(),
    ];
    let report = run(&args);
    assert_eq!(report.code, 0);
    assert_eq!(
        std::fs::read(&manifest).unwrap(),
        include_bytes!("../../../tests/fixtures/release/swift_manifest/updated.swift")
    );
}

#[test]
fn migration_release_swift_manifest_command_verifies_without_mutating_file() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("artifact");
    let manifest = directory.path().join("Package.swift");
    let original = include_bytes!("../../../tests/fixtures/release/swift_manifest/updated.swift");
    std::fs::write(&artifact, b"opaque adapter fixture").unwrap();
    std::fs::write(&manifest, original).unwrap();
    let args = [
        "verify".to_owned(),
        "not-a-version".to_owned(),
        artifact.display().to_string(),
        manifest.display().to_string(),
        "opaque-checksum".to_owned(),
    ];
    let report = run(&args);
    assert_eq!(report.code, 0);
    assert_eq!(
        report.stdout,
        "verified Swift package manifest for not-a-version\n"
    );
    assert_eq!(std::fs::read(&manifest).unwrap(), original);
}
