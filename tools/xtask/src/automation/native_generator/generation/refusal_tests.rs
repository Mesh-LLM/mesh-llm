use super::tests::{setup, trace};
use super::*;

#[test]
fn empty_sources_refuse_before_report_directory_when_no_cpp_exists() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::remove_file(root.path().join("src/models/a.cpp")).unwrap();
    std::fs::remove_file(root.path().join("src/models/z.cpp")).unwrap();
    let result = execute(&options, &Cancellation::default(), |_| {
        panic!("publication reached")
    });
    assert!(matches!(result, Err(Error::Sources)));
    assert!(!options.report.parent().unwrap().exists());
}

#[test]
fn invalid_marker_refuses_without_fallback_when_marker_is_not_utf8() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(root.path().join(".mesh-llm-upstream-sha"), b"\xff").unwrap();
    let result = execute(&options, &Cancellation::default(), |_| {
        panic!("publication reached")
    });
    assert!(matches!(result, Err(Error::Io(_))));
    assert_eq!(trace(root.path()).len(), 1);
}

#[test]
fn first_gate_prevents_second_invocation_when_builder_is_inherited() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(
        root.path().join("first.json"),
        r#"{"builders":[{"verdict":"already_transformed"}]}"#,
    )
    .unwrap();
    let result = execute(&options, &Cancellation::default(), |_| {
        panic!("publication reached")
    });
    assert!(matches!(
        result,
        Err(Error::Report(generator::Error::Inherited(_)))
    ));
    assert_eq!(trace(root.path()).len(), 3);
}

#[test]
fn second_gate_prevents_diff_when_builder_remains_transformable() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(
        root.path().join("second.json"),
        r#"{"builders":[{"verdict":"transformable"}]}"#,
    )
    .unwrap();
    let result = execute(&options, &Cancellation::default(), |_| {
        panic!("publication reached")
    });
    assert!(matches!(
        result,
        Err(Error::Report(generator::Error::Remaining(_)))
    ));
    assert_eq!(trace(root.path()).len(), 4);
}

#[test]
fn exact_frozen_mail_bytes_are_published_when_both_reports_pass() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    let diff = include_bytes!("../../../../tests/fixtures/migration/rewriter-patch/two-model.diff");
    let expected =
        include_bytes!("../../../../tests/fixtures/migration/rewriter-patch/two-model.patch");
    std::fs::write(root.path().join("input.diff"), diff).unwrap();
    let result = execute(&options, &Cancellation::default(), |bytes| {
        publish_count(bytes, &options.publication)
    });
    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&options.publication.output).unwrap(),
        expected
    );
}

#[test]
fn empty_diff_preserves_combined_sentinel_when_both_reports_pass() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(root.path().join("input.diff"), b"").unwrap();
    std::fs::write(&options.publication.output, b"sentinel").unwrap();
    let result = execute(&options, &Cancellation::default(), |bytes| {
        publish_count(bytes, &options.publication)
    });
    assert!(matches!(
        result,
        Err(Error::Patch(
            super::super::super::rewriter_patch::PatchError::EmptyDiff
        ))
    ));
    assert_eq!(
        std::fs::read(&options.publication.output).unwrap(),
        b"sentinel"
    );
}

#[test]
fn prior_typed_report_survives_when_finalization_also_fails() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(root.path().join("fail-first"), b"").unwrap();
    let preceding = execute(&options, &Cancellation::default(), |_| {
        panic!("publication reached")
    });
    let result = scope::finalize(
        preceding,
        Err(crate::automation::command_interrupt::Reason::Interrupted),
    );
    assert!(
        matches!(result, Err(scope::Failure::Finalization { preceding: Err(Error::Command { report, .. }), .. }) if report.status.unwrap().code() == Some(23) && report.cleanup.complete)
    );
}

#[test]
fn parser_refuses_partial_shards_before_starting_processes() {
    let root = tempfile::tempdir().unwrap();
    let executable = fixture_path();
    let arguments = [
        "--source-root",
        root.path().to_str().unwrap(),
        "--git",
        executable.to_str().unwrap(),
        "--rewriter",
        executable.to_str().unwrap(),
        "--build-dir",
        "build",
        "--report",
        "report.json",
        "--output",
        "out",
        "--max-diff-bytes",
        "1",
        "--shard-output-dir",
        "shards",
    ]
    .map(str::to_owned);
    let result = options::parse(&arguments);
    assert!(matches!(
        result,
        Err(Error::Arguments("shard options must be used together"))
    ));
}

fn fixture_path() -> PathBuf {
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_generator_fixture{}",
            std::env::consts::EXE_SUFFIX
        ))
}
