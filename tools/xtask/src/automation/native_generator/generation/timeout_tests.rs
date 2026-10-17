use super::tests::{setup, trace};
use super::*;
use crate::process::Outcome;
use std::time::Duration;

#[test]
fn complete_rewriter_passes_have_a_separate_default_budget_from_git() {
    let root = tempfile::tempdir().unwrap();
    let fixture = setup(root.path());
    let mut arguments = [
        "--source-root",
        root.path().to_str().unwrap(),
        "--git",
        fixture.rewriter.to_str().unwrap(),
        "--rewriter",
        fixture.rewriter.to_str().unwrap(),
        "--build-dir",
        "build",
        "--report",
        "report.json",
        "--output",
        "out.patch",
        "--max-diff-bytes",
        "65536",
    ]
    .map(str::to_owned)
    .to_vec();
    let defaults = options::parse(&arguments).unwrap();
    assert_eq!(defaults.publication.git.timeout, Duration::from_secs(60));
    assert_eq!(defaults.rewriter_timeout, Duration::from_secs(600));

    arguments.extend(["--timeout".to_owned(), "2".to_owned()]);
    let explicit = options::parse(&arguments).unwrap();
    assert_eq!(explicit.publication.git.timeout, Duration::from_secs(2));
    assert_eq!(explicit.rewriter_timeout, Duration::from_secs(2));
    *arguments.last_mut().unwrap() = "0".into();
    assert!(matches!(
        options::parse(&arguments),
        Err(Error::Arguments(_))
    ));
}

#[test]
fn both_rewriter_passes_can_outlast_the_short_git_budget() {
    let root = tempfile::tempdir().unwrap();
    let mut options = setup(root.path());
    options.publication.git.timeout = Duration::from_secs(1);
    for pass in ["first", "second"] {
        std::fs::write(root.path().join(format!("delay-{pass}-ms")), b"1200").unwrap();
    }
    execute(&options, &Cancellation::default(), |diff| {
        publish_count(diff, &options.publication)
    })
    .unwrap();
    assert!(options.publication.output.is_file());
    assert_eq!(trace(root.path()).len(), 5);
}

#[test]
fn a_rewriter_deadline_keeps_readable_diagnostics_and_never_publishes() {
    for pass in ["first", "second"] {
        let root = tempfile::tempdir().unwrap();
        let mut options = setup(root.path());
        options.rewriter_timeout = Duration::from_secs(1);
        std::fs::write(root.path().join(format!("delay-{pass}-ms")), b"60000").unwrap();
        std::fs::write(&options.publication.output, b"sentinel").unwrap();
        let error = execute(&options, &Cancellation::default(), |_| {
            panic!("timed-out generation must not publish")
        })
        .unwrap_err();
        let diagnostic = error.to_string();
        assert!(diagnostic.contains(&format!("{pass} rewriter failed: outcome=Deadline")));
        assert!(diagnostic.contains(&format!("rewriter {pass} fixture started\n")));
        assert!(!diagnostic.contains("bytes_retained"));
        assert!(matches!(error, Error::Command { report, .. }
            if report.outcome == Outcome::Deadline && report.cleanup.complete));
        assert_eq!(
            std::fs::read(&options.publication.output).unwrap(),
            b"sentinel"
        );
        assert!(!trace(root.path()).iter().any(|call| call[0] == "diff"));
    }
}
