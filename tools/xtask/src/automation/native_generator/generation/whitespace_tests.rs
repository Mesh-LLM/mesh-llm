use super::tests::{setup, trace};
use super::*;

#[test]
fn marker_identity_strips_each_python_separator_when_present_at_both_edges() {
    for separator in [' ', '\t', '\n', '\r'] {
        let root = tempfile::tempdir().unwrap();
        let options = setup(root.path());
        std::fs::write(
            root.path().join(".mesh-llm-upstream-sha"),
            format!("{separator} \tmarker-identity\r\n{separator}"),
        )
        .unwrap();
        std::fs::write(root.path().join("head.txt"), b"different-head").unwrap();

        execute(&options, &Cancellation::default(), |_| Ok(0)).unwrap();

        let calls = trace(root.path());
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[1][3], "marker-identity");
        assert_eq!(calls[2][3], "marker-identity");
        assert!(!calls.iter().any(|call| call[0] == "rev-parse"));
    }
}

#[test]
fn fallback_identity_strips_each_python_separator_when_marker_is_absent() {
    for separator in [' ', '\t', '\n', '\r'] {
        let root = tempfile::tempdir().unwrap();
        let options = setup(root.path());
        std::fs::write(
            root.path().join("head.txt"),
            format!("{separator} \tfallback-control\r\n{separator}"),
        )
        .unwrap();

        execute(&options, &Cancellation::default(), |_| Ok(0)).unwrap();

        let calls = trace(root.path());
        assert_eq!(calls[1], ["rev-parse", "HEAD"]);
        assert_eq!(calls[2][3], "fallback-control");
        assert_eq!(calls[3][3], "fallback-control");
    }
}

#[test]
fn generation_accepts_blank_tracked_status_when_it_contains_python_separators() {
    for separator in [' ', '\t', '\n', '\r'] {
        let root = tempfile::tempdir().unwrap();
        let options = setup(root.path());
        std::fs::write(
            root.path().join("status.txt"),
            format!("{separator} \t\r\n{separator}"),
        )
        .unwrap();

        let result = execute(&options, &Cancellation::default(), |_| Ok(0));

        assert!(result.is_ok());
        let calls = trace(root.path());
        assert_eq!(calls.len(), 5);
        assert_eq!(calls[4][0], "diff");
    }
}

#[test]
fn dirty_status_precedes_invalid_marker_when_separators_surround_actual_changes() {
    for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
        let root = tempfile::tempdir().unwrap();
        let options = setup(root.path());
        std::fs::write(
            root.path().join("status.txt"),
            format!("{separator} M src/models/a.cpp\r\n{separator}"),
        )
        .unwrap();
        std::fs::write(root.path().join(".mesh-llm-upstream-sha"), b"\xff").unwrap();

        let result = execute(&options, &Cancellation::default(), |_| {
            panic!("publication must not run")
        });

        assert!(matches!(result, Err(Error::Dirty)));
        assert_eq!(
            trace(root.path()),
            vec![vec!["status", "--porcelain", "--untracked-files=no"]]
        );
    }
}

#[test]
fn empty_marker_identity_remains_authoritative_when_marker_contains_only_separators() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(root.path().join(".mesh-llm-upstream-sha"), b" \t\r\n").unwrap();
    std::fs::write(root.path().join("head.txt"), b"different-head").unwrap();

    execute(&options, &Cancellation::default(), |_| Ok(0)).unwrap();

    let calls = trace(root.path());
    assert_eq!(calls.len(), 4);
    assert_eq!(calls[1][3], "");
    assert_eq!(calls[2][3], "");
    assert!(!calls.iter().any(|call| call[0] == "rev-parse"));
}

#[test]
fn marker_identity_preserves_interior_separators_when_only_edges_are_stripped() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(
        root.path().join(".mesh-llm-upstream-sha"),
        b" left\x1c\x1d\x1e\x1fright\r\n",
    )
    .unwrap();

    execute(&options, &Cancellation::default(), |_| Ok(0)).unwrap();

    let calls = trace(root.path());
    assert_eq!(calls[1][3], "left\u{1c}\u{1d}\u{1e}\u{1f}right");
    assert_eq!(calls[2][3], "left\u{1c}\u{1d}\u{1e}\u{1f}right");
}
