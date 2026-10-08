use super::super::{GitDiff, args};
use super::*;
use crate::process::Value;
use std::collections::BTreeMap;
use std::time::Duration;

fn fixture() -> PathBuf {
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

pub(super) fn setup(root: &Path) -> options::Options {
    std::fs::create_dir_all(root.join("src/models")).unwrap();
    for name in ["z.cpp", "a.cpp", "ignored.h"] {
        std::fs::write(root.join("src/models").join(name), b"source").unwrap();
    }
    std::fs::write(root.join("status.txt"), b"").unwrap();
    std::fs::write(
        root.join("first.json"),
        r#"{"builders":[{"file":"a","verdict":"transformable"}],"summary":{"transformable":1}}"#,
    )
    .unwrap();
    std::fs::write(root.join("second.json"), r#"{"builders":[{"file":"a","verdict":"already_transformed"}],"summary":{"already_transformed":1}}"#).unwrap();
    std::fs::write(
        root.join("input.diff"),
        b"diff --git a/src/models/a.cpp b/src/models/a.cpp\n+caf\xc3\xa9\r\n",
    )
    .unwrap();
    options::Options {
        publication: args::Options {
            git: GitDiff {
                executable: fixture(),
                source_root: root.to_owned(),
                base: "fixture-base".into(),
                environment: ["SYSTEMROOT", "WINDIR"]
                    .into_iter()
                    .filter_map(|key| {
                        std::env::var_os(key).map(|value| (key.into(), Value::Public(value)))
                    })
                    .collect(),
                max_bytes: std::num::NonZeroUsize::new(65536).unwrap(),
                timeout: Duration::from_secs(5),
            },
            output: root.join("combined.patch"),
            shards: None,
        },
        build: root.join("build"),
        rewriter: fixture(),
        report: root.join("reports/report.first.json"),
        extra: vec!["-std=c++17".into(), "-DFIXTURE=1".into()],
    }
}

pub(super) fn trace(root: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(root.join("trace.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn clean_refusal_precedes_rewriter_when_git_reports_tracked_changes() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(root.path().join("status.txt"), b" M src/models/a.cpp\n").unwrap();
    let result = execute(&options, &Cancellation::default(), |_| {
        panic!("publication must not run")
    });
    assert!(matches!(result, Err(Error::Dirty)));
    assert_eq!(
        trace(root.path()),
        vec![vec!["status", "--porcelain", "--untracked-files=no"]]
    );
    assert!(!options.report.exists());
}

#[test]
fn marker_identity_and_sorted_sources_reach_both_passes_when_generation_succeeds() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    std::fs::write(
        root.path().join(".mesh-llm-upstream-sha"),
        b" marker-identity\r\n",
    )
    .unwrap();
    let result = execute(&options, &Cancellation::default(), |diff| {
        publish_count(diff, &options.publication)
    })
    .unwrap();
    let calls = trace(root.path());
    assert_eq!(calls.len(), 4);
    let mut first = vec![
        "--source-root".to_owned(),
        root.path().display().to_string(),
        "--llama-commit".into(),
        "marker-identity".into(),
        "--report".into(),
        options.report.display().to_string(),
        "-p".into(),
        options.build.display().to_string(),
        "--apply".into(),
        "--extra-arg".into(),
        "-std=c++17".into(),
        "--extra-arg".into(),
        "-DFIXTURE=1".into(),
        root.path().join("src/models/a.cpp").display().to_string(),
        root.path().join("src/models/z.cpp").display().to_string(),
    ];
    assert_eq!(calls[1], first);
    first[5] = root
        .path()
        .join("reports/report.first-second.json")
        .display()
        .to_string();
    first.remove(8);
    assert_eq!(calls[2], first);
    assert_eq!(
        calls[3],
        [
            "diff",
            "--no-ext-diff",
            "--binary",
            "--full-index",
            "fixture-base",
            "--",
            "src/models"
        ]
    );
    let published = std::fs::read(&options.publication.output).unwrap();
    assert!(published.ends_with(b"+caf\xc3\xa9\r\n-- \n2.54.0\n\n"));
    assert_eq!(result.shards, 0);
}

#[test]
fn fallback_identity_is_trimmed_when_marker_is_absent() {
    let root = tempfile::tempdir().unwrap();
    let options = setup(root.path());
    execute(&options, &Cancellation::default(), |_| Ok(0)).unwrap();
    let calls = trace(root.path());
    assert_eq!(calls[1], ["rev-parse", "HEAD"]);
    assert_eq!(calls[2][3], "fallback-identity");
}

#[test]
fn publication_is_not_reached_when_either_rewriter_subprocess_fails() {
    for pass in ["first", "second"] {
        let root = tempfile::tempdir().unwrap();
        let options = setup(root.path());
        std::fs::write(root.path().join(format!("fail-{pass}")), b"").unwrap();
        std::fs::write(&options.publication.output, b"combined sentinel").unwrap();
        let result = execute(&options, &Cancellation::default(), |_| {
            panic!("publication must not run")
        });
        assert!(
            matches!(result, Err(Error::Command { report, .. }) if report.status.unwrap().code() == Some(23) && report.cleanup.complete)
        );
        assert_eq!(
            std::fs::read(&options.publication.output).unwrap(),
            b"combined sentinel"
        );
        assert!(!trace(root.path()).iter().any(|call| call[0] == "diff"));
    }
}

#[test]
fn success_json_uses_python_spacing_ascii_escaping_and_sorted_keys() {
    let generated = Generated {
        output: PathBuf::from("/caf\u{e9}/patch"),
        first_summary: BTreeMap::from([("transformable", 9007199254740993)]),
        second_summary: BTreeMap::new(),
        shards: 2,
    };
    let encoded = output::encode(&generated).unwrap();
    assert_eq!(
        format!("{encoded}\n"),
        "{\"first_summary\": {\"transformable\": 9007199254740993}, \"output\": \"/caf\\u00e9/patch\", \"second_summary\": {}, \"shards\": 2}\n"
    );
}

#[test]
fn finalization_retains_typed_error_when_restoration_fails() {
    let result = scope::finalize::<()>(
        Err(Error::Dirty),
        Err(crate::automation::command_interrupt::Reason::Interrupted),
    );
    assert!(matches!(
        result,
        Err(scope::Failure::Finalization {
            preceding: Err(Error::Dirty),
            ..
        })
    ));
}

#[test]
fn finalization_retains_success_receipt_when_restoration_fails() {
    let result = scope::finalize(
        Ok(Generated {
            output: "/patch".into(),
            first_summary: BTreeMap::new(),
            second_summary: BTreeMap::new(),
            shards: 3,
        }),
        Err(crate::automation::command_interrupt::Reason::Interrupted),
    );
    assert!(matches!(
        result,
        Err(scope::Failure::Finalization {
            preceding: Ok(Generated { shards: 3, .. }),
            ..
        })
    ));
}
