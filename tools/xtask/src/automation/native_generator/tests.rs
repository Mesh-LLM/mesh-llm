use super::*;

#[path = "real_git.rs"]
mod real_git;

fn raw_fixture() -> PathBuf {
    std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_raw_capture_fixture{}",
            std::env::consts::EXE_SUFFIX
        ))
}

#[test]
fn output_and_shards_remain_untouched_when_run_overflows_raw_capture() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("input.diff"), b"diff exceeds cap\n").unwrap();
    let output = root.path().join("combined.patch");
    let shards = root.path().join("shards");
    std::fs::write(&output, b"combined sentinel").unwrap();
    std::fs::create_dir(&shards).unwrap();
    std::fs::write(shards.join("sentinel"), b"shard sentinel").unwrap();
    let arguments = [
        "--source-root".into(),
        root.path().to_str().unwrap().into(),
        "--git".into(),
        raw_fixture().to_str().unwrap().into(),
        "--diff-base".into(),
        "fixture-base".into(),
        "--output".into(),
        output.to_str().unwrap().into(),
        "--max-diff-bytes".into(),
        "1".into(),
        "--shard-output-dir".into(),
        shards.to_str().unwrap().into(),
        "--family-source-map".into(),
        root.path()
            .join("missing-map.json")
            .to_str()
            .unwrap()
            .into(),
        "--family-manifest".into(),
        root.path()
            .join("missing-manifest.json")
            .to_str()
            .unwrap()
            .into(),
    ];

    let error = run(&arguments).unwrap_err();

    assert!(matches!(
        error.downcast_ref::<Error>(),
        Some(Error::Process(process::Failure::RawCaptureOverflow {
            stream: process::Stream::Stdout,
            limit: 1
        }))
    ));
    assert_eq!(std::fs::read(output).unwrap(), b"combined sentinel");
    assert_eq!(
        std::fs::read(shards.join("sentinel")).unwrap(),
        b"shard sentinel"
    );
    assert_eq!(std::fs::read_dir(&shards).unwrap().count(), 1);
    let mut entries: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        ["combined.patch", "input.diff", "shards"].map(OsString::from)
    );
}
#[test]
fn combined_output_survives_when_optional_map_is_invalid() {
    let root = tempfile::tempdir().unwrap();
    let diff = include_bytes!("../../../tests/fixtures/migration/rewriter-patch/two-model.diff");
    let expected =
        include_bytes!("../../../tests/fixtures/migration/rewriter-patch/two-model.patch");
    let map = root.path().join("map.json");
    std::fs::write(&map, b"invalid json").unwrap();
    let options = args::Options {
        git: GitDiff {
            executable: root.path().join("git"),
            source_root: root.path().to_owned(),
            base: "HEAD".into(),
            environment: BTreeMap::new(),
            max_bytes: std::num::NonZeroUsize::new(1024).unwrap(),
            timeout: Duration::from_secs(1),
        },
        output: root.path().join("combined.patch"),
        shards: Some(args::ShardOutput {
            output: root.path().join("shards"),
            map,
            manifest: root.path().join("manifest.json"),
        }),
    };
    assert!(matches!(publish(diff, &options), Err(Error::Json(_))));
    assert_eq!(std::fs::read(options.output).unwrap(), expected);
}
#[test]
fn git_adapter_preserves_sensitive_hunks_and_exact_legacy_argv() {
    let root = tempfile::tempdir().unwrap();
    let executable = std::env::current_exe().unwrap();
    let profile = executable.parent().unwrap().parent().unwrap();
    let fixture = profile.join("examples").join(format!(
        "migration_raw_capture_fixture{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        fixture.is_file(),
        "build migration_raw_capture_fixture first"
    );
    let mut expected = b"diff --git a/src/models/a.cpp b/src/models/a.cpp\n+token password secret authorization invite\r\n".to_vec();
    expected.extend(vec![b'x'; 9000]);
    std::fs::write(root.path().join("input.diff"), &expected).unwrap();
    let input = GitDiff {
        executable: fixture,
        source_root: root.path().to_owned(),
        base: "fixture-base".into(),
        environment: ["SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect(),
        max_bytes: std::num::NonZeroUsize::new(expected.len()).unwrap(),
        timeout: Duration::from_secs(5),
    };
    let result = capture_diff(&input, &Cancellation::default()).unwrap();
    assert_eq!(result.as_bytes(), expected);
}
#[test]
fn raw_invalid_utf8_reaches_encoder_without_diagnostic_translation() {
    let root = tempfile::tempdir().unwrap();
    let executable = std::env::current_exe().unwrap();
    let fixture = executable
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_raw_capture_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
    std::fs::write(root.path().join("input.diff"), b"token\0\xff\r\n").unwrap();
    let input = GitDiff {
        executable: fixture,
        source_root: root.path().to_owned(),
        base: "fixture-base".into(),
        environment: ["SYSTEMROOT", "WINDIR"]
            .into_iter()
            .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
            .collect(),
        max_bytes: std::num::NonZeroUsize::new(64).unwrap(),
        timeout: Duration::from_secs(5),
    };
    let raw = capture_diff(&input, &Cancellation::default()).unwrap();
    assert!(matches!(
        encode_mail_patch("fixture", raw.as_bytes()),
        Err(super::super::rewriter_patch::PatchError::InvalidUtf8)
    ));
}
#[test]
fn output_is_replaced_with_complete_series_when_shards_are_valid() {
    let root = tempfile::tempdir().unwrap();
    let diff = include_bytes!("../../../tests/fixtures/migration/rewriter-patch/two-model.diff");
    let map = serde_json::json!({"schema_version": 1, "families": {"fixture": ["src/models/a.cpp", "src/models/b.cpp"]}});
    let manifest = serde_json::json!({"models": [{"family": "fixture", "class": "causal_generation", "profile": "full"}]});
    let patches = encode_family_shards(diff, &map, &manifest).unwrap();
    let output = root.path().join("shards");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("old"), b"old").unwrap();
    publication::publish(&output, &patches).unwrap();
    assert_eq!(
        std::fs::read(output.join("series")).unwrap(),
        patches.series
    );
    assert_eq!(
        std::fs::read(output.join("series.json")).unwrap(),
        patches.series_json
    );
    for shard in patches.shards {
        assert_eq!(std::fs::read(output.join(shard.file)).unwrap(), shard.bytes);
    }
}
#[test]
fn parser_requires_explicit_cap_and_complete_shard_options() {
    let executable = raw_fixture();
    let arguments = [
        "--source-root",
        ".",
        "--git",
        executable.to_str().unwrap(),
        "--output",
        "out",
        "--max-diff-bytes",
        "0",
    ]
    .map(str::to_owned);
    assert!(matches!(
        args::parse(&arguments),
        Err(Error::Arguments("invalid byte cap"))
    ));
}

#[test]
fn parser_rejects_shard_directory_when_option_group_is_incomplete() {
    let executable = raw_fixture();
    let arguments = [
        "--source-root",
        ".",
        "--git",
        executable.to_str().unwrap(),
        "--output",
        "out",
        "--max-diff-bytes",
        "1",
        "--shard-output-dir",
        "shards",
    ]
    .map(str::to_owned);

    let result = args::parse(&arguments);

    assert!(matches!(
        result,
        Err(Error::Arguments("shard options must be used together"))
    ));
}
