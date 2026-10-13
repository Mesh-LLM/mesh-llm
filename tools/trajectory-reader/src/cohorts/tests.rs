use super::{Selection, document, input, messages};
use parquet::basic::Compression;
#[path = "../../tests/parquet_cohort_fixture.rs"]
mod fixture;
use fixture::{Row, row, write};
fn policy() -> Selection {
    Selection {
        cohorts: vec!["warmup".into(), "1".into()],
        frameworks: vec!["z".into(), "a".into(), "m".into()],
        sources: vec!["source".into()],
        trajectories_per_framework: None,
        sessions_per_cohort: Some(4),
        min_isl: 8192,
        max_isl_exclusive: 65536,
        min_turns: 2,
    }
}
fn rows() -> Vec<Row> {
    (0..6)
        .flat_map(|i| ["z", "a", "m"].map(|f| row(&format!("{f}-{i}"), f)))
        .collect()
}
#[test]
fn balanced_whole_cohorts_preserve_declared_order_messages_and_digest() {
    let state = tempfile::tempdir().unwrap();
    let file = state.path().join("input.parquet");
    for compression in [Compression::SNAPPY, Compression::ZSTD(Default::default())] {
        write(&file, &rows(), compression);
        let cohorts = input::select(&file, &policy()).unwrap();
        let first = &cohorts["warmup"];
        assert_eq!(
            first
                .iter()
                .map(|r| r.agent_framework.as_str())
                .collect::<Vec<_>>(),
            ["z", "z", "a", "m"]
        );
        let ids = cohorts
            .values()
            .flatten()
            .map(|r| &r.session_id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(ids.len(), 8);
        assert_eq!(first[0].messages[1].content, "");
        assert_eq!(first[0].messages[2].tool_call_id.as_deref(), Some("call"));
        let doc = document::manifest(cohorts, &policy(), &"a".repeat(40), &"b".repeat(64)).unwrap();
        use sha2::{Digest, Sha256};
        let ids = doc["cohorts"]["warmup"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["session_id"].as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            doc["metadata"]["cohorts"]["warmup"]["session_ids_sha256"],
            Sha256::digest(ids)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        assert_eq!(doc["metadata"]["cohorts"]["warmup"]["assistant_turns"], 8);
    }
}
#[test]
fn sixteen_sessions_balance_six_five_five_in_declared_order() {
    let mut s = policy();
    s.cohorts = vec!["one".into()];
    s.sessions_per_cohort = Some(16);
    assert_eq!(s.allocation().unwrap(), [6, 5, 5]);
    s.frameworks.push("z".into());
    assert!(s.allocation().is_err());
}
#[test]
fn best_global_duplicate_and_actual_assistant_threshold_are_order_independent() {
    let state = tempfile::tempdir().unwrap();
    let file = state.path().join("input.parquet");
    let mut values = rows();
    let mut duplicate = values[0].clone();
    duplicate.isl = 11000;
    duplicate.body = duplicate.body.replace("answer", "best");
    values.push(duplicate);
    let mut low = row("low", "z");
    low.turns = 100;
    low.body = "[{\"role\":\"assistant\",\"content\":\"only one\"}]".into();
    values.push(low);
    write(&file, &values, Compression::SNAPPY);
    let first = input::select(&file, &policy()).unwrap();
    values.reverse();
    write(&file, &values, Compression::SNAPPY);
    let second = input::select(&file, &policy()).unwrap();
    assert_eq!(
        serde_json::to_value(&first).unwrap(),
        serde_json::to_value(&second).unwrap()
    );
    assert!(!first.values().flatten().any(|r| r.session_id == "low"));
    let mut one = policy();
    one.cohorts = vec!["all".into()];
    one.sessions_per_cohort = Some(18);
    let selected = input::select(&file, &one).unwrap();
    let best = selected
        .values()
        .flatten()
        .find(|r| r.session_id == "z-0")
        .unwrap();
    assert_eq!(best.max_isl, 11000);
    assert_eq!(best.messages[3].content, "best");
}
#[test]
fn malformed_messages_roles_tools_and_short_framework_refuse() {
    for body in [
        "[]",
        "[1]",
        "[{\"role\":\"invalid\"}]",
        "[{\"role\":\"assistant\",\"content\":3}]",
        "[{\"role\":\"assistant\",\"tool_calls_json\":\"{}\"}]",
        "[{\"role\":\"user\",\"content\":\"no assistant\"}]",
    ] {
        assert!(messages(body).is_err(), "{body}");
    }
    let state = tempfile::tempdir().unwrap();
    let file = state.path().join("input.parquet");
    write(&file, &[row("one", "z")], Compression::SNAPPY);
    assert!(input::select(&file, &policy()).is_err());
    assert!(document::manifest(Default::default(), &policy(), "mutable-main", "").is_err());
    std::fs::write(&file, b"not parquet").unwrap();
    assert!(input::select(&file, &policy()).is_err());
}
#[test]
fn actual_cohort_cli_publishes_validated_manifest_and_refuses_without_output() {
    let state = tempfile::tempdir().unwrap();
    let file = state.path().join("input.parquet");
    let output = state.path().join("manifest.json");
    write(&file, &rows(), Compression::SNAPPY);
    let mut args = vec![
        "--dataset-file".into(),
        file.to_str().unwrap().into(),
        "--dataset-revision".into(),
        "a".repeat(40),
        "--output".into(),
        output.to_str().unwrap().into(),
        "--sessions-per-cohort".into(),
        "4".into(),
        "--min-turns".into(),
        "2".into(),
    ];
    for (key, values) in [
        ("--cohort", vec!["warmup", "1"]),
        ("--framework", vec!["z", "a", "m"]),
        ("--source-dataset", vec!["source"]),
    ] {
        for value in values {
            args.extend([key.into(), value.into()]);
        }
    }
    super::cli::run(&args).unwrap();
    let bytes = std::fs::read(&output).unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(doc["metadata"]["cohorts"]["1"]["trajectory_count"], 4);
    assert!(super::cli::run(&args).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), bytes);
    std::fs::remove_file(&output).unwrap();
    args[3] = "main".into();
    assert!(super::cli::run(&args).is_err());
    assert!(!output.exists());
}
