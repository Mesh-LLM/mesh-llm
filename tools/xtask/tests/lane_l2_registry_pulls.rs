#[path = "../src/ci_operations/registry_pulls.rs"]
mod registry_pulls;

use registry_pulls::{Observation, Source, summarize};

fn observations(upstream: u32, depot: u32) -> Vec<Observation> {
    [Source::Upstream, Source::Depot]
        .into_iter()
        .flat_map(|source| {
            (1..=5).map(move |sample| Observation {
                source,
                sample,
                elapsed_ms: match source {
                    Source::Upstream => upstream,
                    Source::Depot => depot,
                },
                digest: format!("sha256:{}", "a".repeat(64)),
            })
        })
        .collect()
}

#[test]
fn requires_absolute_and_relative_improvement() {
    assert!(
        summarize(&observations(60_000, 40_000), 5)
            .unwrap()
            .eligible
    );
    assert!(
        !summarize(&observations(20_000, 14_000), 5)
            .unwrap()
            .eligible
    );
    assert!(
        !summarize(&observations(100_000, 89_000), 5)
            .unwrap()
            .eligible
    );
}

#[test]
fn rejects_digest_mismatch() {
    let mut values = observations(60_000, 40_000);
    values[9].digest = "sha256:b".into();
    assert!(summarize(&values, 5).is_err());
}

#[test]
fn rejects_duplicate_and_missing_samples() {
    let mut values = observations(60_000, 40_000);
    values[9].sample = 4;
    assert!(summarize(&values, 5).is_err());
    assert!(summarize(&values[..9], 5).is_err());
}

#[test]
fn preserves_half_millisecond_even_median() {
    let mut values = observations(60_000, 40_000);
    values.retain(|item| item.sample <= 2);
    values[1].elapsed_ms = 60_001;
    assert!((summarize(&values, 2).unwrap().upstream_median_ms - 60_000.5).abs() < f64::EPSILON);
}

#[test]
fn command_writes_reports_and_enforces_negative_evidence() {
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("observations");
    std::fs::create_dir(&source).unwrap();
    for (index, observation) in observations(20_000, 14_000).iter().enumerate() {
        std::fs::write(
            source.join(format!("{index}.json")),
            serde_json::to_vec(observation).unwrap(),
        )
        .unwrap();
    }
    let json = scratch.path().join("summary.json");
    let markdown = scratch.path().join("summary.md");
    let args = vec![
        source.display().to_string(),
        "--enforce".into(),
        "--json-out".into(),
        json.display().to_string(),
        "--markdown-out".into(),
        markdown.display().to_string(),
    ];
    let result = registry_pulls::run(&args).unwrap();
    assert_eq!(result.code, 1);
    assert!(result.stdout.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(json).unwrap()).unwrap();
    assert_eq!(report["eligible"], false);
    assert!(
        std::fs::read_to_string(markdown)
            .unwrap()
            .contains("| Adoption gate | fail |")
    );
}
