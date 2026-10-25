use super::super::super::{
    Digest, ReceiptContext, SourceFamilyPlan, TestPackageInputs, WorkflowRun,
};
use super::super::{FamilyEvidence, FeedbackDraft, FeedbackState, verify};
use super::*;
use serde_json::json;
use std::{collections::BTreeSet, fs, path::Path};

const IDENTITY: &[u8] =
    include_bytes!("../../../../tests/migration_canary_receipts/fixtures/identity.json");

fn context(families: &[&str]) -> ReceiptContext {
    let models: Vec<_> = families
        .iter()
        .map(|family| {
            json!({"family":family,"class":"causal_generation",
        "certification_lanes":["chain","single-step","state-handoff"],"mmproj_artifact":null})
        })
        .collect();
    let rows: Vec<_> = families
        .iter()
        .enumerate()
        .map(|(index, family)| json!({"shard_index":index,"families":family}))
        .collect();
    let shards: Vec<_> = families
        .iter()
        .enumerate()
        .map(|(index, family)| json!({"shard_index":index,"families":[family]}))
        .collect();
    let plan = json!({"selected_models":models,"required_certification_lanes":["chain","single-step","state-handoff"],
        "github_matrix":{"include":rows},"shards":shards});
    ReceiptContext::from_test_package(
        TestPackageInputs {
            identity: serde_json::from_slice(IDENTITY).unwrap(),
            identity_sha256: Digest::of_bytes(IDENTITY),
            plan: SourceFamilyPlan::parse(&serde_json::to_vec(&plan).unwrap()).unwrap(),
        },
        WorkflowRun {
            run_id: "123".into(),
            run_attempt: "4".to_owned().try_into().unwrap(),
        },
    )
    .unwrap()
}
fn source(root: &Path, context: &ReceiptContext, family: &str, rows: &[serde_json::Value]) {
    let directory = root.join("sources").join(family);
    fs::create_dir_all(&directory).unwrap();
    let results: Vec<u8> = rows
        .iter()
        .flat_map(|row| {
            let mut bytes = serde_json::to_vec(row).unwrap();
            bytes.push(b'\n');
            bytes
        })
        .collect();
    fs::write(directory.join("results.jsonl"), &results).unwrap();
    let receipt = json!({"candidate":context.package.identity.candidate,"family":family,"identity_sha256":context.package.identity_sha256,
        "outcome":"failure","pass_id":context.package.identity.pass_id,"results_sha256":Digest::of_bytes(&results),
        "run_attempt":"4","run_id":"123","runner":"fixture"});
    fs::write(
        directory.join("receipt.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
}
fn failed(family: &str, lane: &str, note: &str) -> serde_json::Value {
    json!({"family":family,"outcomes":[{"name":lane,"status":"fail","note":note}]})
}
fn admitted(root: &Path, context: &ReceiptContext, families: &[&str]) -> VerifiedFeedback {
    let evidence = families
        .iter()
        .map(|name| {
            let family = (*name).to_owned().try_into().unwrap();
            FamilyEvidence::admit(context, &family, &root.join("sources").join(name)).unwrap()
        })
        .collect();
    let destination = root.join("feedback");
    FeedbackDraft::new(
        context,
        FeedbackState::CandidateRepairable,
        evidence,
        BTreeSet::new(),
        Vec::new(),
    )
    .unwrap()
    .publish(context, &destination)
    .unwrap();
    verify(context, &destination, FeedbackState::CandidateRepairable).unwrap()
}

#[test]
fn summary_groups_first_failed_lane_notes_and_sorts_families() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense", "hybrid"]);
    source(
        root.path(),
        &context,
        "dense",
        &[
            json!({"family":"dense","outcomes":[{"name":"chain","status":"pass"}]}),
            failed(
                "dense",
                "stage-replay",
                " VIEW geometry exceeds source storage ",
            ),
            failed("dense", "later", "must not appear"),
        ],
    );
    source(
        root.path(),
        &context,
        "hybrid",
        &[failed(
            "hybrid",
            "stage-replay",
            "RESHAPE changes element count",
        )],
    );
    let summary = admitted(root.path(), &context, &["hybrid", "dense"])
        .summary()
        .unwrap();
    assert!(summary.contains("- stage-replay (2):\n  dense: VIEW geometry exceeds source storage\n  hybrid: RESHAPE changes element count\n"));
    assert!(!summary.contains("later"));
    assert!(summary.starts_with(HEADING));
    assert!(summary.ends_with(FOOTER));
}

#[test]
fn summary_uses_first_lexical_log_trace_and_handles_invalid_utf8() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense"]);
    source(
        root.path(),
        &context,
        "dense",
        &[failed("dense", "chain", " \n ")],
    );
    let directory = root.path().join("sources/dense");
    fs::write(
        directory.join("a.log"),
        b"ordinary status\n  PANIC: \xfffirst useful trace\nERROR later\n",
    )
    .unwrap();
    fs::write(
        directory.join("z.log"),
        b"ERROR must not win lexical ordering\n",
    )
    .unwrap();
    let summary = admitted(root.path(), &context, &["dense"])
        .summary()
        .unwrap();
    assert!(summary.contains("dense: a.log: PANIC: \u{fffd}first useful trace\n"));
    assert!(!summary.contains("ERROR later"));
    assert!(!summary.contains("must not win"));
}

#[test]
fn summary_trace_limit_counts_unicode_characters_and_sanitizes_controls() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense"]);
    let note = format!("A\u{1b}\n{}", "界".repeat(350));
    source(
        root.path(),
        &context,
        "dense",
        &[failed("dense", "stage\nreplay", &note)],
    );
    let summary = admitted(root.path(), &context, &["dense"])
        .summary()
        .unwrap();
    let trace = summary
        .lines()
        .find_map(|line| line.strip_prefix("  dense: "))
        .unwrap();
    assert_eq!(trace.chars().count(), 300);
    assert!(trace.starts_with("A  "));
    assert!(summary.contains("- stage replay (1):"));
    assert!(!summary.contains('\u{1b}'));
}

#[test]
fn summary_log_scan_budget_is_shared_across_files_and_bounds_long_lines() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense"]);
    source(
        root.path(),
        &context,
        "dense",
        &[failed("dense", "chain", "")],
    );
    let directory = root.path().join("sources/dense");
    let mut bytes = vec![b'x'; usize::try_from(MAXIMUM_LOG_SCAN).unwrap()];
    bytes.extend_from_slice(b"\nERROR beyond scan budget\n");
    fs::write(directory.join("a.log"), bytes).unwrap();
    fs::write(directory.join("z.log"), b"ERROR beyond shared budget\n").unwrap();
    let summary = admitted(root.path(), &context, &["dense"])
        .summary()
        .unwrap();
    assert!(summary.contains("dense: see family evidence\n"));
    assert!(!summary.contains("beyond"));
}

#[test]
fn summary_reads_owned_snapshot_after_all_original_evidence_is_replaced() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense"]);
    source(
        root.path(),
        &context,
        "dense",
        &[failed("dense", "chain", "original admitted failure")],
    );
    let verified = admitted(root.path(), &context, &["dense"]);
    fs::remove_dir_all(root.path().join("sources")).unwrap();
    fs::remove_dir_all(root.path().join("feedback")).unwrap();
    fs::create_dir(root.path().join("feedback")).unwrap();
    fs::write(
        root.path().join("feedback/feedback.json"),
        b"forged replacement",
    )
    .unwrap();
    let summary = verified.summary().unwrap();
    assert!(summary.contains("dense: original admitted failure"));
    assert!(!summary.contains("forged"));
}

#[test]
fn summary_caps_total_output_with_marker_and_keeps_final_gate_instruction() {
    let root = tempfile::tempdir().unwrap();
    let names: Vec<_> = (0..180).map(|index| format!("family-{index:03}")).collect();
    let families: Vec<_> = names.iter().map(String::as_str).collect();
    let context = context(&families);
    for family in &families {
        source(
            root.path(),
            &context,
            family,
            &[failed(family, "stage-replay", &"界".repeat(300))],
        );
    }
    let summary = admitted(root.path(), &context, &families)
        .summary()
        .unwrap();
    assert!(summary.len() <= MAXIMUM_OUTPUT);
    assert!(summary.contains(TRUNCATED));
    assert!(summary.ends_with(FOOTER));
    assert!(summary.contains("stage-replay (180)"));
    assert!(summary.contains("family-000"));
    assert!(!summary.contains("family-179"));
}

#[test]
fn summary_defaults_missing_failed_lane_and_prefers_note_over_logs() {
    let root = tempfile::tempdir().unwrap();
    let context = context(&["dense"]);
    source(
        root.path(),
        &context,
        "dense",
        &[json!({"family":"dense","outcomes":[{"status":"fail","note":"note wins"}]})],
    );
    fs::write(
        root.path().join("sources/dense/phase.log"),
        b"ERROR lower priority log",
    )
    .unwrap();
    let summary = admitted(root.path(), &context, &["dense"])
        .summary()
        .unwrap();
    assert!(summary.contains("- unclassified (1):\n  dense: note wins\n"));
    assert!(!summary.contains("lower priority"));
}
