//! Every frozen planner case and lane: the digest-bound lane plan, the real
//! lane workflow graph and a `needs` document where exactly the planned jobs
//! succeeded validate cleanly, including the required protected native producers.

use crate::support::{
    Call, Golden, LANES, TestResult, expected_jobs, lane_workflow, needs, repository_root, text,
    validate_entrypoints,
};
use serde_json::Value;
use std::collections::BTreeSet;

fn planned(jobs: &Value) -> Vec<&str> {
    jobs.as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

#[test]
fn migration_ci_graph_every_case_and_lane_validates_planned_success() -> TestResult {
    // Given: 38 frozen plans across draft, ready, main and manual-full
    // profiles, each projected into the five lanes.
    let cases = expected_jobs()?;
    assert_eq!(cases.len(), 38, "frozen case count");
    let mut profiles = BTreeSet::new();
    let (mut selected, mut absent) = (0, 0);
    for (case, lanes) in &cases {
        let golden = Golden::load(case)?;
        profiles.insert(golden.output("profile").to_owned());
        for lane in LANES {
            let jobs = planned(&lanes[lane]);
            if jobs.is_empty() {
                absent += 1;
            } else {
                selected += 1;
            }
            // When: the lane summary validates planned-success results
            // against the digest-bound plan and the lane's parsed graph.
            let call = Call::lane(&golden.lane_plan(lane), &needs(lane, &jobs))
                .with("--workflow", &lane_workflow(lane))
                .with_text("--plan-digest", golden.output("plan_digest"))
                .with_text("--canonical-plan", golden.output("plan_json"));
            let output = call.run()?;
            // Then: it succeeds silently.
            assert_eq!(
                (output.status.code(), text(&output.stderr)),
                (Some(0), String::new()),
                "{case}/{lane}"
            );
        }
    }
    let expected: BTreeSet<String> = ["main", "manual-full", "pr-draft", "pr-ready"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(profiles, expected);
    assert!(
        selected > 0 && absent > 0,
        "{selected} selected, {absent} absent"
    );
    println!("validated {selected} selected and {absent} absent lane projections");
    Ok(())
}

#[test]
fn migration_ci_graph_unplanned_skip_passes_and_missing_rows_default_to_empty() -> TestResult {
    // Given: a quality plan with an unplanned, skipped runner contract, and a
    // Windows plan whose optional matrices are omitted entirely.
    let quality =
        r#"{"lane":"quality","required":true,"required_slices":["quality"],"matrices":{}}"#;
    let windows = r#"{"lane":"windows","required":false,"required_slices":[],"matrices":{}}"#;
    // When: each is validated.
    let skipped = Call::lane(
        quality,
        r#"{"quality":{"result":"success"},"runner_contract":{"result":"skipped"},"authority_source":{"result":"success"},"authority_linux_x64":{"result":"success"}}"#,
    )
    .run()?;
    let empty = Call::lane(windows, "{}").run()?;
    // Then: both pass.
    assert_eq!(skipped.status.code(), Some(0), "{}", text(&skipped.stderr));
    assert_eq!(empty.status.code(), Some(0), "{}", text(&empty.stderr));
    Ok(())
}

#[test]
fn migration_ci_graph_windows_smoke_requires_success_when_selected() -> TestResult {
    // Given: a Windows projection with only the smoke matrix selected.
    let plan = r#"{"lane":"windows","required":true,"required_slices":[],"matrices":{"smoke":[{"id":"windows-cpu"}]}}"#;
    // When: the smoke consumer succeeds.
    let output = Call::lane(plan, r#"{"product_smoke":{"result":"success"}}"#).run()?;
    // Then: the lane accepts the selected consumer.
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    Ok(())
}

#[test]
fn migration_ci_graph_real_entrypoints_keep_five_native_lanes() -> TestResult {
    // Given: the checked-in PR, main and manual-full entrypoints.
    let workflows = repository_root().join(".github/workflows");
    // When: their parsed graphs are validated.
    let output = validate_entrypoints(&workflows)?;
    // Then: five independent PR and main lanes and a dispatch-only controller.
    assert_eq!(output.status.code(), Some(0), "{}", text(&output.stderr));
    Ok(())
}

#[test]
fn lane_help_names_current_command_and_complete_evidence_options() -> TestResult {
    for flag in ["-h", "--help"] {
        let output = Call::raw(&[flag]).run()?;
        assert_eq!(output.status.code(), Some(0));
        assert!(output.stderr.is_empty());
        let help = text(&output.stdout);
        assert!(help.starts_with("usage: cargo xtool ci validate-lane "));
        for option in [
            "--lane-plan",
            "--needs",
            "--workflow",
            "--plan-digest",
            "--canonical-plan",
        ] {
            assert!(help.contains(option), "{option}");
        }
    }
    Ok(())
}
