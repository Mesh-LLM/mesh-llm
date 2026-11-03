use super::*;
use crate::ci_operations::ci_metrics_value::parse;
use serde_json::{Value as Json, json};

fn report(provider: &str, first_id: u64) -> Json {
    let runs: Vec<Json> = (first_id..first_id + 3)
        .map(|id| {
            json!({
                "id": id, "attempt": 1, "conclusion": "success",
                "workflow": "Linux", "event": "push",
                "head_sha": "0123456789abcdef0123456789abcdef01234567",
                "plan_profile": "full", "plan_digest": "plan-1",
                "change_class": "runtime", "runner_image": "image-1",
                "toolchain_epoch": "toolchain-1", "cache_mode": "cold"
            })
        })
        .collect();
    json!({
        "selection": {"skipped_runs": {}}, "runs": runs,
        "jobs": {
            "by_runner": [{"provider": provider, "operating_system": "linux",
                "architecture": "amd64", "runner_role": "linux-cpu", "job_names": ["build"]}],
            "comparison_cohort": {
                "providers": [provider], "job_names": ["build"],
                "dimensions": {"operating_system": ["linux"], "architecture": ["amd64"],
                    "runner_role": ["linux-cpu"]},
                "runner_queue_seconds": {"count": 3, "p95": 10},
                "execution_seconds": {"p95": 100}
            }
        }
    })
}

fn recommendation_for(baseline: &Json, candidate: &Json) -> String {
    let base = parse(&serde_json::to_vec(baseline).unwrap()).unwrap();
    let next = parse(&serde_json::to_vec(candidate).unwrap()).unwrap();
    display(field(&compare_reports(&base, &next), "recommendation"))
}

#[test]
fn independent_matched_successful_runs_only_request_review() {
    let baseline = report("github-hosted", 1);
    let candidate = report("depot", 10);
    assert_eq!(recommendation_for(&baseline, &candidate), "review");
}

#[test]
fn insufficient_failed_duplicate_or_excluded_runs_hold() {
    let baseline = report("github-hosted", 1);
    let candidate = report("depot", 10);
    let mut variants = Vec::new();
    let mut fewer = candidate.clone();
    fewer["runs"].as_array_mut().unwrap().pop();
    variants.push(fewer);
    let mut failed = candidate.clone();
    failed["runs"][0]["conclusion"] = json!("failure");
    variants.push(failed);
    let mut rerun = candidate.clone();
    rerun["runs"][0]["attempt"] = json!(2);
    variants.push(rerun);
    let mut duplicate = candidate.clone();
    duplicate["runs"][1]["id"] = json!(10);
    variants.push(duplicate);
    let mut reused = candidate.clone();
    reused["runs"][0]["id"] = json!(1);
    variants.push(reused);
    let mut excluded = candidate.clone();
    excluded["selection"]["skipped_runs"]["conclusion_failure"] = json!(1);
    variants.push(excluded);
    for variant in variants {
        assert_eq!(recommendation_for(&baseline, &variant), "hold");
    }
}

#[test]
fn mismatched_or_missing_source_plan_cache_and_dimensions_hold() {
    let baseline = report("github-hosted", 1);
    let candidate = report("depot", 10);
    for key in RUN_IDENTITY {
        let mut missing = candidate.clone();
        missing["runs"][0][key] = Json::Null;
        assert_eq!(
            recommendation_for(&baseline, &missing),
            "hold",
            "missing {key}"
        );
        let mut changed = candidate.clone();
        for run in changed["runs"].as_array_mut().unwrap() {
            run[key] = json!(if key == "head_sha" {
                "fedcba9876543210fedcba9876543210fedcba98"
            } else {
                "different"
            });
        }
        assert_eq!(
            recommendation_for(&baseline, &changed),
            "hold",
            "changed {key}"
        );
    }
    let mut missing_role = candidate.clone();
    missing_role["jobs"]["by_runner"][0]["runner_role"] = Json::Null;
    assert_eq!(recommendation_for(&baseline, &missing_role), "hold");
    let mut extra_job = candidate.clone();
    extra_job["jobs"]["comparison_cohort"]["job_names"] = json!(["build", "test"]);
    assert_eq!(recommendation_for(&baseline, &extra_job), "hold");
}

#[test]
fn swapping_job_platforms_does_not_count_as_matching_dimensions() {
    let mut baseline = report("github-hosted", 1);
    let mut candidate = report("depot", 10);
    for (report, provider, swapped) in [
        (&mut baseline, "github-hosted", false),
        (&mut candidate, "depot", true),
    ] {
        report["jobs"]["comparison_cohort"]["job_names"] = json!(["build", "test"]);
        report["jobs"]["comparison_cohort"]["dimensions"]["architecture"] =
            json!(["amd64", "arm64"]);
        report["jobs"]["by_runner"] = json!([
            {"provider":provider,"operating_system":"linux","architecture":"amd64",
                "runner_role":"linux-cpu","job_names":[if swapped {"test"} else {"build"}]},
            {"provider":provider,"operating_system":"linux","architecture":"arm64",
                "runner_role":"linux-cpu","job_names":[if swapped {"build"} else {"test"}]}
        ]);
    }
    assert_eq!(recommendation_for(&baseline, &candidate), "hold");
}

#[test]
fn queue_contamination_still_requests_rollback_without_qualification() {
    let baseline = report("github-hosted", 1);
    let mut candidate = report("depot", 10);
    candidate["runs"][0]["conclusion"] = json!("failure");
    candidate["jobs"]["comparison_cohort"]["runner_queue_seconds"]["p95"] = json!(300);
    assert_eq!(recommendation_for(&baseline, &candidate), "rollback");
}
