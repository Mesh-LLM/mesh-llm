use super::run_pass;
use crate::ci_operations::{
    ci_metrics_input::load_runs,
    ci_metrics_normalize::normalize_run,
    ci_metrics_observe::{contaminated, observation},
};

#[test]
fn ci_metrics_terminal_and_step_correlations_reuse_one_observation_per_executed_job() {
    let raw = load_runs(include_bytes!(
        "../../tests/fixtures/ci_operations/ci_metrics/inputs/sample_runs.json"
    ))
    .unwrap();
    let runs = raw
        .iter()
        .map(normalize_run)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let pass = run_pass(&runs, "success").unwrap();
    let expected = runs
        .iter()
        .flat_map(|run| &run.jobs)
        .filter(|job| job.conclusion != "skipped")
        .count();
    assert_eq!(pass.observations.len(), expected);
    for run in &runs {
        for job in run.jobs.iter().filter(|job| job.conclusion != "skipped") {
            assert_eq!(
                pass.observations
                    .iter()
                    .filter(|sample| std::ptr::eq(sample.job, job))
                    .count(),
                1
            );
        }
    }
    for step in &pass.steps {
        assert!(std::ptr::eq(
            pass.observations[step.observation].job,
            step.job
        ));
    }
    for (index, run) in runs.iter().enumerate() {
        let terminal = run
            .jobs
            .iter()
            .filter(|job| job.conclusion != "skipped")
            .max_by_key(|job| job.completed)
            .unwrap();
        let sample = pass
            .observations
            .iter()
            .find(|sample| std::ptr::eq(sample.job, terminal))
            .unwrap();
        assert_eq!(pass.terminal_queue[index], sample.queue);
        assert_eq!(pass.terminal_runner_queue[index], sample.runner_queue);
        assert_eq!(pass.terminal_execution[index], sample.duration);
    }
}

#[test]
fn ci_metrics_dependency_wait_is_not_runner_capacity_contamination() {
    let raw = load_runs(
        br#"[{"status":"completed","conclusion":"success","jobs":[{
        "id":1,"name":"dependency build","conclusion":"success",
        "created_at":"2026-07-06T00:00:05Z","started_at":"2026-07-06T00:08:35Z",
        "dependency_ready_at":"2026-07-06T00:08:25Z","completed_at":"2026-07-06T00:09:35Z"
    }]}]"#,
    )
    .unwrap();
    let run = normalize_run(&raw[0]).unwrap();
    let sample = observation(&run, &run.jobs[0]);
    assert_eq!(sample.queue, Some(510.0));
    assert_eq!(sample.runner_queue, Some(10.0));
    assert_eq!(sample.dependency_wait, Some(500.0));
    assert!(!contaminated(sample.runner_queue));
    assert!(contaminated(sample.queue));
}

#[test]
fn ci_metrics_duplicate_ids_preserve_exact_terminal_job_observation() {
    let raw = load_runs(
        br#"[{"status":"completed","conclusion":"success","jobs":[{
        "id":1,"name":"terminal","conclusion":"success",
        "created_at":"2026-07-06T00:00:00Z","started_at":"2026-07-06T00:00:02Z",
        "completed_at":"2026-07-06T00:10:00Z"
    },{
        "id":1,"name":"earlier duplicate","conclusion":"success",
        "created_at":"2026-07-06T00:00:00Z","started_at":"2026-07-06T00:04:00Z",
        "completed_at":"2026-07-06T00:05:00Z"
    }]}]"#,
    )
    .unwrap();
    let run = normalize_run(&raw[0]).unwrap();
    let pass = run_pass(std::slice::from_ref(&run), "success").unwrap();
    assert_eq!(pass.observations.len(), 2);
    assert_eq!(pass.terminal_queue, [Some(2.0)]);
    assert_eq!(pass.terminal_runner_queue, [Some(2.0)]);
    assert_eq!(pass.terminal_execution, [Some(598.0)]);
}
