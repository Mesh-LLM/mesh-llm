mod checks;
mod coordinator;
mod options;
mod setup;
use crate::{command::DynResult, process};
use std::{io::Write, path::Path, time::Duration};

const CHECKS: &[&str] = &[
    "prereq.current-binary",
    "prereq.owner-identity",
    "zero_model_serve_ready",
    "runtime_mode_serve",
    "runtime_mode_on_demand",
    "best_effort_startup",
    "fail_fast_startup",
    "runtime_load_model",
    "runtime_unload_model",
    "runtime_ensure_model",
    "runtime_drain_model",
    "activity_override",
    "privacy_no_raw_data",
    "clean_process_teardown",
];

#[derive(thiserror::Error)]
#[error("daemon lifecycle narrative failed; evidence retained")]
struct Failure {
    report: Result<process::retained::Report<String>, super::retained_session::Error<String>>,
    joined: bool,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            crate::cli_output::stdout(),
            "automation daemon-lifecycle --current-binary PATH [--evidence-dir DIR] [--base-port PORT] [--max-wait SECONDS] [--keep-logs] [--print-plan]"
        )?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    if options.plan {
        serde_json::to_writer(
            crate::cli_output::stdout(),
            &serde_json::json!({"script":"qa-runtime-daemon-lifecycle.sh","current_binary":options.binary,"evidence_dir":options.evidence,"base_port":options.base,"max_wait_seconds":options.wait.as_secs(),"checks":CHECKS}),
        )?;
        writeln!(crate::cli_output::stdout())?;
        return Ok(());
    }
    let prepared = setup::prepare(root, &options)?;
    let (requests, jobs) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut owner = coordinator::Owner {
        version: Some(prepared.version),
        auth: Some(prepared.auth),
        launches: prepared.launches.into(),
        phase: coordinator::Phase::Version,
        index: 0,
        current: None,
        requests,
        responses,
        pending: false,
        base: options.base,
        wait: options.wait,
        completed: Vec::new(),
        prerequisites: Vec::new(),
        owner_available: false,
        usage_exit: false,
    };
    let cancellation = process::Cancellation::default();
    let (report, joined, completed, prerequisites) = std::thread::scope(|scope| {
        let directory = &prepared.directory;
        let worker_cancel = &cancellation;
        let worker = scope.spawn(move || {
            while let Ok(check) = jobs.recv() {
                let result = checks::execute(
                    check,
                    &directory.join("control"),
                    options.wait,
                    worker_cancel,
                );
                if results.send(result).is_err() {
                    break;
                }
            }
        });
        let report = super::retained_session::run(
            &mut owner,
            &process::Limits {
                execution: options.wait * 20 + Duration::from_secs(60),
                graceful_shutdown: Duration::from_secs(5),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
        );
        let completed = std::mem::take(&mut owner.completed);
        let prerequisites = std::mem::take(&mut owner.prerequisites);
        cancellation.cancel();
        drop(owner);
        (report, worker.join().is_ok(), completed, prerequisites)
    });
    let success = joined && report.as_ref().is_ok_and(|report| report.success());
    let mut rows: Vec<_> = completed
        .into_iter()
        .map(|name| serde_json::json!({"status":"PASS","name":name,"message":"check completed"}))
        .collect();
    rows.extend(prerequisites.into_iter().map(|name|serde_json::json!({"status":"PREREQ","name":name,"message":"local prerequisite unavailable"})));
    rows.push(serde_json::json!({"status":if success{"PASS"}else{"FAIL"},"name":"clean_process_teardown","message":"retained session finalized"}));
    let mut bytes = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut bytes, row)?;
        bytes.push(b'\n');
    }
    std::fs::write(prepared.directory.join("results.jsonl"), bytes)?;
    std::fs::write(
        prepared.directory.join("manifest.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"binary":options.binary,"base_port":options.base,"max_wait_seconds":options.wait.as_secs()}),
        )?,
    )?;
    std::fs::write(
        prepared.directory.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"overall":if success{"pass"}else{"fail"},"results":rows,"evidence_dir":prepared.directory}),
        )?,
    )?;
    if let Ok(report) = &report {
        let receipts:Vec<_>=report.members.iter().map(|member|serde_json::json!({"name":String::from_utf8_lossy(member.member.name()),"generation":member.member.generation(),"pid":member.process.pid,
            "disposition":member.disposition.label(),"exit_code":member.process.status.and_then(|status|status.code()),"cleanup_complete":member.process.cleanup.complete,
            "expected_exit":member.completion.as_ref().map(|receipt|serde_json::json!({"status":receipt.status,"allowed":receipt.statuses,"deadline_ms":receipt.deadline.as_millis(),"elapsed_ms":receipt.elapsed.as_millis()}))})).collect();
        std::fs::write(
            prepared.directory.join("processes.json"),
            serde_json::to_vec_pretty(&receipts)?,
        )?;
        std::fs::write(
            prepared.directory.join("session.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outcome":format!("{:?}",report.outcome),"rejection":report.rejection}),
            )?,
        )?;
    }
    if success && !options.keep {
        std::fs::remove_dir_all(prepared.directory.join("state"))?;
    }
    if !success {
        return Err(Failure { report, joined }.into());
    }
    writeln!(
        crate::cli_output::stdout(),
        "{}",
        prepared.directory.display()
    )?;
    Ok(())
}
