mod checks;
mod coordinator;
mod options;
mod setup;
use crate::{command::DynResult, process};
use std::{io::Write, path::Path, time::Duration};

const CHECKS: &[&str] = &[
    "logging_restart_privacy",
    "logging_retention_cascade",
    "logging_trusted_local_rejection",
    "logging_sse_recovery",
    "logging_fail_open",
    "logging_fail_open_inference",
    "cleanup",
];
#[derive(thiserror::Error)]
#[error("logging recovery narrative failed; evidence retained")]
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
            "automation logging-recovery --current-binary PATH [--evidence-dir DIR] [--base-port PORT] [--max-wait SECONDS] [--deterministic-openai-endpoint URL] [--deterministic-openai-model MODEL] [--keep-logs] [--print-plan]"
        )?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    if options.plan {
        serde_json::to_writer(
            crate::cli_output::stdout(),
            &serde_json::json!({"script":"qa-logging-recovery.sh","current_binary":options.binary,"evidence_dir":options.evidence,"base_port":options.base,"max_wait_seconds":options.wait.as_secs(),
            "deterministic_openai_endpoint_supplied":!options.endpoint.is_empty(),"deterministic_openai_model":if options.endpoint.is_empty(){None}else{Some(&options.model)},"checks":CHECKS,
            "optional_plugin_behavior":{"without_endpoint":{"logging_fail_open_inference":"PREREQ"},"with_endpoint":{"logging_fail_open_inference":"execute"}},
            "evidence_files":["manifest.json","commands.jsonl","results.jsonl","summary.json","summary.md","logs/","status/","requests/","versions/"]}),
        )?;
        writeln!(crate::cli_output::stdout())?;
        return Ok(());
    }
    let prepared = setup::prepare(root, &options)?;
    let (requests, jobs) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut owner = coordinator::Owner {
        initial: Some(prepared.initial),
        restarted: Some(prepared.restarted),
        fail_open: Some(prepared.fail_open),
        phase: coordinator::Phase::InitialReady,
        requests,
        responses,
        pending: false,
        base: options.base,
        id: String::new(),
        model: (!options.endpoint.is_empty()).then(|| options.model.clone()),
        completed: Vec::new(),
        prerequisites: Vec::new(),
    };
    let cancellation = process::Cancellation::default();
    let (report, joined, completed, prerequisites) = std::thread::scope(|scope| {
        let directory = &prepared.directory;
        let worker_cancel = &cancellation;
        let worker = scope.spawn(move || {
            while let Ok(check) = jobs.recv() {
                let result = checks::execute(
                    check,
                    checks::Context {
                        root: &directory.join("requests"),
                        wait: options.wait,
                        cancellation: worker_cancel,
                    },
                );
                if results.send(result).is_err() {
                    break;
                }
            }
        });
        let report = super::retained_session::run(
            &mut owner,
            &process::Limits {
                execution: options.wait * 12 + Duration::from_secs(60),
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
        .map(|name| serde_json::json!({"status":"PASS","name":name}))
        .collect();
    rows.extend(
        prerequisites
            .into_iter()
            .map(|name| serde_json::json!({"status":"PREREQ","name":name})),
    );
    rows.push(serde_json::json!({"status":if success{"PASS"}else{"FAIL"},"name":"cleanup"}));
    let mut bytes = Vec::new();
    for row in &rows {
        serde_json::to_writer(&mut bytes, row)?;
        bytes.push(b'\n');
    }
    std::fs::write(prepared.directory.join("results.jsonl"), bytes)?;
    std::fs::write(
        prepared.directory.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"overall":if success{"pass"}else{"fail"},"results":rows,"evidence_dir":prepared.directory}),
        )?,
    )?;
    std::fs::write(
        prepared.directory.join("manifest.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"binary":options.binary,"base_port":options.base,"max_wait_seconds":options.wait.as_secs(),"deterministic_openai_endpoint_supplied":!options.endpoint.is_empty()}),
        )?,
    )?;
    if let Ok(report) = &report {
        std::fs::write(
            prepared.directory.join("session.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outcome":format!("{:?}",report.outcome),"rejection":report.rejection}),
            )?,
        )?;
        let members:Vec<_>=report.members.iter().map(|member|serde_json::json!({"name":String::from_utf8_lossy(member.member.name()),"generation":member.member.generation(),"pid":member.process.pid,"disposition":member.disposition.label(),"cleanup_complete":member.process.cleanup.complete})).collect();
        std::fs::write(
            prepared.directory.join("processes.json"),
            serde_json::to_vec_pretty(&members)?,
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
