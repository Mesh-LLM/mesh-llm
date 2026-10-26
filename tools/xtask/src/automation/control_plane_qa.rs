mod coordinator;
mod http_checks;
mod options;
mod setup;
use crate::{command::DynResult, process};
use std::{io::Write, path::Path, time::Duration};

#[derive(thiserror::Error)]
#[error("mixed-version control narrative failed; evidence retained")]
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
            "automation control-plane-qa --current-binary PATH --released-binary PATH [--local-only] [--config-only] [--model REF] [--current-model REF] [--released-model REF] [--base-port PORT] [--max-wait SECONDS] [--stable-probes N] [--chat-max-time SECONDS] [--ctx-size TOKENS] [--skip-cargo-tests] [--require-public] [--print-plan] [--keep-logs] [--evidence-dir DIR]"
        )?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    if options.plan {
        serde_json::to_writer(
            crate::cli_output::stdout(),
            &serde_json::json!({"script":"qa-control-plane-mixed-version.sh","current_binary":options.current,"released_binary":options.released,
            "evidence_dir":options.evidence,"local_only":options.local,"config_only":options.config,"run_cargo_tests":options.cargo,"require_public":options.public_required,
            "checks":["mixed_version_private_mesh_both_directions","public_client_auto_when_enabled","control_status_privacy","explicit_owner_control_bootstrap","same_owner_get_config_and_scan_refresh","wrong_owner_rejection","current_lifecycle_acceptance","typed_legacy_unsupported","cleanup"]}),
        )?;
        writeln!(crate::cli_output::stdout())?;
        return Ok(());
    }
    let prepared = setup::prepare(root, &options)?;
    let (requests, jobs) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut owner = coordinator::Owner {
        steps: prepared.steps,
        requests,
        responses,
        checking: None,
        token: String::new(),
        endpoint: String::new(),
        completed: Vec::new(),
        prerequisites: Vec::new(),
        owner_available: false,
        current_help: String::new(),
        released_help: String::new(),
    };
    if !options.cargo {
        owner.prerequisites.push("config-cargo-tests".into());
    }
    let cancellation = process::Cancellation::default();
    let (report, joined, completed, prerequisites) = std::thread::scope(|scope| {
        let directory = &prepared.directory;
        let worker_options = &options;
        let worker_cancel = &cancellation;
        let worker = scope.spawn(move || {
            while let Ok(check) = jobs.recv() {
                let result = http_checks::execute(
                    check,
                    http_checks::Context {
                        options: worker_options,
                        directory: &directory.join("control"),
                        cancel: worker_cancel,
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
                execution: options.wait * 30
                    + options.chat * 4
                    + Duration::from_secs(if options.cargo { 7200 } else { 120 }),
                graceful_shutdown: Duration::from_secs(5),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 1_048_576,
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
        prepared.directory.join("manifest.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"current_binary":options.current,"released_binary":options.released,"base_port":options.base,"local_only":options.local,"config_only":options.config}),
        )?,
    )?;
    std::fs::write(
        prepared.directory.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"overall":if success{"pass"}else{"fail"},"results":rows,"evidence_dir":prepared.directory}),
        )?,
    )?;
    if let Ok(report) = &report {
        std::fs::write(
            prepared.directory.join("session.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outcome":format!("{:?}",report.outcome),"rejection":report.rejection,"failure":format!("{:?}",report.failure)}),
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
