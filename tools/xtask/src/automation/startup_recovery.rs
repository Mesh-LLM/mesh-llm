mod coordinator;
mod options;
mod projection;
mod setup;
mod worker;
use crate::{
    command::DynResult,
    process::{self, retained::recovery::Stability},
};
use std::{
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        writeln!(
            crate::cli_output::stdout(),
            "automation startup-recovery BINARY MODEL; MESH_SPLIT_CERT_* configures workers, ports, budgets, recovery and inference"
        )?;
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    let prepared = setup::prepare(root, &options)?;
    let (requests, jobs) = std::sync::mpsc::sync_channel(1);
    let (results, responses) = std::sync::mpsc::sync_channel(1);
    let mut launches = prepared.launches.into_iter();
    let mut owner = coordinator::Owner {
        options: &options,
        seed: launches.next(),
        workers: launches.collect(),
        requests,
        responses,
        phase: coordinator::Phase::Invite,
        pending: false,
        next: Instant::now(),
        until: Instant::now() + options.startup,
        driver: 0,
        killed: String::new(),
        old_run: String::new(),
        stopped: 0,
        stability: Stability::new(options.stable, options.expected),
        completed: Vec::new(),
    };
    let cancellation = process::Cancellation::default();
    let (report, joined, completed) = std::thread::scope(|scope| {
        let worker_options = &options;
        let worker_cancel = &cancellation;
        let worker = scope.spawn(move || {
            while let Ok(request) = jobs.recv() {
                let facts = worker::work(request, worker_options, worker_cancel);
                if results.send(facts).is_err() {
                    break;
                }
            }
        });
        let report = super::retained_session::run(
            &mut owner,
            &process::Limits {
                execution: options.startup * (u32::try_from(options.workers).unwrap_or(15) + 2)
                    + options.recovery
                    + Duration::from_secs(300),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
        );
        let completed = std::mem::take(&mut owner.completed);
        cancellation.cancel();
        drop(owner);
        (report, worker.join().is_ok(), completed)
    });
    let success = joined
        && report
            .as_ref()
            .is_ok_and(|report| report.recovery_success());
    let mut rows = Vec::new();
    for name in completed {
        serde_json::to_writer(&mut rows, &serde_json::json!({"status":"PASS","name":name}))?;
        rows.push(b'\n');
    }
    serde_json::to_writer(
        &mut rows,
        &serde_json::json!({"status":if success{"PASS"}else{"FAIL"},"name":"split-startup-recovery-certification"}),
    )?;
    rows.push(b'\n');
    std::fs::write(prepared.work.join("result.jsonl"), rows)?;
    if let Ok(report) = &report {
        std::fs::write(
            prepared.work.join("session.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outcome":format!("{:?}",report.outcome),"rejection":report.rejection,"failure":format!("{:?}",report.failure)}),
            )?,
        )?;
        let members:Vec<_>=report.members.iter().map(|member|serde_json::json!({"name":String::from_utf8_lossy(member.member.name()),"generation":member.member.generation(),"pid":member.process.pid,
            "disposition":member.disposition.label(),"cleanup_complete":member.process.cleanup.complete,"forced":member.process.cleanup.forced})).collect();
        std::fs::write(
            prepared.work.join("processes.json"),
            serde_json::to_vec_pretty(&members)?,
        )?;
    }
    if prepared.process_owned {
        std::fs::remove_dir_all(prepared.process_root)?;
    }
    if success && !options.keep && options.work.is_none() {
        std::fs::remove_dir_all(&prepared.work)?;
    }
    if !success {
        return Err("split startup/recovery or cleanup failed; evidence retained".into());
    }
    writeln!(
        crate::cli_output::stdout(),
        "Split worker startup + recovery certification passed"
    )?;
    Ok(())
}
