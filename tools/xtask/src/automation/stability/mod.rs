mod agents;
mod cases;
mod evidence;
mod kv_cache;
mod kv_cache_probe;
mod kv_certification;
mod kv_conversation;
mod kv_mode;
mod kv_native_logs;
mod kv_options;
mod kv_overlap;
mod kv_plans;
mod kv_reports;
mod kv_requests;
mod kv_tool_calls;
mod kv_transcripts;
mod options;
mod plans;
mod release_attestation;
mod reports;
mod requests;
mod responses;
mod sse;
mod surface;
mod tool_calls;
mod tool_turns;
mod transport;

use crate::{
    command::DynResult,
    command_interrupt::Interrupt,
    repository::{RepositoryRoot, check_report::CheckReport},
};
use options::{Mode, Options};
use reports::{CommandRow, CommandStatus};
use std::{
    path::Path,
    time::{Duration, Instant, SystemTime},
};

pub(crate) use options::USAGE;

pub(crate) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    if args.first().is_some_and(|mode| mode == "kv-tool-loop") {
        return kv_mode::run(&args[1..]);
    }
    let help = args == ["--help"]
        || matches!(args, [mode, flag] if matches!(mode.as_str(), "nightly" | "tool-call") && flag == "--help");
    if help {
        let usage = if args == ["--help"] {
            format!("{USAGE}\n{}\n", kv_options::USAGE)
        } else {
            format!("{USAGE}\n")
        };
        return CheckReport::success(usage).emit();
    }
    let mut options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => return CheckReport::usage(USAGE, &error).emit(),
    };
    if !options.output.is_absolute() {
        options.output = std::env::current_dir()?.join(&options.output);
    }
    let plan = plans::build(&options);
    if options.plan {
        return crate::command::print_json(&plan);
    }
    if cfg!(windows) && !options.agents.is_empty() {
        return CheckReport::usage(USAGE, "optional agent smoke adapters require Unix Bash").emit();
    }
    let root = if options.agents.is_empty() {
        None
    } else {
        Some(RepositoryRoot::resolve(root)?)
    };
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let http = transport::Http::new(
        options.base.clone(),
        options.timeout,
        cancellation.clone(),
        "mesh-llm-agent-probe",
    )?;
    let probes = if options.mode == Mode::Nightly {
        let http = transport::Http::new(
            options.base.clone(),
            options.timeout,
            cancellation.clone(),
            "mesh-llm-stability",
        )?;
        runtime.block_on(surface::run(&http, &options))
    } else {
        Vec::new()
    };
    let started = Instant::now();
    let tools = runtime.block_on(tool_turns::run(&http, &options));
    let tool_elapsed = transport::millis(started);
    runtime.shutdown_timeout(Duration::from_secs(1));
    let mut commands = Vec::new();
    if options.mode == Mode::Nightly {
        commands.push(tool_command(&tools, tool_elapsed));
    }
    if let Some(root) = root.as_ref() {
        commands.extend(agents::run(root.as_path(), &options, &cancellation)?);
    }
    let attestation = release_attestation::inspect(&options);
    let signal_code = finish(interrupt)?;
    let cancelled = cancellation.is_cancelled() || signal_code.is_some();
    let tool_ok = !cancelled && !tools.is_empty() && tools.iter().all(|row| row.ok);
    let code = if options.mode == Mode::ToolCall {
        evidence::jsonl(&options.output, &tools)?;
        if tool_ok { 0 } else { 1 }
    } else {
        let summary = reports::summarize(&commands, &probes, &attestation, cancelled);
        let mut manifest = plan;
        manifest["created_at"] = serde_json::json!(timestamp()?);
        write_nightly(&options, &manifest, &commands, &probes, &tools, &summary)?;
        if summary.counts.ok { 0 } else { 1 }
    };
    CheckReport {
        stdout: format!(
            "stability: {}\nevidence: {}\n",
            if code == 0 { "PASS" } else { "FAIL" },
            options.output.display()
        ),
        stderr: String::new(),
        code: signal_code.unwrap_or(code),
    }
    .emit()
}

fn tool_command(tools: &[cases::Case], elapsed_ms: u64) -> CommandRow {
    let ok = !tools.is_empty() && tools.iter().all(|row| row.ok);
    CommandRow {
        name: "tool-call-reliability".into(),
        status: if ok {
            CommandStatus::Pass
        } else {
            CommandStatus::Fail
        },
        exit_code: if ok { 0 } else { 1 },
        elapsed_ms,
        log: "logs/tool-call-reliability.log".into(),
        detail: format!(
            "{}/{} phases passed",
            tools.iter().filter(|row| row.ok).count(),
            tools.len()
        ),
    }
}

fn write_nightly(
    options: &Options,
    manifest: &serde_json::Value,
    commands: &[CommandRow],
    probes: &[cases::Case],
    tools: &[cases::Case],
    summary: &reports::Summary<'_>,
) -> DynResult<()> {
    let output = &options.output;
    evidence::json(&output.join("manifest.json"), manifest)?;
    evidence::jsonl(&output.join("commands.jsonl"), commands)?;
    evidence::jsonl(&output.join("results.jsonl"), probes)?;
    evidence::jsonl(
        &output.join("agent-tool-call-reliability/results.jsonl"),
        tools,
    )?;
    evidence::json(
        &output.join("release-attestation.json"),
        summary.release_attestation,
    )?;
    evidence::json(&output.join("summary.json"), summary)?;
    evidence::bytes(
        &output.join("summary.md"),
        evidence::markdown(summary, probes, commands).as_bytes(),
    )?;
    if let Some(tool) = commands.first() {
        evidence::bytes(&output.join(&tool.log), tool.detail.as_bytes())?;
    }
    Ok(())
}

fn timestamp() -> DynResult<String> {
    let time = crate::ci_operations::ci_metrics_time::Instant::from_system_time(SystemTime::now())
        .ok_or("stability timestamp out of range")?;
    Ok(crate::ci_operations::ci_metrics_time::isoformat(time))
}

#[cfg(unix)]
fn finish(interrupt: Interrupt) -> DynResult<Option<i32>> {
    Ok(interrupt.finish_signal()?.map(|signal| 128 + signal))
}

#[cfg(windows)]
fn finish(interrupt: Interrupt) -> DynResult<Option<i32>> {
    match interrupt.finish() {
        Ok(()) => Ok(None),
        Err(crate::command_interrupt::Reason::Interrupted) => Ok(Some(130)),
        Err(error) => Err(error.into()),
    }
}
