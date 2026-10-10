use super::{
    recorded_requests::{Selection, Turn},
    trajectory_execution::exchange,
};
use crate::command::DynResult;
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use std::{
    io::Write,
    time::{Duration, Instant},
};

pub(super) use super::cell_workload::Workload;

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix execute-cell --input PATH --requests-output PATH --summary-output PATH",
        values: &["--input", "--requests-output", "--summary-output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input = parsed.last("--input").ok_or("missing --input")?;
    let requests = parsed
        .last("--requests-output")
        .ok_or("missing --requests-output")?;
    let summary = parsed
        .last("--summary-output")
        .ok_or("missing --summary-output")?;
    let workload: Workload = serde_json::from_slice(&std::fs::read(input)?)?;
    if !workload.following_cells.is_empty()
        || workload.runtime_context.is_some()
        || workload.model_pin.is_some()
        || workload.minimum_recurrent_restored_tokens.is_some()
    {
        return Err("lifecycle qualification requires the retained server-cell owner".into());
    }
    run_workload(
        &workload,
        std::path::Path::new(requests),
        std::path::Path::new(summary),
    )
}

pub(super) fn run_workload(
    workload: &Workload,
    requests: &std::path::Path,
    summary: &std::path::Path,
) -> DynResult<()> {
    if measure(workload, requests, summary)? {
        Ok(())
    } else {
        Err("cell execution failed; raw requests and summary retained".into())
    }
}

pub(super) fn measure(
    workload: &Workload,
    requests: &std::path::Path,
    summary: &std::path::Path,
) -> DynResult<bool> {
    workload.validate()?;
    let phase = if workload.qualification_probe {
        super::progress::Phase::Preflight
    } else if workload.warmup_turns.is_some() {
        super::progress::Phase::Warmup
    } else {
        super::progress::Phase::Cell
    };
    let started = Instant::now();
    let mut progress = super::progress::Record::boundary(
        phase,
        super::progress::Event::Started,
        summary.display().to_string(),
    );
    progress.concurrency = Some(workload.concurrency);
    progress.sessions = Some(workload.trajectories.len());
    progress.turns = Some(workload.warmup_turns.unwrap_or_else(|| {
        if workload.mode() == super::replay_profile::Mode::All {
            workload
                .trajectories
                .iter()
                .map(|trajectory| {
                    trajectory
                        .messages
                        .iter()
                        .filter(|message| {
                            message.get("role").and_then(serde_json::Value::as_str)
                                == Some("assistant")
                        })
                        .count()
                })
                .sum()
        } else {
            workload.trajectories.len()
        }
    }));
    super::progress::emit(&progress)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut file = std::fs::File::create(requests)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = runtime.block_on(execute(
        workload,
        &mut file,
        interrupt.cancellation(),
        &progress,
    ));
    let finish = interrupt.finish();
    let records = result?;
    let mut evidence = super::replay_profile::summarize(
        &workload.trajectories,
        &records,
        workload.concurrency,
        workload.mode(),
    )?;
    evidence["model_id"] = workload.model.clone().into();
    if let Some(qualification) = &workload.measured_prefix {
        let problems = super::measured_prefix::problems(&records, qualification);
        if !problems.is_empty() {
            evidence["acceptance"]["passed"] = false.into();
            evidence["acceptance"]["problems"]
                .as_array_mut()
                .ok_or("missing acceptance problems")?
                .extend(problems.into_iter().map(serde_json::Value::String));
        }
    }
    if let Some(budget) = &workload.eligibility {
        super::cell_qualification::apply(
            &mut evidence,
            (&workload.trajectories, &records),
            budget,
        )?;
    }
    if let Some(turns) = workload.warmup_turns {
        evidence["warmup"] = true.into();
        evidence["acceptance"] = serde_json::json!({
            "passed": records.len() == turns && records.iter().all(|record| record.get("error").is_none()),
            "expected_turns": turns
        });
    }
    crate::command::write_json_file(summary, &evidence)?;
    progress.event = super::progress::Event::Completed;
    progress.elapsed_seconds = started.elapsed().as_secs_f64();
    progress.turns = Some(records.len());
    progress.outcome = Some(if evidence["acceptance"]["passed"] == true {
        super::progress::Outcome::Success
    } else {
        super::progress::Outcome::Error
    });
    progress.metrics = Some(super::progress::Metrics {
        prompt_tokens: records
            .iter()
            .filter_map(|record| record["prompt_tokens"].as_u64())
            .try_fold(0_u64, u64::checked_add),
        completion_tokens: records
            .iter()
            .filter_map(|record| record["completion_tokens"].as_u64())
            .try_fold(0_u64, u64::checked_add),
        cache_pct: evidence["cache_pct"].as_f64(),
        decode_tokens_per_second: evidence["decode_tokens_per_second"].as_f64(),
        workload_output_tokens_per_second: evidence["workload_output_tokens_per_second"].as_f64(),
        ttft_p50_seconds: evidence["ttft_p50_seconds"].as_f64(),
    });
    super::progress::emit(&progress)?;
    finish?;
    Ok(evidence["acceptance"]["passed"] == true)
}

async fn execute(
    workload: &Workload,
    file: &mut std::fs::File,
    cancellation: crate::process::Cancellation,
    progress: &super::progress::Record,
) -> DynResult<Vec<serde_json::Value>> {
    let selection = Selection {
        model: &workload.model,
        maximum_output_tokens: if workload.qualification_probe {
            1
        } else {
            workload.max_output_tokens
        },
        turn_limit: None,
        qualification_probe: workload.qualification_probe,
    };
    let mut sessions =
        super::replay_profile::build(&workload.trajectories, &selection, workload.mode())?;
    if let Some(turns) = workload.warmup_turns {
        let available = sessions.iter().try_fold(0_usize, |total, session| {
            total
                .checked_add(session.len())
                .ok_or("warmup turn count overflow")
        })?;
        if available < turns {
            return Err("warmup cohort has insufficient recorded turns".into());
        }
        let mut remaining = turns;
        for session in &mut sessions {
            session.truncate(remaining);
            remaining = remaining.saturating_sub(session.len());
        }
        sessions.retain(|session| !session.is_empty());
    }
    let mut pending = sessions.into_iter();
    let mut active = tokio::task::JoinSet::new();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(workload.concurrency);
    let epoch = Instant::now();
    let launch = |turns: Vec<Turn>, active: &mut tokio::task::JoinSet<Result<(), String>>| {
        let sender = sender.clone();
        let cancellation = cancellation.clone();
        let base = workload.base_url.clone();
        let timeout = Duration::from_secs(workload.request_timeout_seconds);
        let concurrency = workload.concurrency;
        let warmup = workload.warmup_turns.is_some();
        let progress = progress.clone();
        active.spawn(async move {
            for turn in turns {
                let started = epoch.elapsed();
                let request_started = Instant::now();
                let mut request_progress = progress.clone();
                request_progress.phase = if turn.qualification_probe { super::progress::Phase::Preflight } else { super::progress::Phase::Request };
                request_progress.event = super::progress::Event::Started;
                request_progress.request = Some(super::progress::Request { request_id: turn.request_id.clone(), session_id: turn.session_id.clone(), assistant_turn: turn.assistant_turn });
                let operation = async {
                    tokio::select! {
                        result = tokio::time::timeout(timeout, exchange(&base, &turn)) => match result {
                            Ok(Ok(evidence)) => (super::progress::Outcome::Success, Ok(evidence)),
                            Ok(Err(error)) => (super::progress::Outcome::Error, Err(error.to_string())),
                            Err(error) => (super::progress::Outcome::Timeout, Err(error.to_string())),
                        },
                        () = cancelled(&cancellation) => (super::progress::Outcome::Cancelled, Err("replay interrupted".into())),
                    }
                };
                let (outcome, result) = super::progress::in_flight(operation, request_progress.clone()).await;
                request_progress.event = super::progress::Event::Completed;
                request_progress.elapsed_seconds = request_started.elapsed().as_secs_f64();
                request_progress.outcome = Some(outcome);
                request_progress.metrics = result.as_ref().ok().map(|evidence| super::progress::Metrics {
                    prompt_tokens: Some(evidence.prompt_tokens), completion_tokens: Some(evidence.completion_tokens), ..Default::default()
                });
                let _ = super::progress::emit(&request_progress);
                let mut record = serde_json::to_value(&turn).map_err(|error| error.to_string())?;
                let object = record.as_object_mut().ok_or("invalid turn record")?;
                object.remove("body");
                match result {
                    Ok(evidence) => {
                        let first_token_at = started.as_secs_f64() + evidence.ttft_seconds;
                        let evidence = serde_json::to_value(evidence).map_err(|error| error.to_string())?;
                        object.extend(evidence.as_object().ok_or("invalid stream evidence")?.clone());
                        object.insert("started".into(), started.as_secs_f64().into());
                        object.insert("first_token_at".into(), first_token_at.into());
                    }
                    Err(error) => { object.insert("error".into(), error.into()); }
                }
                record["completed"] = epoch.elapsed().as_secs_f64().into();
                record["concurrency"] = concurrency.into();
                record["warmup"] = warmup.into();
                sender.send(record).await.map_err(|_| "request evidence writer closed")?;
                if cancellation.is_cancelled() { break; }
            }
            Ok(())
        });
    };
    for turns in pending.by_ref().take(workload.concurrency) {
        launch(turns, &mut active);
    }
    let mut records = Vec::new();
    while !active.is_empty() {
        tokio::select! {
            Some(record) = receiver.recv() => preserve(file, record, &mut records)?,
            completed = active.join_next() => {
                completed.ok_or("missing session completion")??.map_err(|error| -> crate::command::DynError { error.into() })?;
                if !cancellation.is_cancelled() && let Some(turns) = pending.next() { launch(turns, &mut active); }
            }
        }
    }
    drop(sender);
    while let Some(record) = receiver.recv().await {
        preserve(file, record, &mut records)?;
    }
    Ok(records)
}

async fn cancelled(cancellation: &crate::process::Cancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn preserve(
    file: &mut std::fs::File,
    record: serde_json::Value,
    records: &mut Vec<serde_json::Value>,
) -> DynResult<()> {
    serde_json::to_writer(&mut *file, &record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    records.push(record);
    Ok(())
}
