use super::{
    l3_contract::{Config, Phase, Request},
    l3_execution::Mode,
    recorded_requests::{Selection, Trajectory, Turn},
};
use crate::command::DynResult;
use std::{
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

fn turns(
    trajectories: &[Trajectory],
    config: &Config,
    serving_model: &str,
    mode: Mode,
    name: &str,
) -> DynResult<Vec<Turn>> {
    let selection = Selection {
        model: serving_model,
        maximum_output_tokens: config.max_output_tokens,
        turn_limit: None,
        qualification_probe: false,
    };
    let mut selected = Vec::new();
    match mode {
        Mode::Identical(repeats) => {
            let trajectory = trajectories.first().ok_or("empty lifecycle cohort")?;
            for index in 0..repeats {
                let mut turn = super::recorded_requests::build(trajectory, &selection)?
                    .pop()
                    .ok_or("missing final checkpoint")?;
                turn.request_id = format!("{}:{name}:{index}", turn.session_id);
                selected.push(turn);
            }
        }
        _ => {
            let cohort = match mode {
                Mode::Additional => trajectories.get(1..).ok_or("missing additional sources")?,
                Mode::HighLoad(c) => trajectories
                    .get(..c)
                    .ok_or("insufficient high-load cohort")?,
                _ => trajectories,
            };
            for trajectory in cohort {
                let mut turns = super::recorded_requests::build(trajectory, &selection)?;
                if matches!(mode, Mode::Growth) {
                    selected.extend(turns);
                } else {
                    selected.push(turns.pop().ok_or("missing final checkpoint")?);
                }
            }
        }
    }
    if selected.is_empty() || selected.len() > 1_000_000 {
        return Err("empty or oversized L3 request phase".into());
    }
    for turn in &mut selected {
        if let Some(sample) = name.strip_prefix("disk_off_cold_") {
            turn.request_id = format!("{}:disk-off:{sample}", turn.session_id);
        }
        if let Some(sample) = name.strip_prefix("restart_l3_") {
            turn.request_id = format!("{}:restart:{sample}", turn.session_id);
        }
    }
    Ok(selected)
}
async fn request(
    base: &str,
    turn: Turn,
    timeout: Duration,
    epoch: Instant,
    cancellation: Option<crate::process::Cancellation>,
) -> Result<serde_json::Value, String> {
    let started = epoch.elapsed().as_secs_f64();
    let mut progress = super::progress::Record::boundary(
        super::progress::Phase::Request,
        super::progress::Event::Started,
        "disk-L3".into(),
    );
    progress.request = Some(super::progress::Request {
        request_id: turn.request_id.clone(),
        session_id: turn.session_id.clone(),
        assistant_turn: turn.assistant_turn,
    });
    let result = super::progress::in_flight(
        async {
            let cancelled=async { if let Some(cancellation)=&cancellation { while !cancellation.is_cancelled() { tokio::time::sleep(Duration::from_millis(10)).await; } } else { std::future::pending::<()>().await; } };
            tokio::select! { result=tokio::time::timeout(timeout,super::trajectory_execution::exchange(base,&turn))=>match result {
                Ok(Ok(value)) => (super::progress::Outcome::Success, Ok(value)),
                Ok(Err(error)) => (super::progress::Outcome::Error, Err(error.to_string())),
                Err(error) => (super::progress::Outcome::Timeout, Err(error.to_string())),
            }, ()=cancelled=>(super::progress::Outcome::Cancelled,Err("disk-L3 request interrupted".into())) }
        },
        progress.clone(),
    )
    .await;
    progress.event = super::progress::Event::Completed;
    progress.elapsed_seconds = epoch.elapsed().as_secs_f64() - started;
    progress.outcome = Some(result.0);
    progress.metrics = result.1.as_ref().ok().map(|e| super::progress::Metrics {
        prompt_tokens: Some(e.prompt_tokens),
        completion_tokens: Some(e.completion_tokens),
        ..Default::default()
    });
    let _ = super::progress::emit(&progress);
    let mut record = serde_json::to_value(turn).map_err(|e| e.to_string())?;
    let object = record.as_object_mut().ok_or("invalid L3 request")?;
    object.remove("body");
    match result.1 {
        Ok(evidence) => {
            let value = serde_json::to_value(evidence).map_err(|e| e.to_string())?;
            object.extend(value.as_object().ok_or("invalid stream evidence")?.clone());
        }
        Err(error) => {
            object.insert("error".into(), error.into());
        }
    }
    object.insert("started".into(), started.into());
    object.insert("completed".into(), epoch.elapsed().as_secs_f64().into());
    Ok(record)
}
pub(super) struct Endpoint<'a> {
    pub base: &'a str,
    pub model: &'a str,
    pub timeout: Duration,
    pub cancellation: Option<crate::process::Cancellation>,
}
pub(super) async fn measure(
    endpoint: &Endpoint<'_>,
    trajectories: &[Trajectory],
    config: &Config,
    mode: Mode,
    name: &str,
    raw: &Path,
) -> DynResult<Phase> {
    let base = endpoint.base;
    let timeout = endpoint.timeout;
    if timeout.is_zero() {
        return Err("L3 request timeout must be positive".into());
    }
    if endpoint.model.is_empty() {
        return Err("missing admitted serving model identity".into());
    }
    let turns = turns(trajectories, config, endpoint.model, mode, name)?;
    let epoch = Instant::now();
    let mut records = Vec::new();
    let mut file = std::fs::File::create(raw)?;
    if matches!(mode, Mode::Growth | Mode::Final | Mode::Additional) {
        for turn in turns {
            let record = request(base, turn, timeout, epoch, endpoint.cancellation.clone()).await?;
            serde_json::to_writer(&mut file, &record)?;
            file.write_all(b"\n")?;
            file.flush()?;
            records.push(record);
            if endpoint
                .cancellation
                .as_ref()
                .is_some_and(crate::process::Cancellation::is_cancelled)
            {
                break;
            }
        }
    } else {
        let barrier = Arc::new(tokio::sync::Barrier::new(turns.len()));
        let mut active = tokio::task::JoinSet::new();
        for turn in turns {
            let base = base.to_owned();
            let barrier = Arc::clone(&barrier);
            let cancellation = endpoint.cancellation.clone();
            active.spawn(async move {
                barrier.wait().await;
                request(&base, turn, timeout, epoch, cancellation).await
            });
        }
        while let Some(record) = active.join_next().await {
            let record = record??;
            serde_json::to_writer(&mut file, &record)?;
            file.write_all(b"\n")?;
            file.flush()?;
            records.push(record);
        }
    }
    let summary = if let Mode::HighLoad(concurrency) = mode {
        Some(serde_json::from_value(super::cell_summary::summarize(
            &trajectories[..concurrency],
            &records,
            concurrency,
        )?)?)
    } else {
        None
    };
    let requests = records
        .into_iter()
        .map(serde_json::from_value::<Request>)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Phase {
        requests,
        summary,
        ..Default::default()
    })
}
