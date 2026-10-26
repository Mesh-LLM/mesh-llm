use crate::command::DynResult;
use std::time::{Duration, Instant};

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    let [startup, console_port, workload, requests, summary] = args else {
        return Err("invalid server cell worker arguments".into());
    };
    let startup: u64 = startup.parse()?;
    let console_port: u16 = console_port.parse()?;
    if !(1..=86400).contains(&startup) {
        return Err("invalid startup deadline".into());
    }
    let mut workload: super::cell_execution::Workload =
        serde_json::from_slice(&std::fs::read(workload)?)?;
    let following = std::mem::take(&mut workload.following_cells);
    for cell in &following {
        if cell.workload.base_url != workload.base_url || !cell.workload.following_cells.is_empty()
        {
            return Err("follow-on cells must use the owned endpoint and cannot nest".into());
        }
        if !cell.requests_output.is_absolute() || !cell.summary_output.is_absolute() {
            return Err("follow-on cell outputs must be absolute".into());
        }
    }
    let base = &workload.base_url;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let deadline = Instant::now() + Duration::from_secs(startup);
    let model = loop {
        if Instant::now() >= deadline {
            return Err("replay model startup deadline exceeded".into());
        }
        let budget = Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now()));
        if let Ok(bytes) = runtime.block_on(async {
            tokio::time::timeout(
                budget,
                get(&format!("{}/models", base.trim_end_matches('/'))),
            )
            .await?
        }) && let Ok(list) = serde_json::from_slice::<Models>(&bytes)
            && let [model] = list.data.as_slice()
            && !model.id.is_empty()
        {
            break model.id.clone();
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut effective_context = None;
    if let Some(context) = &workload.runtime_context {
        if context.required_tokens == 0 || !context.output.is_absolute() {
            return Err(
                "runtime context requires a positive token budget and absolute output".into(),
            );
        }
        let url = format!("http://127.0.0.1:{console_port}/api/runtime");
        loop {
            if Instant::now() >= deadline {
                return Err("runtime context startup deadline exceeded".into());
            }
            let budget =
                Duration::from_secs(2).min(deadline.saturating_duration_since(Instant::now()));
            if let Ok(bytes) =
                runtime.block_on(async { tokio::time::timeout(budget, get(&url)).await? })
                && let Ok(document) =
                    serde_json::from_slice::<super::session_evidence::Runtime>(&bytes)
                && document.models.len() == 1
            {
                std::fs::write(&context.output, &bytes)?;
                effective_context = Some(document.context(context.required_tokens)?);
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    bind_context(&mut workload, effective_context)?;
    workload.model.clone_from(&model);
    let initial_passed = super::cell_execution::measure(
        &workload,
        std::path::Path::new(requests),
        std::path::Path::new(summary),
    )?;
    if !initial_passed && !workload.qualification_probe {
        return Err("initial replay cell failed; report retained".into());
    }
    let mut passed = initial_passed;
    for mut cell in following {
        bind_context(&mut cell.workload, effective_context)?;
        cell.workload.model.clone_from(&model);
        passed &= super::cell_execution::measure(
            &cell.workload,
            &cell.requests_output,
            &cell.summary_output,
        )?;
    }
    if passed {
        Ok(())
    } else {
        Err("measured cells failed; all cell evidence retained".into())
    }
}

fn bind_context(
    workload: &mut super::cell_execution::Workload,
    context: Option<u64>,
) -> DynResult<()> {
    if let Some(budget) = &mut workload.eligibility {
        budget.context_tokens = context.ok_or("cell eligibility requires owned runtime context")?;
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct Models {
    data: Vec<Model>,
}
#[derive(serde::Deserialize)]
struct Model {
    id: String,
}

pub(super) async fn get(url: &str) -> DynResult<Vec<u8>> {
    crate::automation::openai_exchange::get(url).await
}

pub(super) fn limits(execution: Duration) -> crate::process::Limits {
    crate::process::Limits {
        execution,
        graceful_shutdown: Duration::from_secs(30),
        forced_shutdown: Duration::from_secs(10),
        retained_bytes_per_stream: 65536,
        readiness: crate::process::Readiness::None,
        completion: crate::process::Completion::Exit,
    }
}
