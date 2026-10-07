//! Live Decisions HTTP qualification for an already running System One model.
#[path = "decisions_smoke/options.rs"]
mod options;
#[path = "decisions_smoke/request.rs"]
mod request;
#[path = "decisions_smoke/response.rs"]
mod response;
use super::guardrail_corpus::transport;
use crate::{command::DynResult, command_interrupt::Interrupt, process::Cancellation};
use std::time::{Duration, Instant};

async fn cancelled(token: &Cancellation) {
    while !token.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn exchange(
    base: &str,
    suffix: &str,
    body: Option<&serde_json::Value>,
    deadline: Instant,
    token: &Cancellation,
) -> DynResult<serde_json::Value> {
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("Decisions smoke cancelled or deadline reached".into());
    }
    let url = url::Url::parse(base)?;
    let curl = url.scheme() == "https" || !matches!(url.host(), Some(url::Host::Ipv4(_)));
    let future = transport::exchange_owned_with_private_authorization(
        base, suffix, body, deadline, token, None,
    );
    let response = if curl {
        // The supervised curl owner includes cleanup in this budget. Do not abandon its worker.
        future
            .await
            .map_err(|_| "Decisions smoke bounded transfer refused")?
    } else {
        tokio::select! {
            biased;
            _ = cancelled(token) => return Err("Decisions smoke cancelled".into()),
            result = tokio::time::timeout_at(deadline.into(), future) =>
                result.map_err(|_| "Decisions smoke request deadline")?
                    .map_err(|_| "Decisions smoke HTTP transfer refused")?,
        }
    };
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("Decisions smoke cancelled or deadline reached".into());
    }
    if response.status != 200 {
        return Err(format!("Decisions smoke requires HTTP 200, got {}", response.status).into());
    }
    serde_json::from_slice(&response.body)
        .map_err(|_| "Decisions smoke response is not JSON".into())
}
async fn smoke(
    options: &options::Options,
    deadline: Instant,
    token: &Cancellation,
) -> DynResult<String> {
    let models = exchange(&options.base, "v1/models", None, deadline, token).await?;
    let model = response::select_model(&models, options.model.as_deref())?;
    let body = request::questions(&model);
    let answers = exchange(&options.base, "v1/decisions", Some(&body), deadline, token).await?;
    response::validate(&answers, &model)?;
    Ok(model)
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{}", options::USAGE);
        return Ok(());
    }
    let options = options::Options::parse(args)?;
    let interrupt = Interrupt::install()?;
    let token = interrupt.cancellation();
    let deadline = Instant::now() + options.timeout;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(smoke(&options, deadline, &token));
    interrupt.finish()?;
    if token.is_cancelled() || Instant::now() >= deadline {
        return Err("Decisions smoke terminal cancellation/deadline".into());
    }
    println!(
        "Decisions live smoke passed: model={}, questions=predicate,choice,score",
        result?
    );
    Ok(())
}
