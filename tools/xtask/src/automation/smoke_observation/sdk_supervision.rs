//! Closed retained SDK child supervision, not installed-package or model qualification.
use super::super::{command_interrupt::Interrupt, endpoint_model_discovery};
use crate::{command::DynResult, process::Cancellation};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[path = "sdk_supervision/child.rs"]
mod child;
#[cfg(all(test, unix))]
#[path = "sdk_supervision/tests.rs"]
mod tests;
fn pairs(args: &[String]) -> DynResult<std::collections::BTreeMap<&str, &str>> {
    let (pairs, tail) = args.as_chunks::<2>();
    if !tail.is_empty() {
        return Err("SDK owner needs named option pairs".into());
    }
    let mut values = std::collections::BTreeMap::new();
    for [key, value] in pairs {
        if !matches!(
            key.as_str(),
            "--base-url" | "--timeout-secs" | "--python" | "--client" | "--model" | "--receipt"
        ) || values.insert(key.as_str(), value.as_str()).is_some()
        {
            return Err("SDK owner unknown/duplicate option".into());
        }
    }
    Ok(values)
}
fn seconds(values: &std::collections::BTreeMap<&str, &str>) -> DynResult<u64> {
    let n = values
        .get("--timeout-secs")
        .ok_or("SDK owner requires whole budget")?
        .parse()?;
    if !(1..=86400).contains(&n) {
        return Err("SDK whole budget must be 1..86400 seconds".into());
    }
    Ok(n)
}
async fn ready(base: &str, deadline: Instant, cancel: &Cancellation) -> DynResult<String> {
    loop {
        if cancel.is_cancelled() || Instant::now() >= deadline {
            return Err("SDK readiness deadline/cancelled".into());
        }
        let attempt = deadline.min(Instant::now() + Duration::from_secs(5));
        if let Ok(model) = endpoint_model_discovery::discover(base, "", attempt, cancel).await {
            return Ok(model);
        }
        tokio::time::sleep_until(
            (deadline.min(Instant::now() + Duration::from_millis(100))).into(),
        )
        .await;
    }
}
pub(super) fn run(verb: &str, args: &[String]) -> DynResult<()> {
    let values = pairs(args)?;
    let base = endpoint_model_discovery::endpoint(
        values.get("--base-url").ok_or("SDK base URL required")?,
    )?;
    let deadline = Instant::now() + Duration::from_secs(seconds(&values)?);
    if verb == "sdk-ready" {
        if values
            .keys()
            .any(|k| !matches!(*k, "--base-url" | "--timeout-secs"))
        {
            return Err("SDK readiness accepts only endpoint/budget".into());
        }
        let interrupt = Interrupt::install()?;
        let cancel = interrupt.cancellation();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let result = runtime.block_on(ready(&base, deadline, &cancel));
        interrupt.finish()?;
        if cancel.is_cancelled() || Instant::now() >= deadline {
            return Err("SDK readiness terminal refusal".into());
        }
        use std::io::Write as _;
        crate::cli_output::stdout().write_all(format!("{}\n", result?).as_bytes())?;
        return Ok(());
    }
    let client = *values.get("--client").ok_or("SDK client required")?;
    let python = PathBuf::from(
        values
            .get("--python")
            .copied()
            .ok_or("SDK interpreter required")?,
    );
    let receipt = PathBuf::from(
        values
            .get("--receipt")
            .copied()
            .ok_or("SDK receipt required")?,
    );
    let model = values.get("--model").copied();
    let root = crate::repo_consistency::repo_root()?.canonicalize()?;
    let source = super::super::python_sdk_source::admit(&root)?;
    let mut spec = child::admit(&source.root, client, &python, &base, model)?;
    spec.pins.extend(source.pins);
    child::run(spec, &receipt, deadline)
}
