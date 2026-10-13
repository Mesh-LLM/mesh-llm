//! Raw inventory command owner. Public xtask release routing and skill cutover are a separate reviewed layer.
mod args;
mod dirty;
mod dirty_file;
mod evidence;
mod github;
mod observation;
mod provenance;
mod publication;
mod remotes;
mod report;
mod transport;
use crate::automation::command_interrupt::Interrupt;
use provenance::{Error, Result};
use std::{
    io::Write,
    path::Path,
    time::{Duration, SystemTime},
};
pub(crate) fn run(arguments: &[String]) -> Result<()> {
    let Some(args) = args::parse(arguments)? else {
        println!("{}", args::USAGE);
        return Ok(());
    };
    let scope = Interrupt::install().map_err(|e| Error(e.to_string()))?;
    let observation =
        observation::Observation::new(scope.cancellation(), Duration::from_secs(900))?;
    let environment: std::collections::BTreeMap<_, _> = std::env::vars_os().collect();
    let cwd = std::env::current_dir()
        .map_err(|_| Error("release inventory current directory unavailable".into()))?;
    let mut tools = transport::Transport::new_until(
        &cwd,
        environment.clone(),
        observation.cancellation.clone(),
        observation.deadline,
    )?;
    let root = report::repository_root(&mut tools, &observation)?;
    let mut tools = transport::Transport::new_until(
        &root,
        environment,
        observation.cancellation.clone(),
        observation.deadline,
    )?;
    let collected_at = timestamp()?;
    let report = report::collect(
        &mut tools,
        &root,
        &args.repository,
        &args.head,
        args.release_tag.as_deref(),
        &collected_at,
        &observation,
    )?;
    let bytes = report.bytes(&observation)?;
    match args.output.as_deref() {
        Some(path) => publish(
            path,
            &bytes,
            &report,
            &mut tools,
            &root,
            &observation,
            scope,
        ),
        None => {
            report.revalidate(&mut tools, &root, &observation, None)?;
            observation.check()?;
            scope.finish().map_err(|e| Error(e.to_string()))?;
            observation.check()?;
            std::io::stdout()
                .lock()
                .write_all(&bytes)
                .map_err(|_| Error("release inventory stdout write failed".into()))
        }
    }
}
fn publish(
    path: &Path,
    bytes: &[u8],
    report: &report::Report,
    tools: &mut transport::Transport,
    root: &Path,
    observation: &observation::Observation,
    scope: Interrupt,
) -> Result<()> {
    let prepared = publication::Prepared::stage(path, bytes, observation)?;
    let mut scope = Some(scope);
    prepared.publish(observation, |phase, prepared| match phase {
        publication::Phase::BeforeReplacement => {
            report.revalidate(tools, root, observation, Some(prepared))
        }
        publication::Phase::AfterReplacement => {
            provenance::revalidate(tools, &report.source)?;
            observation.check()?;
            scope
                .take()
                .ok_or_else(|| Error("release inventory signal owner already finished".into()))?
                .finish()
                .map_err(|e| Error(e.to_string()))?;
            observation.check()
        }
    })
}
fn timestamp() -> Result<String> {
    let instant =
        crate::ci_operations::ci_metrics_time::Instant::from_system_time(SystemTime::now())
            .ok_or_else(|| Error("release inventory clock out of report range".into()))?;
    Ok(crate::ci_operations::ci_metrics_time::isoformat(instant))
}
