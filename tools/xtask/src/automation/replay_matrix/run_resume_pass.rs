use crate::command::DynResult;
use std::{collections::BTreeSet, path::Path};

pub(super) fn verify(
    previous: &serde_json::Value,
    current: &serde_json::Value,
    paths: (&Path, &Path),
    passes: u32,
) -> DynResult<BTreeSet<(u32, String)>> {
    let builds = current["builds"]
        .as_array()
        .ok_or("invalid build identities")?;
    let levels = current["config"]["concurrency"]
        .as_array()
        .ok_or("invalid concurrency")?;
    let results = previous["results"]
        .as_array()
        .ok_or("missing previous results")?;
    let order = results
        .iter()
        .map(|result| serde_json::json!({"pass":result["pass"],"label":result["label"]}))
        .collect::<Vec<_>>();
    if previous["order"] != serde_json::Value::Array(order) {
        return Err("cannot resume: retained arm order differs from results".into());
    }
    let mode = super::replay_profile::Mode::parse(
        current["config"]["replay_mode"].as_str().unwrap_or("all"),
    )?;
    let mut completed = BTreeSet::new();
    for result in results {
        verify_result(result, builds, levels, paths.0, passes, &mut completed)?;
        let pass = result["pass"].as_u64().ok_or("missing resumed pass")?;
        let label = result["label"].as_str().ok_or("missing resumed label")?;
        let directory = paths
            .0
            .parent()
            .ok_or("missing resume artifact parent")?
            .join(format!("data/pass-{pass}/{label}"));
        super::resume_profile::verify(
            &directory,
            paths.1,
            result["cells"].as_array().ok_or("missing resumed cells")?,
            mode,
        )?;
    }
    Ok(completed)
}

fn verify_result(
    result: &serde_json::Value,
    builds: &[serde_json::Value],
    levels: &[serde_json::Value],
    run_path: &Path,
    passes: u32,
    completed: &mut BTreeSet<(u32, String)>,
) -> DynResult<()> {
    let label = result["label"].as_str().ok_or("missing prior arm label")?;
    let build = builds
        .iter()
        .find(|build| build["label"].as_str() == Some(label))
        .ok_or("foreign prior arm")?;
    if result["ref"] != build["ref"] || result["commit"] != build["commit"] {
        return Err("cannot resume: retained arm provenance differs".into());
    }
    let pass = u32::try_from(result["pass"].as_u64().ok_or("missing prior pass")?)?;
    let cells = result["cells"].as_array().ok_or("missing prior cells")?;
    let expected = levels
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect::<BTreeSet<_>>();
    let observed = cells
        .iter()
        .filter_map(|cell| cell["concurrency"].as_u64())
        .collect::<BTreeSet<_>>();
    if !(1..=passes).contains(&pass)
        || !completed.insert((pass, label.into()))
        || cells.len() != levels.len()
        || observed != expected
        || cells
            .iter()
            .any(|cell| !cell["acceptance"]["passed"].is_boolean())
    {
        return Err("cannot resume: incomplete, duplicate or foreign prior pass".into());
    }
    let directory = run_path
        .parent()
        .ok_or("missing run directory")?
        .join(format!("data/pass-{pass}/{label}"));
    super::pass_identity::verify(&directory, &result["artifact_sha256"])?;
    for cell in cells {
        let level = cell["concurrency"]
            .as_u64()
            .ok_or("missing retained concurrency")?;
        let retained: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join(format!("c-{level}.json")))?)?;
        if retained != *cell {
            return Err("cannot resume: run result differs from retained cell summary".into());
        }
    }
    if result["passed"] == true && !super::pass_lifecycle::acceptance_exit(&directory)? {
        return Err("cannot resume: clean lifecycle receipt is missing".into());
    }
    if result["passed"] != true && result["acceptance_failed"] != true {
        return Err("cannot resume: retained pass failed infrastructure".into());
    }
    if build["external_engine"].is_object() {
        verify_external(result, build, &directory)?;
    }
    Ok(())
}

fn verify_external(
    result: &serde_json::Value,
    build: &serde_json::Value,
    directory: &Path,
) -> DynResult<()> {
    for field in ["engine", "version", "version_sha256", "provenance"] {
        if result[field] != build[field] {
            return Err("cannot resume: external pass identity differs".into());
        }
    }
    let command: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("server.command.json"))?)?;
    if result["command"] != command {
        return Err("cannot resume: external command evidence differs".into());
    }
    let workload: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("workload.json"))?)?;
    let endpoint: hyper::Uri = workload["base_url"]
        .as_str()
        .ok_or("missing retained external endpoint")?
        .parse()?;
    let verified: super::external_probe::Verified = serde_json::from_value(build.clone())?;
    let expected = super::external_command::server(
        &verified.external_engine,
        &verified.provenance.resolved_executable,
        endpoint
            .port_u16()
            .ok_or("missing retained external port")?,
    )?;
    if command != serde_json::to_value(expected)? {
        return Err("cannot resume: external command intent differs".into());
    }
    Ok(())
}
