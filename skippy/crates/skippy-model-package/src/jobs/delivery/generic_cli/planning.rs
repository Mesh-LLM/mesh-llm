//! Offline preparation using the existing CPU hardware planner and its typed receipt digest.
use super::{Cli, read, root, write};
use crate::jobs::{CpuJobPlan, HardwareFlavor, plan_cpu_job_from_hardware};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::time::Instant;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuHardwareRequest {
    schema_version: u32,
    hardware: Vec<HardwareFlavor>,
    requested_flavor: String,
    requested_timeout_seconds: u64,
    model_size_bytes: u64,
    max_cost_usd: f64,
}
#[derive(Serialize)]
struct BootstrapResources {
    timeout_seconds: u64,
    cpu_plan_receipt_sha256: String,
    declared_estimate_usd: f64,
    max_cost_usd: f64,
}
fn text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}
fn plan(request: &CpuHardwareRequest) -> Result<CpuJobPlan> {
    if request.schema_version != 1
        || !(1..=256).contains(&request.hardware.len())
        || !text(&request.requested_flavor)
        || !(30..=259200).contains(&request.requested_timeout_seconds)
        || !(1..=16 * 1024_u64.pow(4)).contains(&request.model_size_bytes)
        || !request.max_cost_usd.is_finite()
        || request.max_cost_usd <= 0.0
    {
        bail!("offline planner request bounds refused");
    }
    for (index, hardware) in request.hardware.iter().enumerate() {
        if !text(&hardware.name)
            || [
                &hardware.pretty_name,
                &hardware.cpu,
                &hardware.ram,
                &hardware.unit_label,
            ]
            .into_iter()
            .flatten()
            .any(|value| !text(value))
            || request.hardware[..index]
                .iter()
                .any(|p| p.name == hardware.name)
        {
            bail!("offline hardware identity refused");
        }
        if hardware.accelerator.is_none() {
            let cost = hardware.resolved_unit_cost_usd()?;
            if !cost.is_finite() || cost < 0.0 {
                bail!("offline CPU price refused");
            }
        }
    }
    let plan = plan_cpu_job_from_hardware(
        &request.hardware,
        &request.requested_flavor,
        request.requested_timeout_seconds,
        request.model_size_bytes,
    )?;
    if plan.timeout_seconds > 259200
        || !plan.max_cost_usd.is_finite()
        || plan.max_cost_usd > request.max_cost_usd
    {
        bail!("offline planner budget exceeds declared maximum");
    }
    Ok(plan)
}
pub(super) fn run(cli: &Cli, until: Instant) -> Result<bool> {
    if cli.credential_file.is_some() || cli.submitted_file.is_some() || cli.confirm_submission {
        bail!("offline plan cannot consume credentials or submission authority");
    }
    let path = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("planner input absent"))?;
    let bytes = read(path, 1048576)?;
    let input: CpuHardwareRequest = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("offline planner typed request refused"))?;
    let plan = plan(&input)?;
    let resources = BootstrapResources {
        timeout_seconds: plan.timeout_seconds,
        cpu_plan_receipt_sha256: super::super::admission::digest(&serde_json::to_vec(&plan)?),
        declared_estimate_usd: plan.max_cost_usd,
        max_cost_usd: input.max_cost_usd,
    };
    if Instant::now() >= until {
        bail!("offline planner deadline expired before output");
    }
    let output = cli
        .output_directory
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("planner output absent"))?;
    let root = root(output)?;
    write(&root, "cpu-plan.json", &plan)?;
    write(&root, "bootstrap-resources.json", &resources)?;
    write(
        &root,
        "result.json",
        &serde_json::json!({"schema_version":1,"status":"PLANNED_OFFLINE","input_sha256":super::super::admission::digest(&bytes),"submitted":false,"hardware_observed":false,"cost_observed":false}),
    )?;
    Ok(Instant::now() < until)
}
#[cfg(test)]
#[path = "planning/tests.rs"]
mod tests;
