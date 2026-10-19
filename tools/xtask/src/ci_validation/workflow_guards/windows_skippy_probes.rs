//! The Windows Skippy probe row runs every `^skippy_` probe under the image's
//! 64-bit ctest (msvc-dev-cmd's 32-bit one dies on the graph-contract output),
//! logs the runner's commit headroom first, and gets the row's time budget.
use super::{Node, field, workflow_yaml};
use crate::command::DynResult;
use serde_json::Value;
use std::{fs, path::Path};
const WORKFLOW: &str = ".github/workflows/ci-platform-checks-slice.yml";
const CATALOG: &str = "ci/slices.yml";
const ROW: &str = "windows-skippy-probes";
const PROBE_KIND: &str = "matrix.check.kind == 'skippy-probes'";
const JOB_TIMEOUT: &str = "${{ matrix.check.timeout_minutes || 75 }}";
const DEFAULT_TIMEOUT: u64 = 75;
const CTEST_X64: &str = r"C:\Program Files\CMake\bin\ctest.exe";
const BUILD: &str = r#"PATH="/c/Program Files/CMake/bin:$PATH" scripts/build-llama.sh"#;
const PARALLEL: &str = "${{ matrix.check.ctest_parallel }}";

pub(super) fn check(root: &Path) -> DynResult<()> {
    let workflow = workflow_yaml::parse(&fs::read_to_string(root.join(WORKFLOW))?)?;
    probe_job(
        workflow
            .get("jobs")
            .and_then(|jobs| jobs.get("platform_checks"))
            .ok_or("platform checks job missing")?,
    )?;
    probe_row(&serde_json::from_str(&fs::read_to_string(
        root.join(CATALOG),
    )?)?)
}
fn probe_job(job: &Node) -> DynResult<()> {
    if field(job, "timeout-minutes") != Some(JOB_TIMEOUT) {
        return Err("platform checks must take their timeout from the matrix row".into());
    }
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("platform checks job has no steps".into());
    };
    let probes: Vec<_> = steps
        .iter()
        .filter(|step| field(step, "if").is_some_and(|condition| condition.contains(PROBE_KIND)))
        .collect();
    let find = |needles: &[&str]| {
        probes
            .iter()
            .position(|step| needles.iter().all(|needle| run(step).contains(needle)))
    };
    let check = find(&["0x8664", CTEST_X64]).ok_or("Skippy probes must check the 64-bit ctest")?;
    let headroom = find(&["FreeVirtualMemory", "TotalVirtualMemorySize"])
        .ok_or("Skippy probes must log the runner's commit headroom")?;
    let build = find(&["scripts/build-llama.sh"]).ok_or("Skippy probes have no build step")?;
    if check > build || headroom > build {
        return Err("the ctest check and headroom log must precede the Skippy probes".into());
    }
    let build = probes[build];
    if !run(build).contains(BUILD) {
        return Err("Skippy probes must put the 64-bit CMake first on PATH".into());
    }
    let env = |name| build.get("env").and_then(|env| field(env, name));
    if env("LLAMA_STAGE_BUILD_TESTS") != Some("ON") {
        return Err("Skippy probes must build the probe tests".into());
    }
    if env("CTEST_PARALLEL_LEVEL") != Some(PARALLEL) {
        return Err("Skippy probe parallelism must come from the matrix row".into());
    }
    Ok(())
}
fn run(step: &Node) -> &str {
    field(step, "run").unwrap_or_default()
}
fn probe_row(catalog: &Value) -> DynResult<()> {
    let row = catalog["platform_rows"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|row| row["id"] == ROW)
        .ok_or("windows-skippy-probes row missing")?;
    if row["platform"] != "windows" || row["kind"] != "skippy-probes" {
        return Err("windows-skippy-probes must be a Windows skippy-probes row".into());
    }
    if !row["ctest_parallel"]
        .as_u64()
        .is_some_and(|level| level >= 1)
    {
        return Err("windows-skippy-probes needs a ctest_parallel of at least 1".into());
    }
    if !row["timeout_minutes"]
        .as_u64()
        .is_some_and(|minutes| minutes > DEFAULT_TIMEOUT)
    {
        return Err("windows-skippy-probes must extend the default job timeout".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "windows_skippy_probes_tests.rs"]
mod tests;
