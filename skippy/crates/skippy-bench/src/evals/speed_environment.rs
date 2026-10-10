//! Explicit pinned SPEED SDK preparation; execution performs no installation or dataset acquisition.
use super::*;
use crate::cli::EvalPrepareMcpArgs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[path = "speed_environment/preparation.rs"]
mod preparation;
#[cfg(test)]
#[path = "speed_environment/tests.rs"]
mod tests;

const SOURCE: &str = "ff6779035febf91eed41cdd16757873e6a81d446";
const PROJECT_SHA: &str = "15c7f2a51ecd86c11c456a1954fa59c66acf96ffc006ae8ae348aaa9c72b76ec";
const LOCK_SHA: &str = "008fad1e5ea48efde545ef3b05fa9c45700b32d0fc7db1584c9f3487f82634b2";
const SCRIPT_SHA: &str = "9eacf67452c2b9a829da61c8b077bc5ece9219482e1e2734ad4a9340bdb03911";
const REQUIREMENTS_SHA: &str = "af61baf90f1ee3421845076ff594159b3656062132ed1522470154bccaf7e635";
const DATASET_REVISION: &str = "454f88454792dfa3ccfd7ef15fff248efde44cd1";
const DATASET_SHA: &str = "4f76bc45bdffb38712a5c26d7a7c9a9a791484b392a7b64a6ffb9abe48e02e26";
const DATASET_BYTES: u64 = 364138;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u8,
    source: String,
    dataset_revision: String,
    dataset_sha256: String,
    python: PathBuf,
    tool_pins: BTreeMap<PathBuf, String>,
    environment_pins: BTreeMap<PathBuf, String>,
    benchmark_qualified: bool,
}
pub(super) fn base(root: &Path) -> PathBuf {
    root.join("speed-sdk-v1")
}
pub(super) fn runtime_python(root: &Path) -> PathBuf {
    base(root).join("environment/bin/python")
}
pub(super) fn dataset(root: &Path) -> PathBuf {
    base(root).join("dataset.parquet")
}
fn validate_receipt(receipt: &Receipt) -> Result<()> {
    if receipt.schema_version != 1
        || receipt.source != SOURCE
        || receipt.dataset_revision != DATASET_REVISION
        || receipt.dataset_sha256 != DATASET_SHA
        || receipt.benchmark_qualified
    {
        bail!("SPEED prepared receipt contract refused");
    }
    Ok(())
}
fn validate_dataset(bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 != DATASET_BYTES || sdk_environment::hash_bytes(bytes) != DATASET_SHA {
        bail!("SPEED pinned dataset changed");
    }
    Ok(())
}
fn validate_project(root: &Path) -> Result<()> {
    for (name, expected) in [("pyproject.toml", PROJECT_SHA), ("uv.lock", LOCK_SHA)] {
        let actual = sdk_environment::read(&base(root).join("project").join(name), 1048576)?;
        if sdk_environment::hash_bytes(&actual) != expected {
            bail!("SPEED locked preparation input changed");
        }
    }
    Ok(())
}
fn validate_source(root: &Path) -> Result<()> {
    harness_source::admit_run(root, registry::definition(EvalId::SpeedBench))?;
    let source = harness_dir(root, registry::definition(EvalId::SpeedBench))
        .join("tools/server/bench/speed-bench");
    for (name, expected) in [
        ("speed_bench.py", SCRIPT_SHA),
        ("requirements.txt", REQUIREMENTS_SHA),
    ] {
        if sdk_environment::hash_bytes(&sdk_environment::read(&source.join(name), 1048576)?)
            != expected
        {
            bail!("SPEED upstream SDK source changed");
        }
    }
    Ok(())
}
pub(super) fn admit(root: &Path) -> Result<()> {
    if !cfg!(unix) {
        bail!("SPEED prepared SDK currently supports Unix hosts only");
    }
    let receipt: Receipt = serde_json::from_slice(
        &sdk_environment::read(&base(root).join("receipt.json"), 8 * 1048576)
            .context("SPEED SDK unprepared; run eval prepare-speed explicitly")?,
    )?;
    validate_receipt(&receipt)?;
    validate_source(root)?;
    validate_project(root)?;
    validate_dataset(&sdk_environment::read(&dataset(root), DATASET_BYTES)?)?;
    for (path, expected) in &receipt.tool_pins {
        if sdk_environment::hash(path)? != *expected {
            bail!("SPEED preparation tool changed");
        }
    }
    if receipt.tool_pins.len() != 2 || !receipt.tool_pins.contains_key(&receipt.python) {
        bail!("SPEED preparation tool roster differs");
    }
    if sdk_environment::environment(
        &base(root).join("environment"),
        &receipt.python,
        sdk_environment::PythonProfile::Mcp312,
    )? != receipt.environment_pins
    {
        bail!("SPEED prepared environment changed");
    }
    external_sdk_source::admit_run(EvalId::SpeedBench)
}
pub(super) fn prepare(args: EvalPrepareMcpArgs) -> Result<()> {
    preparation::prepare(args)
}
pub(super) fn isolated(spec: CommandSpec, root: &Path) -> CommandSpec {
    spec.isolated()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", base(root).join("home").display().to_string())
        .env("TMPDIR", base(root).join("tmp").display().to_string())
        .env(
            "UV_CACHE_DIR",
            base(root).join("cache").display().to_string(),
        )
        .env(
            "UV_PROJECT_ENVIRONMENT",
            base(root).join("environment").display().to_string(),
        )
        .env("UV_PYTHON_DOWNLOADS", "never")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
}
