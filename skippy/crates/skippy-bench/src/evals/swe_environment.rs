//! Opt-in locked SWE SDK preparation; runtime never resolves or mutates dependencies.
use super::*;
use crate::cli::{EvalPrepareSweArgs, SweDeployment};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[path = "swe_environment/custody.rs"]
mod custody;
#[path = "swe_environment/preparation.rs"]
mod preparation;
#[cfg(test)]
#[path = "swe_environment/tests.rs"]
mod tests;
const PROJECT_SHA: &str = "6d6600df827e367958a35ee56c8813d8c247b37e9dc2db655e4ce7ad2e855b61";

const LOCK_SHA: &str = "85aceb40793727186c4af3a36a614026af56e906f3e36948869b5aa39a92a7a8";
const PYTHON_VERSION: &str = "Python 3.11.13";
const PREPARATION_SECONDS: u64 = 600;
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    deployment: SweDeployment,
    index_url: String,
}
impl Configuration {
    pub(super) fn runtime() -> Result<Self> {
        if env::var("SWE_BENCH_PRO_PYTHON").is_ok_and(|v| v != "3.11.13") {
            bail!(
                "prepared SWE profile requires explicit CPython3.11.13; conflicting runtime selector refused"
            );
        }
        if env::var("SWE_BENCH_PRO_SWEREX_SPEC")
            .is_ok_and(|v| !v.is_empty() && v != "swe-rex[modal]==1.4.0")
        {
            bail!("prepared SWE profile requires retained swe-rex[modal]==1.4.0");
        }
        let deployment = match env::var("SWE_BENCH_PRO_DEPLOYMENT_TYPE")
            .as_deref()
            .unwrap_or("docker")
        {
            "docker" => SweDeployment::Docker,
            "modal" => SweDeployment::Modal,
            _ => bail!("SWE deployment must be docker or modal"),
        };
        Self::new(
            deployment,
            env::var("SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL")
                .unwrap_or_else(|_| "https://pypi.org/simple".into()),
        )
    }
    fn new(deployment: SweDeployment, index_url: String) -> Result<Self> {
        if index_url.is_empty()
            || index_url.len() > 2048
            || index_url
                .chars()
                .any(|c| c.is_control() || matches!(c, '\'' | '"' | '\\' | '{' | '}' | '`' | '$'))
        {
            bail!("SWE index URL refused");
        }
        let url = reqwest::Url::parse(&index_url)
            .map_err(|_| anyhow::anyhow!("SWE index URL refused"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none_or(str::is_empty)
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            bail!(
                "prepared SWE index URL must be credential-free HTTP(S) without query or fragment"
            );
        }
        Ok(Self {
            deployment,
            index_url,
        })
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    schema_version: u8,
    parent: String,
    agent: String,
    lock_sha256: String,
    patch_profile: String,
    configuration: Configuration,
    uv: PathBuf,
    python: PathBuf,
    git_sha256: String,
    tool_pins: BTreeMap<PathBuf, String>,
    environment_pins: BTreeMap<PathBuf, String>,
    agent_package_pins: BTreeMap<PathBuf, String>,
    modules: BTreeMap<String, PathBuf>,
    benchmark_qualified: bool,
}
fn base(root: &Path) -> PathBuf {
    root.join("swe-sdk-v1")
}
fn project(root: &Path) -> PathBuf {
    base(root).join("project")
}
pub(super) fn agent(root: &Path) -> PathBuf {
    project(root).join("source/swe-bench-pro/SWE-agent")
}
pub(super) fn runtime_python(root: &Path) -> PathBuf {
    base(root).join("environment/bin/python")
}
fn original(root: &Path) -> PathBuf {
    harness_dir(root, registry::definition(EvalId::SweBenchPro))
}
pub(super) fn admit(root: &Path) -> Result<()> {
    if !cfg!(unix) {
        bail!("prepared SWE SDK supports Unix only");
    }
    let bytes = sdk_environment::read(&base(root).join("receipt.json"), 8 * 1048576)
        .context("SWE SDK is unprepared; run eval prepare-swe explicitly")?;
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    let configuration = Configuration::runtime()?;
    validate_receipt_contract(&receipt, &configuration)?;
    harness_source::admit_run(root, registry::definition(EvalId::SweBenchPro))?;
    custody::admit(root, &receipt)
}
pub(super) fn prepare(args: EvalPrepareSweArgs) -> Result<()> {
    preparation::prepare(args)
}
fn patch_profile(configuration: &Configuration) -> &'static str {
    match configuration.deployment {
        SweDeployment::Docker => "swerex-1.4.0-docker-index-v1",
        SweDeployment::Modal => "swerex-1.4.0-modal-modern-v3",
    }
}

fn validate_receipt_contract(receipt: &Receipt, configuration: &Configuration) -> Result<()> {
    if receipt.schema_version != 1
        || receipt.parent != registry::SWE_BENCH_PRO_REF
        || receipt.agent != registry::SWE_AGENT_REF
        || receipt.lock_sha256 != LOCK_SHA
        || receipt.patch_profile != patch_profile(configuration)
        || receipt.configuration != *configuration
        || receipt.benchmark_qualified
    {
        bail!("prepared SWE receipt/configuration refused");
    }
    Ok(())
}
