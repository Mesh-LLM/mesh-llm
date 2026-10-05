//! Resolve a pinned A/B workload before launching either binary.
use super::{acceptance::Contract, fixture_profile, options, prompt_manifest, publish};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Workload {
    rounds: u64,
    families: u64,
    requests_per_family: u64,
    prefix_blocks: u64,
    output_tokens: u64,
    ctx_size: u64,
    lanes: u64,
    admission_concurrency: u64,
    cache_entries: u64,
    stagger_ms: f64,
}

impl Workload {
    fn requests_per_round(&self) -> DynResult<u64> {
        self.families
            .checked_mul(self.requests_per_family)
            .ok_or_else(|| "workload request count overflow".into())
    }

    fn validate(&self) -> DynResult<()> {
        if [
            self.rounds,
            self.families,
            self.requests_per_family,
            self.prefix_blocks,
            self.output_tokens,
            self.ctx_size,
            self.lanes,
            self.cache_entries,
        ]
        .contains(&0)
            || !self.stagger_ms.is_finite()
            || self.stagger_ms < 0.0
        {
            return Err("workload sizes must be positive and stagger nonnegative".into());
        }
        self.requests_per_round()?;
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CacheSeed {
    families: u64,
    prefix_blocks: u64,
    output_tokens: u64,
    stagger_ms: f64,
}

impl CacheSeed {
    fn validate(&self) -> DynResult<()> {
        if [self.families, self.prefix_blocks, self.output_tokens].contains(&0)
            || !self.stagger_ms.is_finite()
            || self.stagger_ms <= 0.0
        {
            return Err("cache seed sizes and stagger must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Overrides {
    cache_entries: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptanceDocument {
    schema_version: u64,
    name: String,
    #[serde(default, rename = "description")]
    _description: Option<String>,
    workload_profile: String,
    #[serde(default)]
    workload_overrides: Overrides,
    cache_seed: Option<CacheSeed>,
    hardware_acceptance: Value,
}

#[derive(Serialize)]
struct Plan {
    schema_version: u64,
    workload_profile: String,
    fixture_catalog_sha256: String,
    model: Value,
    workload: Workload,
    requests_per_round: u64,
    successful_requests_per_binary: u64,
    acceptance_contract_name: Option<String>,
    acceptance_contract_sha256: Option<String>,
    hardware_acceptance: Value,
    cache_seed: Option<CacheSeed>,
    prompt_manifest_sha256: Option<String>,
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn admit_prompts(
    profile: &Value,
    workload: &Workload,
    bytes: Option<&[u8]>,
) -> DynResult<Option<String>> {
    match (profile["corpus"]["kind"].as_str(), bytes) {
        (Some("synthetic"), None) => Ok(None),
        (Some("hf"), Some(bytes)) => {
            let digest = hash(bytes);
            if profile["corpus"]["prompt_manifest_sha256"].as_str() != Some(digest.as_str()) {
                return Err("prompt manifest differs from the pinned fixture SHA-256".into());
            }
            let manifest = prompt_manifest(bytes)?;
            let mut families = BTreeMap::<&str, u64>::new();
            for prompt in &manifest.prompts {
                *families.entry(&prompt.family).or_default() += 1;
            }
            if u64::try_from(families.len())? != workload.families
                || families
                    .values()
                    .any(|count| *count != workload.requests_per_family)
            {
                return Err(
                    "prompt manifest does not cover every workload family and request".into(),
                );
            }
            Ok(Some(digest))
        }
        _ => Err("prompt manifest must match the fixture corpus mode".into()),
    }
}

fn resolve(
    catalog_bytes: &[u8],
    profile_name: &str,
    model_id: &str,
    model_sha256: &str,
    contract_bytes: Option<&[u8]>,
    prompts: Option<&[u8]>,
) -> DynResult<Plan> {
    let catalog: Value = serde_json::from_slice(catalog_bytes)?;
    let profile = fixture_profile::resolve(&catalog, profile_name)?;
    if profile["model"]["id"].as_str() != Some(model_id)
        || profile["model"]["sha256"].as_str() != Some(model_sha256)
    {
        return Err("workload model identity differs from the pinned fixture".into());
    }
    let mut workload: Workload = serde_json::from_value(profile["workload"].clone())?;
    let mut hardware = profile["hardware_acceptance"].clone();
    let mut seed = None;
    let mut name = None;
    if let Some(bytes) = contract_bytes {
        let document: AcceptanceDocument = serde_json::from_slice(bytes)?;
        if document.schema_version != 1 || document.name.trim().is_empty() {
            return Err("acceptance contract requires schema 1 and a nonempty name".into());
        }
        if document.workload_profile != profile_name {
            return Err(
                "acceptance contract workload_profile differs from the selected fixture".into(),
            );
        }
        if let Some(entries) = document.workload_overrides.cache_entries {
            workload.cache_entries = entries;
        }
        if let Some(value) = &document.cache_seed {
            value.validate()?;
        }
        hardware = document.hardware_acceptance;
        seed = document.cache_seed;
        name = Some(document.name);
    }
    workload.validate()?;
    let acceptance: Contract = serde_json::from_value(hardware.clone())?;
    let requests_per_round = workload.requests_per_round()?;
    let successes = requests_per_round
        .checked_mul(workload.rounds)
        .ok_or("workload total request count overflow")?;
    if acceptance.successful_requests_per_binary != successes {
        return Err("acceptance success count differs from the complete workload".into());
    }
    let manifest_sha = admit_prompts(profile, &workload, prompts)?;
    Ok(Plan {
        schema_version: 1,
        workload_profile: profile_name.into(),
        fixture_catalog_sha256: hash(catalog_bytes),
        model: profile["model"].clone(),
        workload,
        requests_per_round,
        successful_requests_per_binary: successes,
        acceptance_contract_name: name,
        acceptance_contract_sha256: contract_bytes.map(hash),
        hardware_acceptance: hardware,
        cache_seed: seed,
        prompt_manifest_sha256: manifest_sha,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &[
            "--catalog",
            "--profile",
            "--model-id",
            "--model-sha256",
            "--contract",
            "--prompt-manifest",
            "--output",
        ],
        &[
            "--catalog",
            "--profile",
            "--model-id",
            "--model-sha256",
            "--output",
        ],
    )?;
    let catalog = std::fs::read(opts["--catalog"])?;
    let contract = opts.get("--contract").map(std::fs::read).transpose()?;
    let prompts = opts
        .get("--prompt-manifest")
        .map(std::fs::read)
        .transpose()?;
    let plan = resolve(
        &catalog,
        opts["--profile"],
        opts["--model-id"],
        opts["--model-sha256"],
        contract.as_deref(),
        prompts.as_deref(),
    )?;
    let mut output = serde_json::to_vec_pretty(&plan)?;
    output.push(b'\n');
    publish(Path::new(opts["--output"]), &output)
}

#[cfg(test)]
#[path = "workload_plan_tests.rs"]
mod tests;
