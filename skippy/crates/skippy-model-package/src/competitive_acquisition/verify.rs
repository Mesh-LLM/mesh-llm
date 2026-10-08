//! Read-only native artifact custody between acquisition and deterministic selection.
use super::{
    Request, acquire,
    contract::{check, digest},
    read,
};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn run(path: &Path, phase: &str) -> Result<()> {
    if !["before-manifest", "after-manifest"].contains(&phase) {
        bail!("custody phase refused");
    }
    let mut request = read::open(path, 1024 * 1024, false)?;
    let bytes = read::read(&mut request, 1024 * 1024)?;
    let input: Request = serde_json::from_slice(&bytes)?;
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(input.timeout_seconds))
        .ok_or_else(|| anyhow::anyhow!("custody deadline"))?;
    if !input.output_directory.is_absolute()
        || !std::fs::symlink_metadata(&input.output_directory)?.is_dir()
        || input.output_directory.canonicalize()? != input.output_directory
        || !(1..=86400).contains(&input.timeout_seconds)
    {
        bail!("owned custody directory/budget refused");
    }
    let mut file = read::open(&input.config, 1024 * 1024, false)?;
    let config_bytes = read::read(&mut file, 1024 * 1024)?;
    if digest(&config_bytes) != input.config_sha256 {
        bail!("custody config pin mismatch");
    }
    let config: Value = super::json::unique(&config_bytes)?;
    let mut receipt = read::open(
        &input.output_directory.join("acquisition.json"),
        8 * 1024 * 1024,
        false,
    )?;
    let final_bytes = read::read(&mut receipt, 8 * 1024 * 1024)?;
    let acquired: Value = super::json::unique(&final_bytes)?;
    if acquired["request_transport_sha256"] != json!(digest(&bytes))
        || acquired["status"] != json!("ACQUIRED_EXPORTED")
        || !acquired["error"].is_null()
    {
        bail!("custody final receipt correlation refused");
    }
    let selected = super::select(&config, &input)?;
    if acquired["families"]
        .as_array()
        .is_none_or(|v| v.len() != selected.len())
    {
        bail!("custody full family roster refused");
    }
    for model in selected {
        check(deadline)?;
        let key = model["key"].as_str().unwrap();
        let rows = BTreeMap::from([(
            model["filename"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("model filename"))?
                .to_string(),
            model["sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("model SHA"))?
                .to_string(),
        )]);
        acquire::recheck(
            &input.output_directory.join("models").join(key),
            &rows,
            deadline,
        )?;
        if !input.skip_vllm_configs {
            let source = model["vllm_hf_config"]["sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("config SHA"))?;
            acquire::recheck(
                &input.output_directory.join("vllm-configs").join(key),
                &BTreeMap::from([("config.json".into(), source.into())]),
                deadline,
            )?;
        }
        if !input.skip_tokenizers {
            let observed = acquired["families"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["key"] == json!(key))
                .ok_or_else(|| anyhow::anyhow!("custody family identity"))?;
            let source_pins: BTreeMap<String, String> =
                serde_json::from_value(observed["source"]["files"].clone())?;
            let source_root = input.output_directory.join("sources").join(key);
            if super::local::pins(&source_root, deadline)? != source_pins {
                bail!("immutable tokenizer source roster drift");
            }
            let exported = super::local::pins(
                &input.output_directory.join("tokenizers").join(key),
                deadline,
            )?;
            if super::contract::tree(&exported)? != input.export_sha256[key] {
                bail!("derived export custody failed");
            }
        }
    }
    if !input.skip_dataset {
        let dataset = &config["thoughtworks"]["dataset"];
        acquire::recheck(
            &input.output_directory.join("thoughtworks"),
            &BTreeMap::from([(
                dataset["filename"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("dataset filename"))?
                    .into(),
                dataset["sha256"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("dataset pin"))?
                    .into(),
            )]),
            deadline,
        )?;
    }
    if read::read(&mut request, 1024 * 1024)? != bytes
        || read::read(&mut file, 1024 * 1024)? != config_bytes
        || read::read(&mut receipt, 8 * 1024 * 1024)? != final_bytes
    {
        bail!("custody request/config/receipt changed");
    }
    check(deadline)?;
    super::publish(
        &input.output_directory.join(format!("custody-{phase}.json")),
        &serde_json::to_vec_pretty(
            &json!({"schema_version":1,"status":"CUSTODY_VERIFIED","phase":phase,"request_transport_sha256":digest(&bytes),"config_sha256":input.config_sha256,"acquisition_receipt_sha256":digest(&final_bytes)}),
        )?,
    )
}
