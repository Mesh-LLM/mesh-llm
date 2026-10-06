//! Native actual-file and static build provenance admission in a supervised current-exe worker.
use super::{adaptive_identity as io, kv_identity, native_identity, options};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Arm {
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub commit: String,
    pub model: PathBuf,
    pub model_sha256: String,
    pub model_id: String,
    pub native_build: PathBuf,
    pub native_build_sha256: String,
    pub ctx_size: u32,
    pub layer_end: u32,
    pub payload: String,
}
impl Arm {
    pub fn validate(&self) -> DynResult<()> {
        let hash = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if [&self.binary, &self.model, &self.native_build]
            .iter()
            .any(|p| !p.is_absolute())
            || !hash(&self.binary_sha256)
            || !hash(&self.model_sha256)
            || !hash(&self.native_build_sha256)
            || ![40, 64].contains(&self.commit.len())
            || !self
                .commit
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || self.model_id.trim().is_empty()
            || self.model_id.len() > 4096
            || self.ctx_size == 0
            || self.layer_end == 0
            || !matches!(
                self.payload.as_str(),
                "resident-kv" | "kv-recurrent" | "full-state"
            )
        {
            return Err("radix arm byte/source/model/profile identity invalid".into());
        }
        Ok(())
    }
}
pub(super) fn evidence(arm: &Arm) -> DynResult<Value> {
    arm.validate()?;
    let mut value = kv_identity::evidence(&kv_identity::Input {
        schema_version: 1,
        binary: arm.binary.clone(),
        model: arm.model.clone(),
    })?;
    if value["binary"]["sha256"] != arm.binary_sha256
        || value["model"]["sha256"] != arm.model_sha256
        || value["model_metadata"]["native_context_tokens"]
            .as_u64()
            .is_none_or(|n| n < u64::from(arm.ctx_size))
    {
        return Err("radix actual binary/model/context differs from declared identity".into());
    }
    let dimensions =
        crate::automation::replay_matrix::model_preflight::dimensions::inspect(Path::new(
            value["model"]["path"]
                .as_str()
                .ok_or("radix model path absent")?,
        ))?
        .ok_or("radix requires complete native model dimensions")?;
    if dimensions.block_count < u64::from(arm.layer_end) {
        return Err("radix stage layer range exceeds actual model blocks".into());
    }
    let native = native_identity::verify(&arm.native_build, &arm.native_build_sha256)?;
    let mut admitted = arm.clone();
    admitted.binary = PathBuf::from(
        value["binary"]["path"]
            .as_str()
            .ok_or("radix binary path absent")?,
    );
    admitted.model = PathBuf::from(
        value["model"]["path"]
            .as_str()
            .ok_or("radix model path absent")?,
    );
    admitted.native_build = arm.native_build.canonicalize()?;
    value["request_sha256"] = json!(io::digest(&serde_json::to_vec(arm)?));
    value["admitted"] = serde_json::to_value(&admitted)?;
    value["native_build_sha256"] = json!(native);
    value["native_profile"] = json!("standalone-static-skippy-server");
    value
        .as_object_mut()
        .unwrap()
        .remove("adjacent_runtime_root");
    value["runtime_policy"] = json!("standalone-static-skippy-server-executed-binary-authority");
    value["native_provenance_scope"] = json!(
        "executed static binary bytes plus declared native build source tree; not dynamically loaded package attestation"
    );
    Ok(value)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let input: Arm = serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 65536)?)?;
    let output = Path::new(flags["--output"]);
    if !output.is_absolute() {
        return Err("radix identity output requires absolute path".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => return Err("radix identity output must be fresh".into()),
    };
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let value = evidence(&input)?;
    if interrupt.cancellation().is_cancelled() {
        return Err("radix identity interrupted".into());
    }
    io::fresh(output, &serde_json::to_vec_pretty(&value)?)?;
    interrupt.finish()?;
    Ok(())
}
