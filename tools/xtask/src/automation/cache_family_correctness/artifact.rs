//! Native artifact tool composition inside the already supervised admission worker.
use super::catalog::Input;
use crate::{
    command::DynResult,
    process::{self, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value as Arg},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation) struct Artifact {
    pub kind: Kind,
    pub tool: PathBuf,
    pub tool_sha256: String,
    pub shard_pins: BTreeMap<String, String>,
}
#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(in crate::automation) enum Kind {
    CompleteShards,
    LayerPackage,
}
impl Artifact {
    pub(super) fn validate(&self, input: &Input) -> DynResult<()> {
        let digest = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        };
        if !self.tool.is_absolute() || !digest(&self.tool_sha256) || self.shard_pins.len() > 128 {
            return Err("artifact tool pin/profile refused".into());
        }
        match self.kind {
            Kind::CompleteShards
                if input.case_key == "minimax_m27" && self.shard_pins.len() == 3 =>
            {
                let prefix = "MiniMax-M2.7-UD-Q2_K_XL";
                for i in 1..=3 {
                    if self
                        .shard_pins
                        .get(&format!("{prefix}-{i:05}-of-00003.gguf"))
                        .is_none_or(|v| !digest(v))
                    {
                        return Err("MiniMax requires all three exact sibling pins".into());
                    }
                }
                let primary = format!("{prefix}-00001-of-00003.gguf");
                if input.model.file_name().and_then(|s| s.to_str()) != Some(primary.as_str())
                    || self.shard_pins.get(&primary) != Some(&input.model_sha256)
                {
                    return Err("MiniMax primary pin mismatch".into());
                }
            }
            Kind::LayerPackage
                if input.case_key == "deepseek3"
                    && self.shard_pins.is_empty()
                    && input.ctx_size == 32
                    && input.prefix_tokens == 4
                    && input.topologies.len() == 1
                    && matches!(input.topologies[0], super::catalog::Topology::PackageStage1) => {}
            _ => return Err(
                "artifact kind requires exact current MiniMax or DeepSeek3 package-only profile"
                    .into(),
            ),
        }
        Ok(())
    }
}
pub(super) fn inspect(input: &mut Input) -> DynResult<Value> {
    let mut artifact = input.artifact.clone().ok_or("artifact profile absent")?;
    let mut profile = Inspection {
        case_key: input.case_key.clone(),
        model: input.model.clone(),
        model_sha256: input.model_sha256.clone(),
        native_build: input.native_build.clone(),
        ctx_size: input.ctx_size,
        cell_seconds: input.cell_seconds,
        settings: input.settings.clone(),
        model_id: input.model_id.clone(),
    };
    let value = inspect_profile(&mut profile, &mut artifact)?;
    input.artifact = Some(artifact);
    Ok(value)
}
pub(in crate::automation) struct Inspection {
    pub case_key: String,
    pub model: PathBuf,
    pub model_sha256: String,
    pub native_build: PathBuf,
    pub ctx_size: u32,
    pub cell_seconds: u64,
    pub settings: BTreeMap<String, String>,
    pub model_id: String,
}
pub(in crate::automation) fn inspect_profile(
    input: &mut Inspection,
    artifact: &mut Artifact,
) -> DynResult<Value> {
    if (artifact.kind == Kind::CompleteShards && input.case_key != "minimax_m27")
        || (artifact.kind == Kind::LayerPackage && input.case_key != "deepseek3")
    {
        return Err("artifact inspection case/profile mismatch".into());
    }
    artifact.tool = artifact.tool.canonicalize()?;
    if !std::fs::symlink_metadata(&artifact.tool)?.is_file()
        || crate::product::digest::file_sha256(&artifact.tool).map_err(|e| e.error)?
            != artifact.tool_sha256
    {
        return Err("artifact tool SHA refused".into());
    }
    let mut args = Vec::new();
    match artifact.kind {
        Kind::CompleteShards => {
            args.extend([
                "admit-source".into(),
                input.model.clone().into_os_string(),
                "--minimum-context".into(),
                input.ctx_size.to_string().into(),
            ]);
            for (name, sha) in &artifact.shard_pins {
                args.extend(["--pin".into(), format!("{name}={sha}").into()]);
            }
        }
        Kind::LayerPackage => args.extend([
            "admit-package".into(),
            input.model.clone().into_os_string(),
            "--manifest-sha256".into(),
            input.model_sha256.clone().into(),
            "--model-id".into(),
            input.model_id.clone().into(),
            "--layer-start".into(),
            "3".into(),
            "--layer-end".into(),
            "4".into(),
            "--minimum-context".into(),
            "32".into(),
        ]),
    }
    let mut environment: BTreeMap<_, _> = input
        .settings
        .iter()
        .map(|(k, v)| (k.clone().into(), Arg::Public(v.clone().into())))
        .collect();
    environment.insert(
        "LLAMA_STAGE_BUILD_DIR".into(),
        Arg::Public(input.native_build.clone().into_os_string()),
    );
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancel = interrupt.cancellation();
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: artifact.tool.clone(),
            arguments: args.into_iter().map(Arg::Public).collect(),
            cwd: input.native_build.clone(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(input.cell_seconds.saturating_sub(3).max(1)),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &cancel,
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1024 * 1024),
            stderr: None,
        },
    );
    let finish = interrupt.finish();
    let result = result?;
    finish?;
    let p = &result.process;
    let raw = result
        .stdout
        .as_ref()
        .ok_or("artifact raw receipt absent")?;
    if !p.success()
        || !p.cleanup.complete
        || p.cleanup.forced
        || p.cleanup.graceful_signal_failed
        || p.cleanup.failure.is_some()
        || raw.as_bytes().len() as u64 != p.stdout.bytes_seen
        || [&p.stdout, &p.stderr]
            .iter()
            .any(|s| !s.line_capture_complete || s.truncated || s.oversized_lines > 0)
    {
        return Err("artifact inspector lifecycle/capture refused".into());
    }
    let mut receipt: Value = serde_json::from_slice(raw.as_bytes())?;
    validate_receipt(&receipt, input, artifact)?;
    receipt["tool_stdout_suppressed_lines"] = json!(p.stdout.suppressed_lines);
    Ok(receipt)
}

fn validate_receipt(receipt: &Value, input: &Inspection, artifact: &Artifact) -> DynResult<()> {
    if receipt["schema_version"] != 1 {
        return Err("artifact receipt schema refused".into());
    }
    let layer = receipt["dimensions"]["layer_count"]
        .as_u64()
        .ok_or("artifact layers absent")?;
    let width = receipt["dimensions"]["activation_width"]
        .as_u64()
        .ok_or("artifact width absent")?;
    if layer < 3
        || width == 0
        || receipt["dimensions"]["native_context_tokens"]
            .as_u64()
            .is_none_or(|n| n < u64::from(input.ctx_size))
    {
        return Err("artifact context/dimensions refused".into());
    }
    match artifact.kind {
        Kind::CompleteShards => {
            if layer != 62
                || width != 3072
                || receipt["kind"] != "gguf-source"
                || receipt["primary"] != json!(input.model)
            {
                return Err("source receipt primary refused".into());
            }
            let files = receipt["ordered_files"]
                .as_array()
                .ok_or("source roster absent")?;
            if files.len() != 3 {
                return Err("source roster incomplete".into());
            }
            for (i, file) in files.iter().enumerate() {
                let name = format!("MiniMax-M2.7-UD-Q2_K_XL-{:05}-of-00003.gguf", i + 1);
                if file["sha256"] != json!(artifact.shard_pins[&name])
                    || file["logical_path"]
                        != json!(input.model.parent().ok_or("shard parent")?.join(name))
                {
                    return Err("ordered source pins refused".into());
                }
            }
        }
        Kind::LayerPackage => {
            if receipt["kind"] != "layer-package"
                || receipt["root"] != json!(input.model)
                || receipt["manifest_sha256"] != input.model_sha256
                || receipt["model_id"] != input.model_id
                || layer != 61
                || width != 7168
                || receipt["state_layer_start"] != 3
                || receipt["state_layer_end"] != 4
                || receipt["independent_full_source_verified"] != false
                || receipt["baseline"] != "n/a-package-only"
                || receipt["selected_files"]
                    .as_array()
                    .is_none_or(Vec::is_empty)
            {
                return Err("DeepSeek3 package-only receipt refused".into());
            }
        }
    }
    Ok(())
}

impl Artifact {
    pub(in crate::automation) fn validate_serving(
        &self,
        model: &std::path::Path,
        sha: &str,
        layers: u32,
    ) -> DynResult<()> {
        let prefix = "MiniMax-M2.7-UD-Q2_K_XL";
        let primary = format!("{prefix}-00001-of-00003.gguf");
        if self.kind != Kind::CompleteShards
            || layers != 62
            || !self.tool.is_absolute()
            || self.tool_sha256.len() != 64
            || !self
                .tool_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || self.shard_pins.len() != 3
            || model.file_name().and_then(|s| s.to_str()) != Some(primary.as_str())
            || self.shard_pins.get(&primary).is_none_or(|v| v != sha)
        {
            return Err("serving requires complete MiniMax shard profile".into());
        }
        for i in 1..=3 {
            if self
                .shard_pins
                .get(&format!("{prefix}-{i:05}-of-00003.gguf"))
                .is_none_or(|v| {
                    v.len() != 64
                        || !v
                            .bytes()
                            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                })
            {
                return Err("complete serving shard pins refused".into());
            }
        }
        Ok(())
    }
}
