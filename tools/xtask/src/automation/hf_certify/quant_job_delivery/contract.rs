use super::*;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u32,
    pub workflow: String,
    pub timeout_secs: u64,
    pub runner: admission::Artifact,
    pub authority: Value,
    pub operator: Value,
    pub receipt_export: receipt_export::Config,
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        self.receipt_export.validate()?;
        let cap = match self.workflow.as_str() {
            "quantization" => 259200,
            "quantization-and-package" => 345600,
            _ => return Err("quant worker workflow refused".into()),
        };
        if self.schema_version != 1
            || !(30..=cap).contains(&self.timeout_secs)
            || self.operator["timeout_seconds"].as_u64() != Some(self.timeout_secs)
            || self.operator["workflow"] != self.workflow
            || !self.runner.path.is_absolute()
            || !bootstrap::contract::hex(&self.runner.sha256, 64)
            || self.timeout_secs <= self.receipt_export.export_budget_secs + 8
            || !self.receipt_export.credential_environment
            || self.receipt_export.credential_file.is_some()
            || !self.operator["window_template"]["credential_file"].is_null()
        {
            return Err("quant worker pinned runner/shared budget/secret/export refused".into());
        }
        let authority = &self.authority;
        let keys = [
            "schema_version",
            "image",
            "mesh_commit",
            "git_tree",
            "cpu_plan_receipt_sha256",
            "declared_estimate_usd",
            "max_cost_usd",
            "tool_kind",
            "profile_version",
        ];
        if authority
            .as_object()
            .is_none_or(|o| o.len() != keys.len() || keys.iter().any(|k| !o.contains_key(*k)))
            || authority["schema_version"] != 1
            || authority["tool_kind"] != self.operator["window_template"]["tool_kind"]
            || authority["profile_version"] != self.operator["window_template"]["profile_version"]
            || ["mesh_commit", "git_tree"].iter().any(|k| {
                authority[k]
                    .as_str()
                    .is_none_or(|s| !bootstrap::contract::hex(s, 40))
            })
            || authority["cpu_plan_receipt_sha256"]
                .as_str()
                .is_none_or(|s| !bootstrap::contract::hex(s, 64))
            || authority["declared_estimate_usd"]
                .as_f64()
                .is_none_or(|n| !n.is_finite() || n < 0.0)
            || authority["max_cost_usd"].as_f64().is_none_or(|n| {
                !n.is_finite() || n <= 0.0 || Some(n) < authority["declared_estimate_usd"].as_f64()
            })
        {
            return Err("quant worker supplied-tool resource declaration refused".into());
        }
        let image = authority["image"]
            .as_str()
            .ok_or("quant image declaration")?;
        let (name, pin) = image
            .rsplit_once("@sha256:")
            .ok_or("quant immutable image digest")?;
        if name.is_empty()
            || name.len() > 256
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/._-:".contains(&b))
            || !bootstrap::contract::hex(pin, 64)
        {
            return Err("quant image digest declaration refused".into());
        }
        let mut op = self.operator.clone();
        op["window_template"]["credential_file"] =
            json!("/work/owned-quant-publication-credential");
        let op: super::super::quant_job::contract::Input = serde_json::from_value(op)?;
        op.validate()?;
        Ok(())
    }
}
pub(super) fn transport(flag: &str, value: &str) -> DynResult<Vec<u8>> {
    match flag {
        "--input" => admission::read(Path::new(value), 8 * 1048576),
        "--mounted-input-environment" => super::super::job_request_transport::environment(value),
        "--input-environment" if value == "MESH_HF_JOB_INPUT" => {
            let value = std::env::var(value).map_err(|_| "quant Jobs secret input absent")?;
            if value.is_empty() || value.len() > 65536 {
                return Err("quant Jobs inline input byte bound".into());
            }
            Ok(value.into_bytes())
        }
        _ => Err("quant Jobs closed input transport".into()),
    }
}
pub(super) fn root(output: &str, input: &Input) -> DynResult<std::path::PathBuf> {
    if !Path::new(output).is_absolute() {
        return Err("quant output absolute path required".into());
    }
    let requested = std::path::absolute(output)?;
    let root = requested
        .parent()
        .ok_or("quant output parent")?
        .canonicalize()?
        .join(requested.file_name().ok_or("quant output leaf")?);
    let template = &input.operator["window_template"];
    for key in ["source_root", "work_root", "target_root"] {
        let p = Path::new(
            template[key]
                .as_str()
                .ok_or("quant source/work/target root")?,
        );
        let p = if p.exists() {
            p.canonicalize()?
        } else {
            p.parent()
                .ok_or("quant root parent")?
                .canonicalize()?
                .join(p.file_name().ok_or("quant root leaf")?)
        };
        if root.starts_with(&p) || p.starts_with(&root) {
            return Err("quant Jobs source/work/evidence ancestry refused".into());
        }
    }
    std::fs::create_dir(&root)?;
    Ok(root)
}
pub(super) fn credential(root: &Path) -> DynResult<tempfile::NamedTempFile> {
    use std::io::Write as _;
    let token = std::env::var("MESH_HF_PUBLICATION_TOKEN")
        .map_err(|_| "explicit quant publication secret absent")?;
    if token.is_empty()
        || token.len() > 4096
        || token
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        return Err("quant publication secret refused".into());
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    file.write_all(token.as_bytes())?;
    file.as_file().sync_all()?;
    Ok(file)
}
