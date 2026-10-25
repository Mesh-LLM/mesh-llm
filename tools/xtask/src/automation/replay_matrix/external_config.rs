use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(try_from = "String")]
pub(super) enum Engine {
    #[serde(
        rename = "llama.cpp",
        alias = "llama",
        alias = "LLAMA",
        alias = "LLAMA.CPP"
    )]
    Llama,
    #[serde(rename = "vllm", alias = "VLLM")]
    Vllm,
    #[serde(rename = "sglang", alias = "SGLANG")]
    Sglang,
}

impl TryFrom<String> for Engine {
    type Error = &'static str;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_ascii_lowercase().as_str() {
            "llama" | "llama.cpp" => Ok(Self::Llama),
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            _ => Err("engine must be llama.cpp, vllm or sglang"),
        }
    }
}

impl Engine {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama.cpp",
            Self::Vllm => "vllm",
            Self::Sglang => "sglang",
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct Arm {
    pub label: String,
    pub engine: Engine,
    pub executable: String,
    pub model: String,
    #[serde(default)]
    pub served_model: Option<String>,
    pub context_size: u64,
    pub max_concurrency: usize,
    pub tokenizer: Option<String>,
    pub hf_config: Option<String>,
    #[serde(default = "enabled")]
    pub prefix_cache: bool,
    #[serde(default = "batch")]
    pub batch_size: u64,
    #[serde(default = "ubatch")]
    pub ubatch_size: u64,
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default)]
    pub cwd: PathBuf,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Comparison {
    pub model: String,
}

#[derive(Deserialize, Serialize)]
struct Document {
    schema_version: u32,
    comparison: Comparison,
    arms: Vec<Arm>,
}

#[derive(Serialize)]
pub(super) struct Config {
    pub path: PathBuf,
    pub sha256: String,
    pub comparison: Comparison,
    pub arms: Vec<Arm>,
}

pub(super) fn load(path: &Path) -> DynResult<Config> {
    let path = expand_home(path)?.canonicalize()?;
    let raw = std::fs::read(&path)?;
    let mut document: Document = serde_json::from_slice(&raw)?;
    if document.schema_version != 1 || document.arms.is_empty() {
        return Err("external config requires schema_version 1 and nonempty arms".into());
    }
    nonempty(&document.comparison.model)?;
    let root = path.parent().ok_or("external config has no parent")?;
    let mut labels = std::collections::BTreeSet::new();
    for arm in &mut document.arms {
        arm.prepare(root, &document.comparison.model)?;
        if !labels.insert(&arm.label) {
            return Err("external arm labels must be unique".into());
        }
    }
    Ok(Config {
        path,
        sha256: hex::encode(Sha256::digest(&raw)),
        comparison: document.comparison,
        arms: document.arms,
    })
}

impl Arm {
    fn prepare(&mut self, root: &Path, comparison: &str) -> DynResult<()> {
        if self.label.is_empty()
            || [".", ".."].contains(&self.label.as_str())
            || !self
                .label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
        {
            return Err("external arm label must be a safe path component".into());
        }
        for value in [&self.executable, &self.model]
            .into_iter()
            .chain(self.tokenizer.iter())
            .chain(self.hf_config.iter())
            .chain(self.extra_args.iter())
        {
            nonempty(value)?;
        }
        let served = self.served_model.get_or_insert_with(|| comparison.into());
        nonempty(served)?;
        if self.context_size == 0
            || self.max_concurrency == 0
            || self.batch_size == 0
            || self.ubatch_size == 0
        {
            return Err("external engine budgets must be positive integers".into());
        }
        self.cwd = if self.cwd.as_os_str().is_empty() {
            root.into()
        } else {
            let cwd = expand_home(&self.cwd)?;
            std::path::absolute(if cwd.is_absolute() {
                cwd
            } else {
                root.join(cwd)
            })?
        };
        Ok(())
    }

    pub(super) fn served_model(&self) -> DynResult<&str> {
        self.served_model
            .as_deref()
            .ok_or_else(|| "unprepared external arm".into())
    }

    pub(super) fn validate_prepared(&self) -> DynResult<()> {
        if !self.cwd.is_absolute() {
            return Err("prepared external arm requires an absolute cwd".into());
        }
        let mut arm = self.clone();
        arm.prepare(&self.cwd, self.served_model()?)
    }
}

impl Config {
    pub(super) fn admit(
        &self,
        model: &str,
        mesh_labels: &[String],
        minimum_context: u64,
    ) -> DynResult<()> {
        if self.comparison.model != model {
            return Err("model must exactly match external comparison.model".into());
        }
        if minimum_context != 0 {
            return Err("runtime context qualification currently requires mesh arms".into());
        }
        if self.arms.iter().any(|arm| mesh_labels.contains(&arm.label)) {
            return Err("Mesh and external engine labels overlap".into());
        }
        Ok(())
    }
}

fn nonempty(value: &str) -> DynResult<()> {
    if value.is_empty() || value.contains('\0') {
        return Err("external engine strings must be nonempty and contain no NUL".into());
    }
    Ok(())
}
pub(super) fn expand_home(path: &Path) -> DynResult<PathBuf> {
    if path == Path::new("~") || path.starts_with("~/") {
        let home = std::env::var_os("HOME").ok_or("HOME is required for tilde path")?;
        return Ok(PathBuf::from(home).join(path.strip_prefix("~")?));
    }
    Ok(path.into())
}
fn enabled() -> bool {
    true
}
fn batch() -> u64 {
    2048
}
fn ubatch() -> u64 {
    512
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_identity_and_label_disjointness_are_admission_requirements() {
        let config = Config {
            path: "/fixture/config".into(),
            sha256: "fixture".into(),
            comparison: Comparison {
                model: "identity".into(),
            },
            arms: vec![
                serde_json::from_value(serde_json::json!({
                "label":"external", "engine":"vLlM", "executable":"fixture", "model":"provenance",
                "context_size":1,"max_concurrency":1}))
                .unwrap(),
            ],
        };
        assert!(config.admit("identity", &["mesh".into()], 0).is_ok());
        assert!(config.admit("different", &[], 0).is_err());
        assert!(config.admit("identity", &["external".into()], 0).is_err());
        assert!(config.admit("identity", &[], 131072).is_err());
    }
}
