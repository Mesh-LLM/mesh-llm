use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub(super) const CATALOG: &[u8] =
    include_bytes!("../../../../../skippy/evals/kv-validation/models.json");
pub(super) const CATALOG_SHA: &str =
    "6a2b724cf32208186d80ac614e21a168bfc08ca72e55afdabc3dc39b338c9206";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Backend {
    Cpu,
    Metal,
    Cuda,
}
impl Backend {
    pub(super) fn device(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Metal => "MTL0",
            Self::Cuda => "CUDA0",
        }
    }
    pub(super) fn kind(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
    pub(super) fn pairs(self) -> usize {
        if self == Self::Metal { 8 } else { 64 }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Tool {
    pub path: PathBuf,
    pub sha256: String,
}
impl Tool {
    pub(super) fn verify(&self) -> DynResult<()> {
        absolute(&self.path)?;
        let actual = crate::product::digest::file_sha256(&self.path).map_err(|e| e.error)?;
        if actual != self.sha256 {
            return Err("KV validation file SHA-256 mismatch".into());
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Producer {
    pub schema_version: u32,
    pub backend: Backend,
    pub workspace: PathBuf,
    pub native_source: PathBuf,
    pub native_source_commit: String,
    pub runtime_bundle: PathBuf,
    pub runtime_sha256: String,
    pub cli: Tool,
    pub correctness: Tool,
    pub serving_tests: Tool,
    pub just: PathBuf,
    pub cmake: PathBuf,
}
impl Producer {
    pub(super) fn verify(&self) -> DynResult<()> {
        if self.schema_version != 1 || !sha(&self.native_source_commit, 40) {
            return Err("invalid KV producer schema/source".into());
        }
        for path in [
            &self.workspace,
            &self.native_source,
            &self.runtime_bundle,
            &self.just,
            &self.cmake,
        ] {
            absolute(path)?;
        }
        if !self.workspace.join("Justfile").is_file()
            || !self.native_source.join("src/llama-context.h").is_file()
        {
            return Err("KV producer requires owning workspace and prepared native source".into());
        }
        for tool in [&self.cli, &self.correctness, &self.serving_tests] {
            tool.verify()?;
        }
        super::verify_bundle(
            &self.runtime_bundle,
            &self.runtime_sha256,
            self.backend,
            &self.native_source_commit,
        )?;
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Prepared {
    pub schema_version: u32,
    pub catalog_sha256: String,
    pub producer: Producer,
    pub matrix: Tool,
}
impl Prepared {
    pub(super) fn verify(&self) -> DynResult<()> {
        if self.schema_version != 1 || self.catalog_sha256 != CATALOG_SHA {
            return Err("KV prepared receipt catalog/schema mismatch".into());
        }
        self.producer.verify()?;
        self.matrix.verify()
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Cell {
    pub schema_version: u32,
    pub prepared: Tool,
    pub model_key: String,
    pub model: PathBuf,
    pub port: u16,
    pub execution_seconds: u64,
}
impl Cell {
    pub(super) fn validate(&self) -> DynResult<Model> {
        if self.schema_version != 1
            || self.port < 1024
            || !(30..=14400).contains(&self.execution_seconds)
        {
            return Err("invalid bounded KV cell".into());
        }
        absolute(&self.model)?;
        models()?
            .into_iter()
            .find(|m| m.key == self.model_key)
            .ok_or_else(|| "KV cell requires one exact declared AgentX model".into())
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Model {
    pub key: String,
    pub family: String,
    pub repo: String,
    pub revision: String,
    pub tokenizer: String,
    pub tokenizer_revision: String,
    pub files: Vec<ModelFile>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelFile {
    pub name: String,
    pub sha256: String,
}
impl Model {
    pub(super) fn reference(&self) -> String {
        format!("hf:{}@{}#{}", self.repo, self.revision, self.files[0].name)
    }
    pub(super) fn recurrent_case(&self) -> bool {
        self.family.starts_with("hybrid-")
    }
}
pub(super) fn models() -> DynResult<Vec<Model>> {
    if super::hash(CATALOG) != CATALOG_SHA {
        return Err("declared AgentX catalog bytes changed".into());
    }
    let models: Vec<Model> = serde_json::from_slice(CATALOG)?;
    if models.len() != 4
        || models.iter().any(|m| {
            m.files.len() != 1
                || !sha(&m.revision, 40)
                || !sha(&m.tokenizer_revision, 40)
                || !sha(&m.files[0].sha256, 64)
        })
    {
        return Err("invalid four-model pinned catalog".into());
    }
    Ok(models)
}
pub(super) fn sha(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn absolute(path: &Path) -> DynResult<()> {
    if !path.is_absolute() {
        return Err("KV validation paths must be absolute".into());
    }
    Ok(())
}
