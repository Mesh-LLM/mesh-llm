//! Prepared offline tools: byte identities are authoritative, source IDs are provenance.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub(super) enum Backend {
    Cpu,
    Cuda,
    Metal,
}
impl Backend {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct File {
    pub path: PathBuf,
    pub sha256: String,
}
impl File {
    pub(super) fn verify(&self) -> DynResult<()> {
        absolute(&self.path)?;
        if crate::product::digest::file_sha256(&self.path).map_err(|e| e.error)? != self.sha256 {
            return Err(format!("prepared file SHA mismatch: {}", self.path.display()).into());
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Tree {
    pub path: PathBuf,
    pub sha256: String,
}
impl Tree {
    pub(super) fn verify(&self) -> DynResult<()> {
        if self.observe()? != self.sha256 {
            return Err(format!("prepared tree SHA mismatch: {}", self.path.display()).into());
        }
        Ok(())
    }
    pub(super) fn observe(&self) -> DynResult<String> {
        absolute(&self.path)?;
        // Unlike the runtime-only validator, a prepared Python environment may
        // contain interpreter links outside the environment. Charge their actual
        // regular-file bytes. Internal directory aliases (venv lib64 -> lib)
        // are hashed as aliases without traversal; external editable installs
        // remain refused.
        let root = self.path.canonicalize()?;
        let mut aliases = Vec::new();
        let mut pending = vec![(self.path.clone(), 0)];
        let mut count = 0usize;
        while let Some((dir, depth)) = pending.pop() {
            if depth > 64 {
                return Err("prepared tree exceeds64 levels".into());
            }
            for entry in std::fs::read_dir(dir)? {
                let entry = entry?;
                count += 1;
                if count > 250_000 {
                    return Err("prepared tree exceeds250000 entries".into());
                }
                entry
                    .file_name()
                    .to_str()
                    .ok_or("prepared tree names must be Unicode")?;
                let kind = entry.file_type()?;
                if kind.is_dir() {
                    pending.push((entry.path(), depth + 1));
                } else if kind.is_symlink() && entry.path().canonicalize()?.is_dir() {
                    let target = entry.path().canonicalize()?;
                    if !target.starts_with(&root) {
                        return Err("prepared directory alias escapes its root".into());
                    }
                    aliases.push((
                        entry
                            .path()
                            .strip_prefix(&self.path)?
                            .to_str()
                            .ok_or("alias path Unicode required")?
                            .to_owned(),
                        target
                            .strip_prefix(&root)?
                            .to_str()
                            .ok_or("alias target Unicode required")?
                            .to_owned(),
                    ));
                } else if !kind.is_file()
                    && !(kind.is_symlink() && entry.path().canonicalize()?.is_file())
                {
                    return Err("prepared trees require regular files or regular-file links".into());
                }
            }
        }
        if count == 0 {
            return Err("prepared tree is empty".into());
        }
        let files = crate::product::digest::tree_sha256(&self.path).map_err(|e| e.error)?;
        if aliases.is_empty() {
            return Ok(files);
        }
        use sha2::{Digest, Sha256};
        aliases.sort();
        let mut hash = Sha256::new();
        hash.update(b"agentx-prepared-tree-aliases-v1\0");
        hash.update(files.as_bytes());
        for (path, target) in aliases {
            for value in [path, target] {
                hash.update((value.len() as u64).to_be_bytes());
                hash.update(value.as_bytes());
            }
        }
        Ok(hex::encode(hash.finalize()))
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Tools {
    pub backend: Backend,
    pub product_source_commit: String,
    pub native_source_commit: String,
    pub native_patch_digest: String,
    pub runtime: Tree,
    pub skippy: File,
    pub llama: File,
    pub standard_runtime: Tree,
    pub standard_upstream: String,
    pub client: File,
    pub client_environment: Tree,
    pub client_lock: File,
    pub tokenizer: Tree,
    pub threads: u32,
    /// None preserves the product's automatic lane plan independently of offered C.
    pub diagnostic_generation_lanes: Option<u32>,
    pub diagnostic_cpu: bool,
    pub cell_seconds: u64,
}
impl Tools {
    pub(super) fn pin(&mut self) -> DynResult<()> {
        for file in [
            &mut self.skippy,
            &mut self.llama,
            &mut self.client,
            &mut self.client_lock,
        ] {
            file.sha256 = crate::product::digest::file_sha256(&file.path).map_err(|e| e.error)?;
        }
        for tree in [
            &mut self.runtime,
            &mut self.standard_runtime,
            &mut self.client_environment,
            &mut self.tokenizer,
        ] {
            tree.sha256 = tree.observe()?;
        }
        Ok(())
    }
    pub(super) fn verify(&self, root: &Path) -> DynResult<()> {
        for source in [
            &self.product_source_commit,
            &self.native_source_commit,
            &self.standard_upstream,
        ] {
            if !hex(source, 40) {
                return Err("prepared source requires full40-hex commit".into());
            }
        }
        if self.threads == 0
            || self.threads > 1024
            || !(60..=14_400).contains(&self.cell_seconds)
            || self
                .diagnostic_generation_lanes
                .is_some_and(|n| n == 0 || n > 256)
            || (self.backend == Backend::Cpu && !self.diagnostic_cpu)
        {
            return Err("invalid prepared bounds or CPU performance qualification request".into());
        }
        for file in [&self.skippy, &self.llama, &self.client, &self.client_lock] {
            file.verify()?;
        }
        for tree in [
            &self.runtime,
            &self.standard_runtime,
            &self.client_environment,
            &self.tokenizer,
        ] {
            tree.verify()?;
        }
        let lock = std::fs::read(root.join("skippy/evals/agentx/uv.lock"))?;
        if super::inputs::digest(&lock) != self.client_lock.sha256 {
            return Err("client lock differs from the owned external AgentX lock".into());
        }
        if std::fs::read_to_string(root.join("skippy/llama_cpp/upstream.txt"))?.trim()
            != self.standard_upstream
        {
            return Err("standard llama must use the candidate's unpatched upstream pin".into());
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(self.runtime.path.join("manifest.json"))?)?;
        self.validate_manifest(&manifest)
    }
    pub(super) fn validate_manifest(&self, manifest: &serde_json::Value) -> DynResult<()> {
        if manifest["runtime"]["backend"]["kind"] != self.backend.name()
            || manifest["build"]["llama_patched_sha"] != self.native_source_commit
            || manifest["build"]["llama_patch_digest"] != self.native_patch_digest
            || manifest["build"]["llama_upstream_sha"] != self.standard_upstream
            || manifest["runtime"]["skippy_abi"] != "0.1.76"
        {
            return Err("runtime must be exact matching ABI76 backend/native leaf".into());
        }
        Ok(())
    }
}
pub(super) fn absolute(path: &Path) -> DynResult<()> {
    if !path.is_absolute() {
        return Err("prepared paths must be absolute".into());
    }
    Ok(())
}
fn hex(value: &str, n: usize) -> bool {
    value.len() == n && value.bytes().all(|b| b.is_ascii_hexdigit())
}
