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
                if entry.file_name() == "pyvenv.cfg" {
                    verify_python_venv_config(&entry.path())?;
                }
                if entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "pth")
                {
                    verify_python_path_file(&entry.path(), &root)?;
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

fn verify_python_venv_config(path: &Path) -> DynResult<()> {
    let bytes = crate::automation::receipt_files::bounded(&path.canonicalize()?, 1024 * 1024)?;
    let mut seen_site_packages = false;
    for line in std::str::from_utf8(&bytes)?.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or("prepared pyvenv.cfg requires key=value entries")?;
        let key = key.trim();
        if key.is_empty() {
            return Err("prepared pyvenv.cfg requires nonempty keys".into());
        }
        if key.eq_ignore_ascii_case("include-system-site-packages") {
            // System paths can execute unpinned .pth/sitecustomize hooks before
            // the client imports. Reject ambiguous or non-boolean declarations.
            if seen_site_packages || !value.trim().eq_ignore_ascii_case("false") {
                return Err("prepared environment must disable system site-packages".into());
            }
            seen_site_packages = true;
        }
    }
    if !seen_site_packages {
        return Err("prepared pyvenv.cfg must explicitly disable system site-packages".into());
    }
    Ok(())
}

fn verify_python_path_file(path: &Path, root: &Path) -> DynResult<()> {
    let bytes = crate::automation::receipt_files::bounded(&path.canonicalize()?, 1024 * 1024)?;
    for line in std::str::from_utf8(&bytes)?.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // .pth import statements execute arbitrary code before AgentX starts;
        // a tree hash cannot establish that they preserve module ownership.
        if line.starts_with("import ") || line.starts_with("import\t") {
            return Err("prepared environment cannot contain executable .pth imports".into());
        }
        let target = path
            .parent()
            .ok_or(".pth parent missing")?
            .join(line)
            .canonicalize()?;
        if !target.starts_with(root) {
            return Err("prepared .pth path escapes its environment".into());
        }
    }
    Ok(())
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
    fn verify_client_environment(&self) -> DynResult<()> {
        absolute(&self.client_environment.path)?;
        verify_python_venv_config(&self.client_environment.path.join("pyvenv.cfg")).map_err(
            |error| -> crate::command::DynError {
                format!("prepared client environment requires an isolated root pyvenv.cfg: {error}")
                    .into()
            },
        )?;
        // Python checks the executable directory first. Probe the actual path,
        // including case-insensitive filesystem resolution, rather than relying
        // only on directory-entry spelling during the generic tree walk.
        for directory in ["bin", "Scripts"] {
            let config = self
                .client_environment
                .path
                .join(directory)
                .join("pyvenv.cfg");
            match std::fs::symlink_metadata(&config) {
                Ok(_) => verify_python_venv_config(&config)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
    pub(super) fn pin(&mut self) -> DynResult<()> {
        self.verify_client_environment()?;
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
        self.verify_client_environment()?;
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
        super::standard_runtime::verify(&self.llama.path, &self.standard_runtime.path)?;
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
        let ffi = std::fs::read(root.join("skippy/crates/skippy-ffi/src/lib.rs"))?;
        let abi = crate::native_policy::package_version::abi_version(&ffi)?.to_string();
        self.validate_manifest(&manifest, &abi)
    }
    pub(super) fn validate_manifest(
        &self,
        manifest: &serde_json::Value,
        abi: &str,
    ) -> DynResult<()> {
        if manifest["runtime"]["backend"]["kind"] != self.backend.name()
            || manifest["build"]["llama_patched_sha"] != self.native_source_commit
            || manifest["build"]["llama_patch_digest"] != self.native_patch_digest
            || manifest["build"]["llama_upstream_sha"] != self.standard_upstream
            || manifest["runtime"]["skippy_abi"] != abi
        {
            return Err("runtime must match the owning FFI ABI, backend and native leaf".into());
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
