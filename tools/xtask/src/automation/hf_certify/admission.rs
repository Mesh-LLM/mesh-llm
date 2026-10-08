//! Local certification identity admission; bulk reads run in a supervised worker.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
};
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(in crate::automation) struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Mode {
    ProjectorOnly,
    MtpAttach,
}
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub mode: Mode,
    pub binary: Artifact,
    pub supplied_mesh_revision: String,
    pub native_profile: String,
    pub projector: Artifact,
    pub target_parts: Vec<Artifact>,
    pub expected_parts: usize,
    pub mtp_draft: Option<Artifact>,
    pub layer_count: u32,
    pub mtp_layer_count: Option<u32>,
    pub ctx_size: u32,
    pub timeout_secs: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Receipt {
    pub schema_version: u64,
    pub request_sha256: String,
    pub admitted: Input,
}
pub(in crate::automation) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub(in crate::automation) fn read(path: &Path, limit: u64) -> DynResult<Vec<u8>> {
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("certification input must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("opened certification input not regular".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("certification byte bound exceeded".into());
    }
    Ok(bytes)
}
pub(in crate::automation) fn publish(path: &Path, value: &impl Serialize) -> DynResult<()> {
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("certification output must be fresh".into()),
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 1048576 {
        return Err("certification receipt exceeds 1MiB".into());
    }
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("output parent")?)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(path)?;
    Ok(())
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.native_profile != "standalone-static-skippy-quantize-cpu"
            || !(5..=3600).contains(&self.timeout_secs)
            || self.ctx_size == 0
            || self.layer_count == 0
            || self.mtp_layer_count == Some(0)
            || self.supplied_mesh_revision.len() != 40
            || !self
                .supplied_mesh_revision
                .bytes()
                .all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid certification schema/profile/budget/dimensions/revision".into());
        }
        if self.mode == Mode::MtpAttach
            && (self.mtp_draft.is_none()
                || self.expected_parts == 0
                || self.expected_parts != self.target_parts.len()
                || self.expected_parts > 1024)
        {
            return Err("MTP certification requires exact nonempty target roster and draft".into());
        }
        if self.mode == Mode::ProjectorOnly
            && (!self.target_parts.is_empty()
                || self.expected_parts != 0
                || self.mtp_draft.is_some())
        {
            return Err("projector-only must not claim target or MTP admission".into());
        }
        let mut paths = std::collections::BTreeSet::new();
        for artifact in std::iter::once(&self.binary)
            .chain(std::iter::once(&self.projector))
            .chain(self.target_parts.iter())
            .chain(self.mtp_draft.iter())
        {
            if !artifact.path.is_absolute()
                || artifact.sha256.len() != 64
                || !artifact
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                || !paths.insert(&artifact.path)
            {
                return Err(
                    "certification paths/digests must be absolute, unique and complete".into(),
                );
            }
        }
        // Preserve declared logical roster order; native report correlation checks it exactly.
        // Canonical aliases can reverse lexical names; uniqueness above remains mandatory.
        Ok(())
    }
}
pub(in crate::automation) fn observe(source: &Path, gguf: bool) -> DynResult<Artifact> {
    let path = source.canonicalize()?;
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err("certification artifact must resolve to regular file".into());
    }
    // Avoid opening a swapped FIFO/symlink before invoking the established byte digest owner.
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(&path)?;
    if !file.metadata()?.is_file() {
        return Err("opened artifact not regular".into());
    }
    if gguf {
        let mut magic = [0; 4];
        file.read_exact(&mut magic)?;
        if magic != *b"GGUF" {
            return Err("artifact GGUF magic refused".into());
        }
    }
    // This established helper reopens its path. Parent ownership bounds worker lifetime;
    // these observations are not a hostile replacement sandbox or loaded-file attestation.
    let actual = crate::product::digest::file_sha256(&path).map_err(|e| e.error)?;
    Ok(Artifact {
        path,
        sha256: actual,
    })
}
pub(in crate::automation) fn admit(artifact: &mut Artifact, gguf: bool) -> DynResult<()> {
    let observed = observe(&artifact.path, gguf)?;
    if observed.sha256 != artifact.sha256 {
        return Err("certification artifact SHA-256 mismatch".into());
    }
    *artifact = observed;
    Ok(())
}
pub(super) fn worker(args: &[String]) -> DynResult<()> {
    let [a, input, b, output] = args else {
        return Err("identity-worker --input FILE --output FILE".into());
    };
    if a != "--input" || b != "--output" {
        return Err("identity-worker closed flags".into());
    }
    match std::fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("identity-worker receipt must be fresh".into()),
    }
    let mut input: Input = serde_json::from_slice(&read(Path::new(input), 262144)?)?;
    input.validate()?;
    let request_sha256 = digest(&serde_json::to_vec(&input)?);
    admit(&mut input.binary, false)?;
    admit(&mut input.projector, true)?;
    for part in &mut input.target_parts {
        admit(part, true)?
    }
    if let Some(draft) = &mut input.mtp_draft {
        admit(draft, true)?
    }
    input.validate()?;
    publish(
        Path::new(output),
        &Receipt {
            schema_version: 1,
            request_sha256,
            admitted: input,
        },
    )
}
