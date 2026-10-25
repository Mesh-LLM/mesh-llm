//! Stage one complete report, preserve the prior output on ordinary failure or observed cancellation.
use super::{
    dirty_file,
    observation::Observation,
    provenance::{Error, Result},
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
pub(super) const REPORT_LIMIT: u64 = 128 * 1024 * 1024;
pub(crate) enum Phase {
    BeforeReplacement,
    AfterReplacement,
}
#[derive(Debug, PartialEq, Eq)]
struct Existing {
    bytes: Vec<u8>,
    identity: dirty_file::ReportIdentity,
}
pub(crate) struct Prepared {
    root: PathBuf,
    destination: PathBuf,
    name: String,
    original: Option<Existing>,
    bytes: Vec<u8>,
    staged: Option<tempfile::NamedTempFile>,
    restore: Option<tempfile::NamedTempFile>,
}
fn io() -> Error {
    Error("release inventory report publication failed".into())
}
impl Prepared {
    pub(crate) fn stage(path: &Path, bytes: &[u8], observation: &Observation) -> Result<Self> {
        observation.check()?;
        if bytes.len() as u64 > REPORT_LIMIT {
            return Err(Error("release inventory report exceeds128MiB".into()));
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|_| io())?;
        let root = parent.canonicalize().map_err(|_| io())?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(io)?
            .to_owned();
        dirty_file::relative(&name)?;
        let destination = root.join(&name);
        let original = existing(&root, &name, observation)?;
        let permissions = fs::symlink_metadata(&destination)
            .ok()
            .map(|m| m.permissions());
        let staged = temporary(&root, bytes, permissions.clone())?;
        let restore = original
            .as_ref()
            .map(|old| temporary(&root, &old.bytes, permissions))
            .transpose()?;
        observation.check()?;
        Ok(Self {
            root,
            destination,
            name,
            original,
            bytes: bytes.to_vec(),
            staged: Some(staged),
            restore,
        })
    }
    pub(crate) fn owns_transient(&self, path: &Path) -> bool {
        self.staged.as_ref().is_some_and(|f| f.path() == path)
            || self.restore.as_ref().is_some_and(|f| f.path() == path)
    }
    pub(crate) fn publish(
        mut self,
        observation: &Observation,
        mut check: impl FnMut(Phase, &Self) -> Result<()>,
    ) -> Result<()> {
        observation.check()?;
        check(Phase::BeforeReplacement, &self)?;
        if existing(&self.root, &self.name, observation)? != self.original {
            return Err(Error(
                "release inventory destination changed before publication".into(),
            ));
        }
        let staged = self.staged.take().ok_or_else(io)?;
        let published_identity = dirty_file::report_identity(staged.as_file())?;
        // Keep the owned published handle live: a replacement writer cannot reuse its identity.
        let _published = staged.persist(&self.destination).map_err(|_| io())?;
        let result = check(Phase::AfterReplacement, &self).and_then(|()| observation.check());
        if let Err(error) = result {
            // Restoration must still work after cancellation; never overwrite a different concurrent writer.
            let restore_observation = Observation::new(
                crate::process::Cancellation::default(),
                std::time::Duration::from_secs(10),
            )?;
            if !existing(&self.root, &self.name, &restore_observation)?.is_some_and(|current| {
                current.bytes == self.bytes && current.identity.same_file(&published_identity)
            }) {
                return Err(Error(
                    "report failed and destination changed; refuses unsafe rollback".into(),
                ));
            }
            if let Some(restore) = self.restore.take() {
                restore.persist(&self.destination).map_err(|_| {
                    Error("report failed and prior output restoration failed".into())
                })?;
            } else {
                fs::remove_file(&self.destination).map_err(|_| io())?;
            }
            return Err(error);
        }
        Ok(())
    }
}
fn temporary(
    root: &Path,
    bytes: &[u8],
    permissions: Option<fs::Permissions>,
) -> Result<tempfile::NamedTempFile> {
    let mut file = tempfile::Builder::new()
        .prefix(".release-inventory-")
        .tempfile_in(root)
        .map_err(|_| io())?;
    file.write_all(bytes).map_err(|_| io())?;
    file.as_file().sync_all().map_err(|_| io())?;
    if let Some(permissions) = permissions {
        file.as_file()
            .set_permissions(permissions)
            .map_err(|_| io())?;
    }
    Ok(file)
}
fn existing(root: &Path, name: &str, observation: &Observation) -> Result<Option<Existing>> {
    match fs::symlink_metadata(root.join(name)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(io()),
        Ok(_) => dirty_file::report_state(root, name, observation, REPORT_LIMIT)
            .map(|(bytes, identity)| Some(Existing { bytes, identity })),
    }
}
