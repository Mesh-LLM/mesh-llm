use super::super::policy::ArtifactIdentity;
use anyhow::{Result, bail};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, Write},
    time::Instant,
};
pub struct Secret(String);
impl Secret {
    pub fn new(value: String) -> Result<Self> {
        if value.is_empty()
            || value.len() > 8192
            || value
                .bytes()
                .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
        {
            bail!("publication credential refused");
        }
        Ok(Self(value))
    }
    pub(in crate::snapshot_promotion) fn expose(&self) -> &str {
        &self.0
    }
}
impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret([redacted])")
    }
}
pub struct LocalFile {
    pub path_in_repo: String,
    pub file: File,
    pub identity: ArtifactIdentity,
}
pub struct Plan {
    pub repo: String,
    pub parent_commit: String,
    pub paths: Vec<String>,
}
pub(in crate::snapshot_promotion) fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(in crate::snapshot_promotion) fn check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        bail!("publication deadline expired");
    }
    Ok(())
}
impl Plan {
    pub(in crate::snapshot_promotion) fn validate(&self) -> Result<()> {
        let pieces = self.repo.split('/').collect::<Vec<_>>();
        if pieces.len() != 2
            || pieces.iter().any(|p| {
                p.is_empty()
                    || matches!(*p, "." | "..")
                    || p.len() > 96
                    || !p
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            })
            || !hex(&self.parent_commit, 40)
            || self.paths.is_empty()
            || self.paths.len() > 32
        {
            bail!("publication repo/parent/roster refused");
        }
        let mut seen = BTreeSet::new();
        for path in &self.paths {
            if path.len() > 256
                || !seen.insert(path)
                || path.split('/').any(|p| {
                    p.is_empty()
                        || matches!(p, "." | "..")
                        || !p
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                })
                || !([".json", ".md", ".txt"]
                    .iter()
                    .any(|suffix| path.ends_with(suffix)))
            {
                bail!("regular metadata path refused; GGUF/LFS unsupported");
            }
        }
        Ok(())
    }
}
pub(in crate::snapshot_promotion) struct Artifact {
    pub identity: ArtifactIdentity,
    pub spool: tempfile::NamedTempFile,
    pub original: File,
}
pub(in crate::snapshot_promotion) struct Staged {
    pub files: BTreeMap<String, Artifact>,
}
fn digest(
    file: &mut File,
    expected: &ArtifactIdentity,
    deadline: Instant,
    mut output: Option<&mut File>,
) -> Result<()> {
    check(deadline)?;
    if !file.metadata()?.is_file()
        || expected.byte_size > 1024 * 1024
        || !hex(&expected.sha256, 64)
        || file.metadata()?.len() != expected.byte_size
    {
        bail!("local regular file identity/type/bound refused");
    }
    file.rewind()?;
    let mut magic = [0; 4];
    let count = file.read(&mut magic)?;
    if count == 4 && magic == *b"GGUF" {
        bail!("GGUF publication requires unsupported LFS stage");
    }
    file.rewind()?;
    let mut hash = Sha256::new();
    let mut size = 0_u64;
    let mut bytes = [0_u8; 65536];
    loop {
        check(deadline)?;
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        size += count as u64;
        if size > expected.byte_size {
            bail!("local source grew during publication staging");
        }
        hash.update(&bytes[..count]);
        if let Some(ref mut output) = output {
            output.write_all(&bytes[..count])?;
        }
    }
    if size != expected.byte_size
        || hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
            != expected.sha256
    {
        bail!("local source byte identity mismatch");
    }
    Ok(())
}
impl Staged {
    pub(in crate::snapshot_promotion) fn new(
        plan: &Plan,
        files: Vec<LocalFile>,
        deadline: Instant,
    ) -> Result<Self> {
        if files.len() != plan.paths.len() {
            bail!("publication local roster mismatch");
        }
        let mut staged = BTreeMap::new();
        let mut total = 0_u64;
        for mut file in files {
            if !plan.paths.contains(&file.path_in_repo) || staged.contains_key(&file.path_in_repo) {
                bail!("publication duplicate/unrequested local path");
            }
            total = total
                .checked_add(file.identity.byte_size)
                .ok_or_else(|| anyhow::anyhow!("publication aggregate overflow"))?;
            if total > 8 * 1024 * 1024 {
                bail!("regular publication total bound exceeded");
            }
            let mut spool = tempfile::NamedTempFile::new()?;
            digest(
                &mut file.file,
                &file.identity,
                deadline,
                Some(spool.as_file_mut()),
            )?;
            spool.flush()?;
            spool.as_file().sync_all()?;
            staged.insert(
                file.path_in_repo,
                Artifact {
                    identity: file.identity,
                    spool,
                    original: file.file,
                },
            );
        }
        Ok(Self { files: staged })
    }
    pub(in crate::snapshot_promotion) fn recheck(&mut self, deadline: Instant) -> Result<()> {
        for file in self.files.values_mut() {
            digest(&mut file.original, &file.identity, deadline, None)?;
        }
        Ok(())
    }
}
