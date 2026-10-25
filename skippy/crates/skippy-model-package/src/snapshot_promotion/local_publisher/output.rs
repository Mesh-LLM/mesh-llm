use anyhow::{Result, bail};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
pub(super) struct Output {
    root: PathBuf,
}
impl Output {
    pub(super) fn fresh(path: &Path) -> Result<Self> {
        if !path.is_absolute() {
            bail!("fresh output absolute path required");
        }
        match std::fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            _ => bail!("fresh output path already present or ambiguous"),
        }
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output parent absent"))?;
        let canonical = parent.canonicalize()?;
        if !canonical.is_dir() {
            bail!("output parent directory refused");
        }
        let root = canonical.join(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("output name absent"))?,
        );
        let builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        let mut builder = builder;
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&root)?;
        Ok(Self { root })
    }
    pub(super) fn write<T: serde::Serialize>(
        &self,
        name: &str,
        value: &T,
        fresh: bool,
    ) -> Result<()> {
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() > 512 * 1024 {
            bail!("publisher receipt byte bound exceeded");
        }
        let mut staged = tempfile::NamedTempFile::new_in(&self.root)?;
        staged.write_all(&bytes)?;
        staged.flush()?;
        staged.as_file().sync_all()?;
        let path = self.root.join(name);
        if fresh {
            staged.persist_noclobber(path)?;
        } else {
            match std::fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.is_file() => (),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                _ => bail!("publisher checkpoint leaf refused"),
            }
            staged.persist(path)?;
        }
        Ok(())
    }
}
