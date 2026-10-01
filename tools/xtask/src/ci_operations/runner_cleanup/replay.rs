use super::{Error, admission::validate_path};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

pub(crate) fn owned_worktrees(root: &Path, porcelain: &[u8]) -> Result<Vec<PathBuf>, Error> {
    let mut owned = Vec::new();
    for field in porcelain.split(|byte| *byte == 0) {
        if let Some(bytes) = field.strip_prefix(b"worktree ") {
            let path = os_path(bytes)?;
            if path != root && path.starts_with(root) {
                validate_path(root, &path)?;
                if path.is_symlink() {
                    return Err(Error::ReplaySymlink(path));
                }
                owned.push(path);
            }
        }
    }
    Ok(owned)
}

fn os_path(bytes: &[u8]) -> Result<PathBuf, Error> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Ok(OsString::from_vec(bytes.to_vec()).into())
    }
    #[cfg(windows)]
    {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| Error::Input("Git worktree path is not Unicode"))?;
        Ok(OsString::from(text).into())
    }
}
