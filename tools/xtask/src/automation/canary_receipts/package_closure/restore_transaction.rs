//! Atomic publication of a dedicated, fresh canary consumer checkout.
use super::{candidate_view, process};
use crate::command::DynResult;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(super) struct Publication {
    pub(super) stage: candidate_view::Owned,
    pub(super) staged_root: PathBuf,
    root: PathBuf,
    swapped: bool,
    committed: bool,
}
impl Publication {
    pub(super) fn new(root: &Path, stage: candidate_view::Owned) -> DynResult<Self> {
        let staged_root = stage.path.join("source");
        if !fs::symlink_metadata(root)?.is_dir()
            || fs::symlink_metadata(root)?.file_type().is_symlink()
            || !fs::symlink_metadata(&staged_root)?.is_dir()
        {
            return Err("restore publication requires real directory roots".into());
        }
        Ok(Self {
            stage,
            staged_root,
            root: root.to_owned(),
            swapped: false,
            committed: false,
        })
    }
    pub(super) fn publish(&mut self) -> DynResult<()> {
        process::check()?;
        exchange(&self.root, &self.staged_root)?;
        self.swapped = true;
        process::check()
    }
    pub(super) fn rollback(&mut self) -> DynResult<()> {
        if !self.swapped {
            return Ok(());
        }
        match process::cleanup(|| exchange(&self.root, &self.staged_root)) {
            Ok(()) => {
                self.swapped = false;
                Ok(())
            }
            Err(error) => {
                self.stage.retain();
                Err(format!(
                    "atomic restore rollback failed; previous checkout retained at {}: {error}",
                    self.staged_root.display()
                )
                .into())
            }
        }
    }
    pub(super) fn commit(mut self) -> DynResult<()> {
        process::check()?;
        self.committed = true;
        self.swapped = false;
        fs::remove_dir_all(&self.stage.path).map_err(|error| {
            self.stage.retain();
            format!(
                "restore committed; previous checkout cleanup failed at {}: {error}",
                self.stage.path.display()
            )
        })?;
        Ok(())
    }
}
impl Drop for Publication {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.rollback();
        }
    }
}

fn exchange(left: &Path, right: &Path) -> DynResult<()> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let left = CString::new(left.as_os_str().as_bytes())?;
        let right = CString::new(right.as_os_str().as_bytes())?;
        #[cfg(target_os = "macos")]
        let status = {
            unsafe extern "C" {
                fn renamex_np(
                    left: *const libc::c_char,
                    right: *const libc::c_char,
                    flags: libc::c_uint,
                ) -> libc::c_int;
            }
            // SAFETY: both terminated paths refer to owned, validated directories;
            // RENAME_SWAP (Darwin sys/stdio.h) atomically exchanges their names.
            unsafe { renamex_np(left.as_ptr(), right.as_ptr(), 0x00000002) }
        };
        #[cfg(target_os = "linux")]
        // SAFETY: paths are terminated and validated; exchange is same-filesystem.
        let status = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                left.as_ptr(),
                libc::AT_FDCWD,
                right.as_ptr(),
                libc::RENAME_EXCHANGE,
            )
        };
        if status != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (left, right);
        Err("atomic canary restore exchange is unsupported on this platform".into())
    }
}
