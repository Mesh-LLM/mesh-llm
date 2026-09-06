//! Filesystem facts the L3 disk tier has to know before it writes.
//!
//! The tier promises a minimum-free-space reserve, owner-only permissions, a
//! refusal to run on network filesystems, and a recency signal cheap enough to
//! update on every cache hit. None of that is in `std`, so the platform calls
//! live here and the store stays free of `unsafe`.

use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};

/// Bytes an unprivileged writer can still add to the filesystem holding
/// `path`. This is `f_bavail`, not `f_bfree`: the reserve check must not spend
/// blocks only root can allocate.
pub fn available_bytes(path: &Path) -> Result<u64> {
    let stat = statvfs(path)?;
    // Widths differ by platform (macOS reports a 32-bit block count, Linux a
    // 64-bit one), so widen by cast rather than by a conversion that is
    // identity on one target and fallible on the other.
    let blocks = stat.f_bavail as u64;
    let block_bytes = stat.f_frsize as u64;
    Ok(blocks.saturating_mul(block_bytes))
}

/// Whether `path` sits on a filesystem the tier refuses to manage. Network
/// filesystems break the atomic-rename and locking assumptions the store is
/// built on, so §10.10 rejects them where they are reliably detectable.
pub fn is_network_filesystem(path: &Path) -> Result<bool> {
    let name = filesystem_type_name(path)?;
    Ok(matches!(
        name.as_str(),
        "nfs" | "smbfs" | "afpfs" | "webdav" | "ftp" | "cifs" | "fuse" | "fuse.sshfs"
    ))
}

/// The filesystem type as the kernel names it, for status reporting.
pub fn filesystem_type_name(path: &Path) -> Result<String> {
    #[cfg(target_os = "macos")]
    {
        let stat = statfs(path)?;
        let raw = stat.f_fstypename;
        let bytes: Vec<u8> = raw
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux reports a magic number rather than a name. Only the values the
        // tier actually refuses are worth naming; anything else is local
        // enough to manage.
        let stat = statfs(path)?;
        Ok(match stat.f_type {
            0x6969 => "nfs".to_string(),
            0xFF53_4D42 => "cifs".to_string(),
            // FUSE_SUPER_MAGIC. statfs cannot name the subtype, so every FUSE
            // mount reports as plain "fuse": sshfs and a local fuse filesystem
            // are indistinguishable here. The tier refuses the whole class
            // rather than guess, which is the conservative reading of §10.10.
            0x6573_5546 => "fuse".to_string(),
            other => format!("0x{other:x}"),
        })
    }
}

/// Mark an entry as used now, so eviction can order by last use rather than
/// last write. One `utimensat` per cache hit is the bounded metadata update
/// §13.4 allows; it writes no payload bytes and allocates no blocks.
pub fn touch(path: &Path) -> Result<()> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the unix epoch")?;
    let times = [
        libc::timespec {
            tv_sec: now.as_secs() as libc::time_t,
            tv_nsec: libc::c_long::from(now.subsec_nanos() as i32),
        },
        libc::timespec {
            tv_sec: now.as_secs() as libc::time_t,
            tv_nsec: libc::c_long::from(now.subsec_nanos() as i32),
        },
    ];
    let c_path = c_path(path)?;
    // SAFETY: `c_path` is a NUL-terminated path that outlives the call, and
    // `times` is a two-element timespec array as utimensat requires.
    let status = unsafe { libc::utimensat(libc::AT_FDCWD, c_path.as_ptr(), times.as_ptr(), 0) };
    if status != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to touch {}", path.display()));
    }
    Ok(())
}

/// Restrict a cache directory to its owner. The first release has no at-rest
/// encryption, so local account permissions are the only confidentiality the
/// tier offers and it must actually apply them.
pub fn restrict_to_owner(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let permissions = fs::Permissions::from_mode(mode);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("failed to restrict permissions on {}", path.display()))
}

/// Refuse a path that reaches the store through a symlink. Following one would
/// let anything with write access to the parent redirect committed cache
/// bytes outside the managed root, past every budget and reserve check.
pub fn refuse_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(
                "{} is a symlink; the cache refuses to traverse it",
                path.display()
            )
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to stat {}", path.display())),
    }
}

/// Refuse a symlink anywhere in the portion of `path` below `root`.
///
/// The system prefix above the cache root is not ours to police: on macOS
/// `/var` is itself a symlink to `/private/var`, so refusing every symlinked
/// ancestor would reject the default temp and cache locations. What must hold
/// is that nothing the store creates under its own resolved root redirects
/// bytes outside it.
pub fn refuse_symlinked_descendant(root: &Path, path: &Path) -> Result<()> {
    let Ok(relative) = path.strip_prefix(root) else {
        bail!(
            "{} is not inside the cache root {}",
            path.display(),
            root.display()
        );
    };
    let mut walked = root.to_path_buf();
    for component in relative.components() {
        walked.push(component);
        refuse_symlink(&walked)?;
    }
    Ok(())
}

fn c_path(path: &Path) -> Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("path {} contains an interior NUL", path.display()))
}

fn statvfs(path: &Path) -> Result<libc::statvfs> {
    let c_path = c_path(path)?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is NUL-terminated and outlives the call; `stat` is
    // valid uninitialized storage the call fills on success.
    let status = unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if status != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to stat filesystem for {}", path.display()));
    }
    // SAFETY: statvfs returned 0, so `stat` is initialized.
    Ok(unsafe { stat.assume_init() })
}

fn statfs(path: &Path) -> Result<libc::statfs> {
    let c_path = c_path(path)?;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: as `statvfs` above.
    let status = unsafe { libc::statfs(c_path.as_ptr(), stat.as_mut_ptr()) };
    if status != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to stat filesystem for {}", path.display()));
    }
    // SAFETY: statfs returned 0, so `stat` is initialized.
    Ok(unsafe { stat.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn available_bytes_reports_a_plausible_figure() {
        let available = available_bytes(Path::new(".")).expect("stat working directory");
        assert!(available > 0, "working directory reports no free space");
    }

    #[test]
    fn touch_moves_modification_time_forward() {
        let directory =
            std::env::temp_dir().join(format!("skippy-fsinfo-touch-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create temp dir");
        let path = directory.join("entry");
        fs::write(&path, b"entry").expect("write entry");
        let before = fs::metadata(&path)
            .expect("stat")
            .modified()
            .expect("mtime");
        std::thread::sleep(std::time::Duration::from_millis(20));
        touch(&path).expect("touch entry");
        let after = fs::metadata(&path)
            .expect("stat")
            .modified()
            .expect("mtime");
        assert!(after > before, "touch did not move the modification time");
        fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn symlinks_are_refused() {
        let directory =
            std::env::temp_dir().join(format!("skippy-fsinfo-symlink-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create temp dir");
        let target = directory.join("target");
        let link = directory.join("link");
        fs::write(&target, b"target").expect("write target");
        let _ = fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).expect("create symlink");
        assert!(refuse_symlink(&link).is_err(), "symlink was accepted");
        assert!(refuse_symlink(&target).is_ok(), "regular file was refused");
        assert!(
            refuse_symlink(&directory.join("absent")).is_ok(),
            "absent path was refused"
        );
        fs::remove_dir_all(&directory).ok();
    }
}
