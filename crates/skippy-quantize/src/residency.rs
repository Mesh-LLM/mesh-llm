use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};

const COPY_CHUNK_BYTES: usize = 16 * 1024 * 1024;

pub fn remove_dir_if_exists(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path).with_context(|| format!("remove {}", path.display()))?;
    }
    Ok(())
}

pub fn copy_file_bounded_with_label(label: &str, source: &Path, target: &Path) -> Result<()> {
    let mut input = fs::File::open(source).with_context(|| format!("open {}", source.display()))?;
    let mut output =
        fs::File::create(target).with_context(|| format!("create {}", target.display()))?;
    let mut buffer = vec![0_u8; COPY_CHUNK_BYTES];
    let mut copied = 0_u64;

    loop {
        let count = input
            .read(&mut buffer)
            .with_context(|| format!("read {}", source.display()))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .with_context(|| format!("write {}", target.display()))?;
        drop_file_cache_range(&input, copied, count as u64);
        copied += count as u64;
    }

    output
        .sync_all()
        .with_context(|| format!("sync {}", target.display()))?;
    let target_dropped = drop_file_cache_range(&output, 0, copied);
    println!(
        "{label}_done source={} target={} size_bytes={copied} target_cache_dropped={target_dropped}",
        source.display(),
        target.display()
    );
    Ok(())
}

#[cfg(unix)]
pub fn symlink_file(source: &Path, target: &Path) -> Result<()> {
    std::os::unix::fs::symlink(source, target)
        .with_context(|| format!("symlink {} -> {}", target.display(), source.display()))
}

#[cfg(not(unix))]
pub fn symlink_file(source: &Path, target: &Path) -> Result<()> {
    fs::copy(source, target)
        .with_context(|| format!("copy {} -> {}", source.display(), target.display()))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn drop_file_cache_range(file: &fs::File, offset: u64, len: u64) -> bool {
    use std::os::fd::AsRawFd;

    if len == 0 {
        return false;
    }

    let rc = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
    rc == 0
}

#[cfg(not(target_os = "linux"))]
fn drop_file_cache_range(_file: &fs::File, _offset: u64, _len: u64) -> bool {
    false
}
