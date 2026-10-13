//! Bounded custody and child capture shared by the two prepared upstream SDK profiles.
use super::*;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read as _};
const FILE_LIMIT: u64 = 128 * 1048576;
const ENVIRONMENT_LIMIT: u64 = 1073741824;
const ENTRY_LIMIT: usize = 32768;
const CAPTURE_LIMIT: u64 = 1048576;
#[derive(Clone, Copy)]
pub(super) enum PythonProfile {
    Mcp312,
    Swe311,
    NoLinks,
}
impl PythonProfile {
    fn allows_link(self, path: &Path) -> bool {
        match self {
            Self::Mcp312 => matches!(
                path.to_str(),
                Some("bin/python" | "bin/python3" | "bin/python3.12")
            ),
            Self::Swe311 => matches!(
                path.to_str(),
                Some("bin/python" | "bin/python3" | "bin/python3.11")
            ),
            Self::NoLinks => false,
        }
    }
}
pub(super) fn read(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    // The same opened descriptor supplies type validation and bounded bytes.
    // A FIFO/path replacement cannot block the Unix open or redirect a validated pathname.
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        bail!("SDK needs a regular bounded file: {}", path.display());
    }
    let mut bytes = Vec::new();
    file.take(cap.checked_add(1).context("SDK read cap overflow")?)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        bail!("SDK file exceeds admission bound");
    }
    Ok(bytes)
}
pub(super) fn hash(path: &Path) -> Result<String> {
    Ok(hash_bytes(&read(path, FILE_LIMIT)?))
}
pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes).iter() {
        text.push(char::from(HEX[usize::from(*byte >> 4)]));
        text.push(char::from(HEX[usize::from(*byte & 15)]));
    }
    text
}
pub(super) fn capture_tools(uv: &Path, python: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut pins = BTreeMap::new();
    for path in [uv, python] {
        if !path.is_absolute() || !fs::symlink_metadata(path)?.is_file() {
            bail!("SDK tool needs absolute canonical regular executable");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            if fs::metadata(path)?.permissions().mode() & 0o111 == 0 {
                bail!("SDK tool not executable");
            }
        }
        pins.insert(path.into(), hash(path)?);
    }
    if pins.len() != 2 {
        bail!("SDK uv/Python tool identities must differ");
    }
    Ok(pins)
}
struct PythonLinks<'a> {
    python: &'a Path,
    profile: PythonProfile,
}
pub(super) fn environment(
    root: &Path,
    python: &Path,
    profile: PythonProfile,
) -> Result<BTreeMap<PathBuf, String>> {
    if !fs::symlink_metadata(root)?.is_dir() {
        bail!("SDK environment must be a real directory");
    }
    let mut pins = BTreeMap::new();
    let mut size = 0;
    let mut entries = 0;
    scan(
        root,
        root,
        &PythonLinks { python, profile },
        &mut pins,
        &mut size,
        &mut entries,
        0,
    )?;
    if !pins.contains_key(Path::new("pyvenv.cfg")) || !pins.contains_key(Path::new("bin/python")) {
        bail!("SDK prepared environment incomplete");
    }
    let entry = root.join("bin/python");
    if fs::symlink_metadata(&entry)?.is_file() && hash(&entry)? != hash(python)? {
        bail!("SDK copied interpreter differs from admitted tool");
    }
    Ok(pins)
}
fn scan(
    root: &Path,
    directory: &Path,
    links: &PythonLinks<'_>,
    pins: &mut BTreeMap<PathBuf, String>,
    size: &mut u64,
    entries: &mut usize,
    depth: usize,
) -> Result<()> {
    if depth > 32 {
        bail!("SDK tree depth exceeds bound");
    }
    for item in fs::read_dir(directory)? {
        *entries += 1;
        if *entries > ENTRY_LIMIT {
            bail!("SDK tree entry bound");
        }
        let path = item?.path();
        let metadata = fs::symlink_metadata(&path)?;
        let relative = path.strip_prefix(root)?.to_path_buf();
        if relative
            .to_str()
            .is_none_or(|p| p.chars().any(char::is_control))
        {
            bail!("SDK path refused");
        }
        if metadata.is_dir() {
            scan(root, &path, links, pins, size, entries, depth + 1)?;
        } else if metadata.is_file() {
            *size = size
                .checked_add(metadata.len())
                .context("SDK tree byte overflow")?;
            if *size > ENVIRONMENT_LIMIT || pins.len() >= ENTRY_LIMIT {
                bail!("SDK environment exceeds finite scope");
            }
            pins.insert(relative, format!("file:{}", hash(&path)?));
        } else if metadata.file_type().is_symlink() {
            // Only uv's three known Python links may leave this private environment.
            if !links.profile.allows_link(&relative) || path.canonicalize()? != links.python {
                bail!("SDK environment has unadmitted link");
            }
            if pins.len() >= ENTRY_LIMIT {
                bail!("SDK link count bound");
            }
            let link = fs::read_link(&path)?;
            pins.insert(
                relative,
                format!("link:{}", link.to_str().context("SDK link is not UTF-8")?),
            );
        } else {
            bail!("SDK tree special file refused");
        }
    }
    Ok(())
}

pub(super) fn package_roster(root: &Path, directory: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut pins = BTreeMap::new();
    scan(
        root,
        directory,
        &PythonLinks {
            python: Path::new("/no-admitted-links"),
            profile: PythonProfile::NoLinks,
        },
        &mut pins,
        &mut 0,
        &mut 0,
        0,
    )?;
    Ok(pins)
}
pub(super) fn capture(
    spec: &CommandSpec,
    execution: Duration,
    stdout: &Path,
    stderr: &Path,
) -> Result<CommandOutcome> {
    let mut command = spec.command();
    configure_child_group(&mut command);
    command.stdin(Stdio::null());
    command.stdout(Stdio::from(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(stdout)?,
    ));
    command.stderr(Stdio::from(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(stderr)?,
    ));
    let mut child = command.spawn().context("start bounded SDK preparation")?;
    let outcome = process_cleanup::wait_with_timeout_observed(&mut child, Some(execution), || {
        for path in [stdout, stderr] {
            if fs::metadata(path)?.len() > CAPTURE_LIMIT {
                bail!("SDK capture limit exceeded");
            }
        }
        Ok(())
    });
    // After owned cleanup retain at most the stated bytes even when a write burst exceeded polling cap.
    for path in [stdout, stderr] {
        if fs::metadata(path)?.len() > CAPTURE_LIMIT {
            fs::OpenOptions::new()
                .write(true)
                .open(path)?
                .set_len(CAPTURE_LIMIT)?;
        }
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn digest_encodes_workspace_sha_bytes_using_known_vectors() {
        assert_eq!(
            hash_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hash_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
