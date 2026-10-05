use super::service_templates::{render_service_env_file, render_service_runner};
use anyhow::{Result, anyhow};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

// The env file is where an operator puts a private mesh token or invite, so it
// is created owner-only and an existing one is tightened. Only the immediate
// parent is restricted to owner-only, and only when it did not already exist: a
// pre-existing parent, and any intermediate directory created by
// `create_dir_all`, keep their modes. The 0600 env file is what keeps the
// secret unreadable; the restricted directory only keeps it out of listings.
pub(crate) fn ensure_service_env_file(service_env_file: &Path) -> Result<()> {
    let parent = service_env_file.parent().ok_or_else(|| {
        anyhow!(
            "service env file path has no parent: {}",
            service_env_file.display()
        )
    })?;
    let parent_existed = parent.exists();
    fs::create_dir_all(parent)?;
    if !parent_existed {
        restrict_service_dir(parent)?;
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    match options.open(service_env_file) {
        Ok(mut file) => {
            restrict_open_service_env_file(&file)?;
            file.write_all(render_service_env_file().as_bytes())?;
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            restrict_existing_service_env_file(service_env_file)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[cfg_attr(not(unix), allow(unused_variables))]
fn restrict_service_dir(service_config_dir: &Path) -> Result<()> {
    #[cfg(unix)]
    fs::set_permissions(service_config_dir, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

// The mode passed to open() is masked by the umask; set it explicitly, through
// the descriptor that was just created.
#[cfg_attr(not(unix), allow(unused_variables))]
fn restrict_open_service_env_file(file: &fs::File) -> Result<()> {
    #[cfg(unix)]
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    Ok(())
}

// Reopen an existing env file without following a link (O_NOFOLLOW) and
// without blocking on a FIFO, check what was opened, and set the mode through
// that descriptor, so a path swapped after a check cannot redirect the chmod.
#[cfg_attr(not(unix), allow(unused_variables))]
fn restrict_existing_service_env_file(service_env_file: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let not_regular = || {
            anyhow!(
                "service env file must be a regular file, not a link or a special file: {}",
                service_env_file.display()
            )
        };
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(service_env_file)
            .map_err(|error| {
                if error.raw_os_error() == Some(libc::ELOOP) {
                    not_regular()
                } else {
                    error.into()
                }
            })?;
        if !file.metadata()?.is_file() {
            return Err(not_regular());
        }
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub(crate) fn write_service_runner(
    service_runner: &Path,
    binary_path: &Path,
    env_file: &Path,
) -> Result<()> {
    let parent = service_runner.parent().ok_or_else(|| {
        anyhow!(
            "service runner path has no parent: {}",
            service_runner.display()
        )
    })?;
    fs::create_dir_all(parent)?;
    fs::write(service_runner, render_service_runner(binary_path, env_file))?;
    set_runner_permissions(service_runner)?;
    Ok(())
}

pub(crate) fn shell_quote(path: &Path) -> String {
    let escaped = path
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('$', "$$")
        .replace('%', "%%");
    format!("\"{escaped}\"")
}

// Only Unix has an execute bit to set on the runner.
#[cfg_attr(not(unix), allow(unused_variables))]
fn set_runner_permissions(service_runner: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(service_runner)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(service_runner, permissions)?;
    }

    Ok(())
}
