use super::service_templates::{render_service_env_file, render_service_runner};
#[cfg(unix)]
use anyhow::bail;
use anyhow::{Result, anyhow};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

// The env file is where an operator puts a private mesh token or invite, so it
// is created owner-only and an existing one is tightened. The directory is
// made owner-only only when setup creates it.
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
            restrict_service_env_file(service_env_file)?;
            file.write_all(render_service_env_file().as_bytes())?;
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            restrict_service_env_file(service_env_file)?;
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

#[cfg_attr(not(unix), allow(unused_variables))]
fn restrict_service_env_file(service_env_file: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let metadata = fs::symlink_metadata(service_env_file)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            bail!(
                "service env file must be a regular file, not a link: {}",
                service_env_file.display()
            );
        }
        // The mode passed to open() is masked by the umask; set it explicitly.
        fs::set_permissions(service_env_file, fs::Permissions::from_mode(0o600))?;
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
