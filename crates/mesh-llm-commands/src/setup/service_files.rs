use super::service_templates::{render_service_env_file, render_service_runner};
use anyhow::{Result, anyhow};
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

pub(crate) fn ensure_service_env_file(service_env_file: &Path) -> Result<()> {
    if service_env_file.exists() {
        return Ok(());
    }

    let parent = service_env_file.parent().ok_or_else(|| {
        anyhow!(
            "service env file path has no parent: {}",
            service_env_file.display()
        )
    })?;
    fs::create_dir_all(parent)?;
    fs::write(service_env_file, render_service_env_file())?;
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

fn set_runner_permissions(service_runner: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let mut permissions = fs::metadata(service_runner)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(service_runner, permissions)?;
    }

    Ok(())
}
