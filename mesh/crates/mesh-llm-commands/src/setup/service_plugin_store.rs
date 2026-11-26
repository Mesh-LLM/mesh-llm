//! The plugin store the setup-installed service reads.
//!
//! systemd and launchd start the node with the service's own environment,
//! not the shell that ran setup: `MESH_LLM_PLUGIN_DIR` comes from
//! `service.env`, and without it the node uses the default store. Setup
//! installs the default plugin set there when it also installs the service,
//! so the node it starts finds them.
use super::SetupPlatform;
use super::service_paths::{ServiceInstallContext, ServicePaths};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const PLUGIN_DIR_VAR: &str = "MESH_LLM_PLUGIN_DIR";

/// The plugin store root of the service `context` describes. `None` on
/// Windows, whose scheduled task starts with the user's own environment, as
/// the shell that ran setup did.
pub(crate) fn service_plugin_store(context: &ServiceInstallContext) -> Result<Option<PathBuf>> {
    if context.platform == SetupPlatform::Windows {
        return Ok(None);
    }
    let env_file = ServicePaths::from_context(context).service_env_file;
    let configured = match std::fs::read_to_string(&env_file) {
        Ok(contents) => plugin_dir_from_env_file(&contents),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("read {}", env_file.display()));
        }
    };
    Ok(Some(configured.unwrap_or_else(|| {
        default_plugin_store(&context.home_dir)
    })))
}

/// What the node uses when nothing sets `MESH_LLM_PLUGIN_DIR`, as
/// `mesh_llm_plugin_manager::default_store_root` computes it.
fn default_plugin_store(home_dir: &Path) -> PathBuf {
    home_dir.join(".mesh-llm").join("plugins")
}

/// `MESH_LLM_PLUGIN_DIR` from a `service.env` of plain `KEY=value` lines,
/// the format its template asks for. The last assignment wins; an empty
/// value counts as unset, as it does for the node. Matching quotes around
/// the value are removed, as systemd and the launchd runner's shell do.
fn plugin_dir_from_env_file(contents: &str) -> Option<PathBuf> {
    let mut value = None;
    for line in contents.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, raw)) = line.split_once('=') else {
            continue;
        };
        if key == PLUGIN_DIR_VAR {
            value = Some(unquote(raw).to_string());
        }
    }
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(platform: SetupPlatform, root: &Path) -> ServiceInstallContext {
        ServiceInstallContext {
            platform,
            home_dir: root.join("home"),
            config_root: root.join("config"),
            local_data_root: root.join("local"),
            binary_path: root.join("bin/mesh-llm"),
            user_id: String::new(),
            start_service: true,
        }
    }

    fn write_env(context: &ServiceInstallContext, contents: &str) {
        let path = ServicePaths::from_context(context).service_env_file;
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn the_service_env_file_names_the_store() {
        let temp = tempfile::tempdir().unwrap();
        for platform in [SetupPlatform::Linux, SetupPlatform::MacOs] {
            let context = context(platform, temp.path());
            write_env(
                &context,
                "# MESH_LLM_PLUGIN_DIR=/commented\nRUST_LOG=info\nMESH_LLM_PLUGIN_DIR=\"/srv/plugins\"\n",
            );
            assert_eq!(
                service_plugin_store(&context).unwrap(),
                Some(PathBuf::from("/srv/plugins"))
            );
        }
    }

    #[test]
    fn without_a_setting_the_service_uses_the_default_store() {
        let temp = tempfile::tempdir().unwrap();
        let context = context(SetupPlatform::Linux, temp.path());
        let default = Some(temp.path().join("home/.mesh-llm/plugins"));
        // No service.env yet: setup writes the template later.
        assert_eq!(service_plugin_store(&context).unwrap(), default);
        // An empty value, as from a line that clears it, is unset.
        write_env(&context, "MESH_LLM_PLUGIN_DIR=/old\nMESH_LLM_PLUGIN_DIR=\n");
        assert_eq!(service_plugin_store(&context).unwrap(), default);
    }

    #[test]
    fn windows_uses_the_invoking_store() {
        let temp = tempfile::tempdir().unwrap();
        let context = context(SetupPlatform::Windows, temp.path());
        assert_eq!(service_plugin_store(&context).unwrap(), None);
    }

    #[test]
    fn env_file_values_parse_as_the_service_reads_them() {
        assert_eq!(
            plugin_dir_from_env_file("export MESH_LLM_PLUGIN_DIR='/a b'\n"),
            Some(PathBuf::from("/a b"))
        );
        assert_eq!(
            plugin_dir_from_env_file("  MESH_LLM_PLUGIN_DIR=/x\nOTHER=/y\n"),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(plugin_dir_from_env_file("MESH_LLM_PLUGIN_DIRS=/y\n"), None);
    }
}
