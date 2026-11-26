//! The plugin store the setup-installed service reads.
//!
//! systemd and launchd start the node with the service's own environment,
//! not the shell that ran setup: `MESH_LLM_PLUGIN_DIR` comes from
//! `service.env`, and without it the node uses the default store. Setup
//! installs the default plugin set there when it also installs the service,
//! so the node it starts finds them.
use super::SetupPlatform;
use super::service_paths::{ServiceInstallContext, ServicePaths};
use anyhow::{Context, Result, bail};
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
        Ok(contents) => plugin_dir_from_env_file(&contents, context.platform)
            .with_context(|| format!("read {PLUGIN_DIR_VAR} from {}", env_file.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("read {}", env_file.display()));
        }
    };
    // systemd (`WorkingDirectory=%h`) and launchd (`WorkingDirectory`) start
    // the node in the home directory, so a relative value names a store there,
    // wherever setup itself runs.
    Ok(Some(match configured {
        Some(path) => context.home_dir.join(path),
        None => default_plugin_store(&context.home_dir),
    }))
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
///
/// The two readers differ past that: systemd's `EnvironmentFile=` treats
/// `$` literally, while the macOS runner sources the file with bash, which
/// expands variables, command substitutions and a leading `~` outside single
/// quotes. Both process backslash escapes and join adjacent quoted parts. A
/// value whose meaning depends on any of this is an error rather than a
/// guess, so setup never installs into a store the service does not read.
fn plugin_dir_from_env_file(contents: &str, platform: SetupPlatform) -> Result<Option<PathBuf>> {
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
            value = Some(raw);
        }
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    let (inner, quoting) = unquote(raw);
    if !value_is_literal(inner, quoting, platform) {
        bail!(
            "the service reads `{PLUGIN_DIR_VAR}={raw}` in a way setup does not reproduce; \
             set it to an absolute path, plain or in single quotes, without variables, `~`, backslashes or other shell syntax"
        );
    }
    Ok(Some(inner)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quoting {
    None,
    Single,
    Double,
}

fn unquote(value: &str) -> (&str, Quoting) {
    for (quote, quoting) in [('"', Quoting::Double), ('\'', Quoting::Single)] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return (inner, quoting);
        }
    }
    (value, Quoting::None)
}

/// Whether the service reads `inner` as exactly these characters. Anything
/// that is not plainly literal for that reader is refused, not interpreted:
/// a quote inside the value (mixed quoting such as `'/a/'"$USER"`), an
/// unquoted value with characters a shell or systemd treats specially, and,
/// outside single quotes, backslashes and (for the macOS runner's bash) `$`
/// and backticks.
fn value_is_literal(inner: &str, quoting: Quoting, platform: SetupPlatform) -> bool {
    if inner.contains('\'') || inner.contains('"') {
        return false;
    }
    match quoting {
        Quoting::Single => true,
        Quoting::Double => {
            !inner.contains('\\')
                && (platform != SetupPlatform::MacOs
                    || !(inner.contains('$') || inner.contains('`')))
        }
        Quoting::None => inner
            .chars()
            .all(|c| c.is_alphanumeric() || "/._-+:@%,=".contains(c)),
    }
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

    /// The service runs in the home directory, so a relative value is a
    /// store there, not under the directory setup was started from.
    #[test]
    fn a_relative_value_names_a_store_in_the_service_home() {
        let temp = tempfile::tempdir().unwrap();
        for platform in [SetupPlatform::Linux, SetupPlatform::MacOs] {
            let context = context(platform, temp.path());
            write_env(&context, "MESH_LLM_PLUGIN_DIR=plugins\n");
            assert_eq!(
                service_plugin_store(&context).unwrap(),
                Some(temp.path().join("home/plugins")),
                "{platform:?}"
            );
        }
    }

    #[test]
    fn windows_uses_the_invoking_store() {
        let temp = tempfile::tempdir().unwrap();
        let context = context(SetupPlatform::Windows, temp.path());
        assert_eq!(service_plugin_store(&context).unwrap(), None);
    }

    #[test]
    fn env_file_values_parse_as_the_service_reads_them() {
        for platform in [SetupPlatform::Linux, SetupPlatform::MacOs] {
            let parse = |contents| plugin_dir_from_env_file(contents, platform).unwrap();
            assert_eq!(
                parse("export MESH_LLM_PLUGIN_DIR='/a b'\n"),
                Some(PathBuf::from("/a b"))
            );
            assert_eq!(
                parse("  MESH_LLM_PLUGIN_DIR=/x\nOTHER=/y\n"),
                Some(PathBuf::from("/x"))
            );
            assert_eq!(parse("MESH_LLM_PLUGIN_DIRS=/y\n"), None);
            // Single quotes keep everything literal for both readers.
            assert_eq!(
                parse("MESH_LLM_PLUGIN_DIR='$HOME/p'\n"),
                Some(PathBuf::from("$HOME/p"))
            );
            // Both readers process backslash escapes.
            assert!(plugin_dir_from_env_file("MESH_LLM_PLUGIN_DIR=/a\\ b\n", platform).is_err());
        }
    }

    /// systemd keeps `$` and `~` literally; the macOS runner's bash expands
    /// them, so setup reports the value instead of installing elsewhere.
    #[test]
    fn values_the_macos_runner_expands_are_reported() {
        for contents in [
            "MESH_LLM_PLUGIN_DIR=\"$HOME/custom-plugins\"\n",
            "MESH_LLM_PLUGIN_DIR=${HOME}/p\n",
            "MESH_LLM_PLUGIN_DIR=`pwd`/p\n",
            "MESH_LLM_PLUGIN_DIR=~/p\n",
        ] {
            let error = plugin_dir_from_env_file(contents, SetupPlatform::MacOs).unwrap_err();
            assert!(
                error.to_string().contains("absolute path"),
                "{contents}: {error}"
            );
        }
        // Mixed quoting: bash reads `/tmp/<user>/plugins`, so neither the
        // outer quotes nor the text between them is the value.
        for platform in [SetupPlatform::Linux, SetupPlatform::MacOs] {
            for contents in [
                "MESH_LLM_PLUGIN_DIR='/tmp/'\"$USER\"'/plugins'\n",
                "MESH_LLM_PLUGIN_DIR=\"/tmp/\"'x'\"/p\"\n",
                "MESH_LLM_PLUGIN_DIR=/tmp/'x'/p\n",
                "MESH_LLM_PLUGIN_DIR=/a b\n",
                "MESH_LLM_PLUGIN_DIR=/a;b\n",
            ] {
                assert!(
                    plugin_dir_from_env_file(contents, platform).is_err(),
                    "{platform:?}: {contents}"
                );
            }
        }
        assert_eq!(
            plugin_dir_from_env_file("MESH_LLM_PLUGIN_DIR=\"$HOME/p\"\n", SetupPlatform::Linux)
                .unwrap(),
            Some(PathBuf::from("$HOME/p"))
        );
        assert_eq!(
            plugin_dir_from_env_file("MESH_LLM_PLUGIN_DIR=\"~/p\"\n", SetupPlatform::MacOs)
                .unwrap(),
            Some(PathBuf::from("~/p"))
        );
    }

    #[test]
    fn an_unreadable_value_fails_the_lookup() {
        let temp = tempfile::tempdir().unwrap();
        let context = context(SetupPlatform::MacOs, temp.path());
        write_env(&context, "MESH_LLM_PLUGIN_DIR=$HOME/p\n");
        let error = format!("{:#}", service_plugin_store(&context).unwrap_err());
        assert!(error.contains("service.env"), "{error}");
    }
}
