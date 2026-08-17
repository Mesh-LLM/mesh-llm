use anyhow::Result;
use mesh_llm_cli::{Cli, ConfigCommand};

pub(crate) fn dispatch_config_command(cli: &Cli, command: &ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Validate { config_path, json } => {
            mesh_llm_commands::config::run_config_validate(cli, config_path.as_deref(), *json)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tempfile::TempDir;

    const VALID_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_valid.toml"
    ));
    const INVALID_FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../mesh-llm-host-runtime/tests/fixtures/schema_driven_controls_invalid.toml"
    ));

    fn write_fixture_file(raw: &str) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().expect("fixture tempdir");
        let path = dir.path().join("config.toml");
        std::fs::write(&path, raw).expect("write fixture config");
        (dir, path)
    }

    #[test]
    fn config_validate_dispatch_accepts_schema_driven_valid_fixture() {
        let (_dir, path) = write_fixture_file(VALID_FIXTURE);
        let cli = Cli::parse_from(["mesh-llm", "config", "validate"]);
        let command = ConfigCommand::Validate {
            config_path: Some(path),
            json: true,
        };

        dispatch_config_command(&cli, &command).expect("dispatch should accept valid fixture");
    }

    #[test]
    fn config_validate_dispatch_rejects_schema_driven_invalid_fixture() {
        let (_dir, path) = write_fixture_file(INVALID_FIXTURE);
        let cli = Cli::parse_from(["mesh-llm", "config", "validate"]);
        let command = ConfigCommand::Validate {
            config_path: Some(path),
            json: true,
        };

        let error = dispatch_config_command(&cli, &command)
            .expect_err("dispatch should surface config validation failure");
        assert!(error.to_string().contains("config validation failed"));
    }
}
