/// The release checks behind the startup notice and the console's latest
/// version. Unset means on; `--auto-update` and `mesh-llm update` ignore it.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct UpdatesConfig {
    #[serde(default)]
    pub check: Option<bool>,
    /// Set by `MESH_LLM_NO_UPDATE_CHECK` for this process only. Never saved,
    /// so an applied config cannot write the variable into `config.toml`.
    #[serde(skip)]
    pub check_override: Option<bool>,
}

impl UpdatesConfig {
    pub fn check_enabled(&self) -> bool {
        self.check_override.or(self.check) != Some(false)
    }
}

#[cfg(test)]
mod tests {
    use crate::{config_to_toml, parse_config_toml};

    #[test]
    fn checks_are_on_unless_the_config_turns_them_off() {
        let parse = |toml: &str| parse_config_toml(toml).unwrap();
        assert!(parse("").updates.check_enabled());
        assert!(parse("[updates]\ncheck = true\n").updates.check_enabled());
        assert!(!parse("[updates]\ncheck = false\n").updates.check_enabled());
    }

    #[test]
    fn a_turned_off_check_survives_a_save_and_reload() {
        let config = parse_config_toml("[updates]\ncheck = false\n").unwrap();
        let reloaded = parse_config_toml(&config_to_toml(&config).unwrap()).unwrap();
        assert_eq!(reloaded.updates.check, Some(false));
    }

    #[test]
    fn the_override_wins_but_is_never_saved() {
        let mut config = parse_config_toml("[updates]\ncheck = true\n").unwrap();
        config.updates.check_override = Some(false);
        assert!(!config.updates.check_enabled());

        let reloaded = parse_config_toml(&config_to_toml(&config).unwrap()).unwrap();
        assert_eq!(reloaded.updates.check, Some(true));
        assert_eq!(reloaded.updates.check_override, None);
        assert!(reloaded.updates.check_enabled());
    }
}
