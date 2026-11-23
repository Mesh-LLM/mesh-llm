/// Controls whether iroh's DNS resolver falls back to public resolvers
/// (Cloudflare, Google, Quad9) when the system DNS resolver fails or has no
/// nameservers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DnsFallbackMode {
    Never,
    IfSystemEmpty,
    Deferred,
}

/// Network transport settings.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct NetworkConfig {
    #[serde(default)]
    pub dns_fallback: Option<DnsFallbackMode>,
    /// Set by `MESH_LLM_DNS_FALLBACK` for this process only. Never saved,
    /// so an applied config cannot write the variable into `config.toml`.
    #[serde(skip)]
    pub dns_fallback_override: Option<DnsFallbackMode>,
}

impl NetworkConfig {
    pub fn effective_dns_fallback(&self) -> Option<DnsFallbackMode> {
        self.dns_fallback_override.or(self.dns_fallback)
    }
}

#[cfg(test)]
mod tests {
    use super::DnsFallbackMode;
    use crate::{config_to_toml, parse_config_toml};

    #[test]
    fn dns_fallback_defaults_to_unset_and_parses_all_modes() {
        let parse = |toml: &str| parse_config_toml(toml).unwrap();
        assert_eq!(parse("").network.effective_dns_fallback(), None);
        assert_eq!(
            parse("[network]\ndns_fallback = \"never\"\n")
                .network
                .effective_dns_fallback(),
            Some(DnsFallbackMode::Never)
        );
        assert_eq!(
            parse("[network]\ndns_fallback = \"if-system-empty\"\n")
                .network
                .effective_dns_fallback(),
            Some(DnsFallbackMode::IfSystemEmpty)
        );
        assert_eq!(
            parse("[network]\ndns_fallback = \"deferred\"\n")
                .network
                .effective_dns_fallback(),
            Some(DnsFallbackMode::Deferred)
        );
    }

    #[test]
    fn dns_fallback_survives_a_save_and_reload() {
        let config = parse_config_toml("[network]\ndns_fallback = \"never\"\n").unwrap();
        let reloaded = parse_config_toml(&config_to_toml(&config).unwrap()).unwrap();
        assert_eq!(reloaded.network.dns_fallback, Some(DnsFallbackMode::Never));
    }

    #[test]
    fn dns_fallback_override_wins_but_is_never_saved() {
        let mut config = parse_config_toml("[network]\ndns_fallback = \"deferred\"\n").unwrap();
        config.network.dns_fallback_override = Some(DnsFallbackMode::Never);
        assert_eq!(
            config.network.effective_dns_fallback(),
            Some(DnsFallbackMode::Never)
        );

        let reloaded = parse_config_toml(&config_to_toml(&config).unwrap()).unwrap();
        assert_eq!(
            reloaded.network.dns_fallback,
            Some(DnsFallbackMode::Deferred)
        );
        assert_eq!(reloaded.network.dns_fallback_override, None);
        assert_eq!(
            reloaded.network.effective_dns_fallback(),
            Some(DnsFallbackMode::Deferred)
        );
    }
}
