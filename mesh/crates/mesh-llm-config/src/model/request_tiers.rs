/// `[request_tiers]`: whose requests this node serves first.
///
/// A request is served at one of three tiers: `local` (this machine, its
/// private nodes and nodes signed by its own owner key), `trusted` (nodes
/// signed by an owner in the trust store) and `mesh` (everyone else).
/// `owners` overrides the tier for nodes signed by a given owner, and
/// `local_kv_reserve_percent` holds back a share of each model's KV budget
/// for `local` requests.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RequestTiersConfig {
    /// Owner id (64 hex characters, as `mesh-llm auth status` prints it) to
    /// the tier this node serves that owner's nodes at.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub owners: std::collections::BTreeMap<String, RequestTierName>,
    /// Percent (0-100) of each model's KV token budget that only `local`
    /// requests may reserve. 0, the default, reserves nothing.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub local_kv_reserve_percent: u8,
}

/// A request tier, as config names it.
#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RequestTierName {
    Local,
    Trusted,
    Mesh,
}

impl RequestTiersConfig {
    pub fn is_empty(&self) -> bool {
        self.owners.is_empty() && self.local_kv_reserve_percent == 0
    }

    /// The first owner key that is not an owner id, if any.
    pub fn invalid_owner(&self) -> Option<&str> {
        self.owners
            .keys()
            .map(String::as_str)
            .find(|owner| !super::is_node_id(owner))
    }

    /// An owner named twice, as keys that differ only in letter case, if
    /// any: the two entries would contradict each other.
    pub fn repeated_owner(&self) -> Option<&str> {
        let mut seen = std::collections::BTreeSet::new();
        self.owners
            .keys()
            .map(String::as_str)
            .find(|owner| !seen.insert(owner.to_ascii_lowercase()))
    }
}

fn is_zero(value: &u8) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::RequestTierName;
    use crate::{config_to_toml, parse_config_toml};

    const OWNER: &str = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0";

    #[test]
    fn overrides_and_reserve_survive_a_save_and_reload() {
        let config = parse_config_toml(&format!(
            "[request_tiers]\nlocal_kv_reserve_percent = 25\n\n[request_tiers.owners]\n\"{OWNER}\" = \"trusted\"\n"
        ))
        .unwrap();
        assert_eq!(
            config.request_tiers.owners.get(OWNER),
            Some(&RequestTierName::Trusted)
        );
        assert_eq!(config.request_tiers.local_kv_reserve_percent, 25);
        let saved = config_to_toml(&config).unwrap();
        assert_eq!(
            parse_config_toml(&saved).unwrap().request_tiers,
            config.request_tiers
        );
    }

    #[test]
    fn an_empty_section_is_not_written() {
        let saved = config_to_toml(&parse_config_toml("").unwrap()).unwrap();
        assert!(!saved.contains("request_tiers"), "{saved}");
    }

    #[test]
    fn an_owner_that_is_not_an_owner_id_is_rejected() {
        let error = parse_config_toml("[request_tiers.owners]\nstudio = \"local\"\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("request_tiers.owners"), "{error}");
    }

    #[test]
    fn an_owner_named_twice_in_different_case_is_rejected() {
        let error = parse_config_toml(&format!(
            "[request_tiers.owners]\n\"{OWNER}\" = \"mesh\"\n\"{}\" = \"local\"\n",
            OWNER.to_ascii_uppercase()
        ))
        .unwrap_err()
        .to_string();
        assert!(error.contains("more than once"), "{error}");
    }

    #[test]
    fn an_unknown_tier_is_rejected() {
        assert!(
            parse_config_toml(&format!(
                "[request_tiers.owners]\n\"{OWNER}\" = \"owner\"\n"
            ))
            .is_err()
        );
    }

    #[test]
    fn a_reserve_above_one_hundred_percent_is_rejected() {
        let error = parse_config_toml("[request_tiers]\nlocal_kv_reserve_percent = 101\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("local_kv_reserve_percent"), "{error}");
    }
}
