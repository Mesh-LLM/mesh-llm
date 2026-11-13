/// `[receipts]`: verifiable inference receipts for the models this node
/// serves (`mesh/docs/design/RECEIPTS.md`).
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct ReceiptsConfig {
    /// Issue a receipt, signed with the node key, for every turn a client
    /// asks one for, and open it when the client challenges it.
    #[serde(default)]
    pub enabled: bool,
    /// Audit about one in ten of the receipted turns peers serve this
    /// node's clients, and block a peer whose audit fails for seven days.
    #[serde(default)]
    pub audit: bool,
    /// Route this node's clients' chat completions only to runtimes that
    /// issue receipts, and audit them: the node default of the `receipts`
    /// request trust tier (`mesh/docs/design/TRUST_TIERS.md`).
    #[serde(default)]
    pub require: bool,
    /// Never audit a receipt, and so never contact Hugging Face or keep
    /// model data for an audit, whatever `audit`, `require` or a request's
    /// `x-mesh-trust` say. A peer's receipts then go unchecked, so under
    /// the `receipts` trust tier only this node's own runtime qualifies.
    #[serde(default)]
    pub offline: bool,
    /// Audit one in this many of the receipted turns peers serve this
    /// node's clients, from 1 (every turn) to [`MAX_AUDIT_ONE_IN`]. The
    /// VRF of each turn still decides which. Unset: one in
    /// [`DEFAULT_AUDIT_ONE_IN`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audit_one_in: Option<u32>,
}

/// The share of receipted turns audited when `audit_one_in` is unset.
pub const DEFAULT_AUDIT_ONE_IN: u32 = 10;
/// The rarest audit `audit_one_in` may ask for.
pub const MAX_AUDIT_ONE_IN: u32 = 10_000;

impl ReceiptsConfig {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The share of receipted turns this node audits: one in
    /// `audit_one_in`, kept in its range.
    pub fn audit_rate(&self) -> f64 {
        let one_in = self
            .audit_one_in
            .unwrap_or(DEFAULT_AUDIT_ONE_IN)
            .clamp(1, MAX_AUDIT_ONE_IN);
        1.0 / f64::from(one_in)
    }

    /// Why `audit_one_in` is out of its range, when it is.
    pub fn audit_one_in_problem(&self) -> Option<String> {
        let one_in = self.audit_one_in?;
        (!(1..=MAX_AUDIT_ONE_IN).contains(&one_in)).then(|| {
            format!("receipts.audit_one_in must be from 1 to {MAX_AUDIT_ONE_IN}, got {one_in}")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_audit_rate_is_one_in_the_setting() {
        let rate = |one_in| {
            ReceiptsConfig {
                audit_one_in: one_in,
                ..ReceiptsConfig::default()
            }
            .audit_rate()
        };
        assert_eq!(rate(None), 0.1);
        assert_eq!(rate(Some(1)), 1.0);
        assert_eq!(rate(Some(10_000)), 1.0 / 10_000.0);
    }

    #[test]
    fn an_audit_rate_out_of_range_is_an_error() {
        let config: crate::MeshConfig = toml::from_str("[receipts]\naudit_one_in = 0\n").unwrap();
        let diagnostics = crate::validate::validate_config_diagnostics(&config);
        assert!(
            diagnostics.iter().any(|d| d
                .message
                .contains("receipts.audit_one_in must be from 1 to 10000")),
            "{diagnostics:?}"
        );
        for one_in in [1, 10, 10_000] {
            let config: crate::MeshConfig =
                toml::from_str(&format!("[receipts]\naudit_one_in = {one_in}\n")).unwrap();
            assert!(config.receipts.audit_one_in_problem().is_none(), "{one_in}");
        }
        let config: crate::MeshConfig =
            toml::from_str("[receipts]\naudit_one_in = 10001\n").unwrap();
        assert!(config.receipts.audit_one_in_problem().is_some());
    }
}
