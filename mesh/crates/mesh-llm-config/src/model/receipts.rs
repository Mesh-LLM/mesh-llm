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
}

impl ReceiptsConfig {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}
