//! The request tier and class types.
//!
//! These mirror the types Skippy gains in Mesh-LLM/skippy#113. The Skippy
//! release this crate builds against has none, so the node classifies
//! requests here and Skippy does not act on the class: it neither orders
//! admission or eviction by tier, nor reserves KV for `local`, nor keeps a
//! cache per class. Once a Skippy release carries the types, use Skippy's
//! and pass the class through again.

use axum::http::HeaderMap;

/// Header this node names a request's tier in on the loopback hop to its own
/// Skippy server, for the exchange hooks that server calls back into.
pub(crate) const SKIPPY_REQUEST_TIER_HEADER: &str = "x-skippy-request-tier";

/// Header that marks a request private on the same hop.
pub(crate) const CACHE_PARTITION_HEADER: &str = "x-skippy-cache-partition";

/// The value of [`CACHE_PARTITION_HEADER`] that marks a request private.
pub(crate) const PRIVATE_CACHE_PARTITION: &str = "private";

/// Whom a request is served for, from lowest to highest precedence.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum InferenceRequestTier {
    /// Anyone else on the mesh.
    Mesh,
    /// A node whose owner this node's operator trusts.
    Trusted,
    /// This machine, or a node of this node's own owner.
    #[default]
    Local,
}

impl InferenceRequestTier {
    /// Every tier, from lowest to highest.
    pub(crate) const ALL: [Self; 3] = [Self::Mesh, Self::Trusted, Self::Local];

    /// The lowercase name used in headers, config and metrics.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::Trusted => "trusted",
            Self::Local => "local",
        }
    }

    /// Parse a tier name, ignoring case and surrounding whitespace.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|tier| tier.as_str().eq_ignore_ascii_case(value))
    }

    /// The lowest tier among the values of `header`, starting from `Local`.
    ///
    /// A value that is not a tier name counts as `Mesh`, so a malformed header
    /// fails low rather than being ignored.
    pub(crate) fn from_headers(headers: &HeaderMap, header: &str) -> Self {
        headers
            .get_all(header)
            .iter()
            .flat_map(|value| {
                value
                    .to_str()
                    .map(|value| value.split(',').map(Self::parse).collect::<Vec<_>>())
                    .unwrap_or_else(|_| vec![None])
            })
            .map(|tier| tier.unwrap_or(Self::Mesh))
            .fold(Self::Local, Self::min)
    }
}

impl std::fmt::Display for InferenceRequestTier {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
