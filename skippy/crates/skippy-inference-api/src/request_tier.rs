//! Request tiers: whom a request is served for. The tier orders admission and
//! eviction, and each tier (and the private partition) gets its own caches.

use axum::http::HeaderMap;

/// Header a trusted front end sets on the loopback hop to name the tier.
///
/// Skippy reads every value of this header and keeps the lowest one, so a
/// header can only lower a request's tier. A request without it is `local`:
/// whoever reaches the server directly is on this machine.
pub const REQUEST_TIER_HEADER: &str = "x-skippy-request-tier";

/// Header that moves a request into the private cache partition.
///
/// The only accepted value is `private`. A private request never reuses, or
/// leaves behind, cache entries visible to non-private requests of its tier.
pub const CACHE_PARTITION_HEADER: &str = "x-skippy-cache-partition";

/// The value of [`CACHE_PARTITION_HEADER`] that selects the private partition.
pub const PRIVATE_CACHE_PARTITION: &str = "private";

/// Whom a request is served for, from lowest to highest precedence.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InferenceRequestTier {
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
    pub const ALL: [Self; 3] = [Self::Mesh, Self::Trusted, Self::Local];

    /// The lowercase name used in headers, config and metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mesh => "mesh",
            Self::Trusted => "trusted",
            Self::Local => "local",
        }
    }

    /// Parse a tier name, ignoring case and surrounding whitespace.
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|tier| tier.as_str().eq_ignore_ascii_case(value))
    }

    /// Scheduling priority: a higher value is admitted first.
    pub const fn priority(self) -> u64 {
        match self {
            Self::Mesh => 0,
            Self::Trusted => 1,
            Self::Local => 2,
        }
    }

    /// Stable one-byte code for wire headers and cache identities.
    pub const fn code(self) -> u8 {
        match self {
            Self::Mesh => 0,
            Self::Trusted => 1,
            Self::Local => 2,
        }
    }

    /// Decode [`Self::code`]; unknown codes are rejected.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Mesh),
            1 => Some(Self::Trusted),
            2 => Some(Self::Local),
            _ => None,
        }
    }

    /// The lowest tier among the values of `header`, starting from `Local`.
    ///
    /// A value that is not a tier name counts as `Mesh`, so a malformed header
    /// fails low rather than being ignored.
    pub fn from_headers(headers: &HeaderMap, header: &str) -> Self {
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

/// A request's tier plus whether it uses the private cache partition.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct InferenceRequestClass {
    tier: InferenceRequestTier,
    private: bool,
}

impl InferenceRequestClass {
    pub const fn new(tier: InferenceRequestTier, private: bool) -> Self {
        Self { tier, private }
    }

    /// Read the class a trusted front end stamped on an HTTP request.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        // `private` is the only partition the header names, so any value, even
        // an unreadable one, keeps the request out of the shared partition.
        let private = headers.contains_key(CACHE_PARTITION_HEADER);
        Self::new(
            InferenceRequestTier::from_headers(headers, REQUEST_TIER_HEADER),
            private,
        )
    }

    pub const fn tier(self) -> InferenceRequestTier {
        self.tier
    }

    pub const fn is_private(self) -> bool {
        self.private
    }

    /// Lower the tier to `tier` if that is lower; never raise it.
    pub fn limited_to(self, tier: InferenceRequestTier) -> Self {
        Self::new(self.tier.min(tier), self.private)
    }

    /// The cache partition label, or `None` for the default partition
    /// (`local`, not private), which keeps cache keys unchanged for
    /// standalone Skippy.
    pub fn cache_partition(self) -> Option<String> {
        match (self.tier, self.private) {
            (InferenceRequestTier::Local, false) => None,
            (tier, false) => Some(tier.as_str().to_owned()),
            (tier, true) => Some(format!("{}-{PRIVATE_CACHE_PARTITION}", tier.as_str())),
        }
    }

    /// Partition a cache namespace by this class. The default class returns
    /// `namespace` unchanged; any other class prefixes it with
    /// `tier:<partition>:`, which no unpartitioned namespace starts with.
    pub fn cache_namespace(self, namespace: Option<String>) -> Option<String> {
        match self.cache_partition() {
            Some(partition) => Some(format!(
                "{CACHE_NAMESPACE_TIER_PREFIX}{partition}:{}",
                namespace.unwrap_or_default()
            )),
            None => namespace,
        }
    }

    /// Recover the class from a namespace built by [`Self::cache_namespace`].
    /// An unpartitioned namespace is the default class; an unrecognized
    /// partition is `mesh`, so a malformed one is evicted first.
    pub fn from_cache_namespace(namespace: Option<&str>) -> Self {
        let Some(rest) = namespace.and_then(|ns| ns.strip_prefix(CACHE_NAMESPACE_TIER_PREFIX))
        else {
            return Self::default();
        };
        let partition = rest
            .split_once(':')
            .map_or(rest, |(partition, _)| partition);
        let (tier, private) = match partition.strip_suffix(PRIVATE_CACHE_PARTITION) {
            Some(tier) => (tier.strip_suffix('-').unwrap_or(tier), true),
            None => (partition, false),
        };
        Self::new(
            InferenceRequestTier::parse(tier).unwrap_or(InferenceRequestTier::Mesh),
            private,
        )
    }
}

/// Prefix that marks a cache namespace partitioned by request class.
const CACHE_NAMESPACE_TIER_PREFIX: &str = "tier:";

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.append(*name, HeaderValue::from_static(value));
        }
        headers
    }

    #[test]
    fn missing_header_is_local() {
        assert_eq!(
            InferenceRequestClass::from_headers(&HeaderMap::new()),
            InferenceRequestClass::default()
        );
        assert_eq!(
            InferenceRequestClass::default().tier(),
            InferenceRequestTier::Local
        );
    }

    #[test]
    fn repeated_values_keep_the_lowest_tier() {
        let headers = headers(&[
            (REQUEST_TIER_HEADER, "local"),
            (REQUEST_TIER_HEADER, "Trusted, local"),
        ]);
        assert_eq!(
            InferenceRequestTier::from_headers(&headers, REQUEST_TIER_HEADER),
            InferenceRequestTier::Trusted
        );
    }

    #[test]
    fn unknown_values_fail_low() {
        let headers = headers(&[(REQUEST_TIER_HEADER, "owner")]);
        assert_eq!(
            InferenceRequestTier::from_headers(&headers, REQUEST_TIER_HEADER),
            InferenceRequestTier::Mesh
        );
    }

    #[test]
    fn limiting_never_raises() {
        let class = InferenceRequestClass::new(InferenceRequestTier::Trusted, false);
        assert_eq!(
            class.limited_to(InferenceRequestTier::Local).tier(),
            InferenceRequestTier::Trusted
        );
        assert_eq!(
            class.limited_to(InferenceRequestTier::Mesh).tier(),
            InferenceRequestTier::Mesh
        );
    }

    #[test]
    fn partitions_are_distinct_and_default_is_unpartitioned() {
        let partitions = InferenceRequestTier::ALL
            .into_iter()
            .flat_map(|tier| [false, true].map(|private| InferenceRequestClass::new(tier, private)))
            .map(InferenceRequestClass::cache_partition)
            .collect::<Vec<_>>();
        assert_eq!(partitions.iter().filter(|p| p.is_none()).count(), 1);
        let named = partitions
            .into_iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(named.len(), 5);
        assert!(
            InferenceRequestClass::from_headers(&headers(&[(CACHE_PARTITION_HEADER, "private")]))
                .is_private()
        );
        assert!(
            InferenceRequestClass::from_headers(&headers(&[(
                CACHE_PARTITION_HEADER,
                "private, private"
            )]))
            .is_private()
        );
        assert!(!InferenceRequestClass::from_headers(&HeaderMap::new()).is_private());
    }

    #[test]
    fn cache_namespaces_round_trip_every_class() {
        for tier in InferenceRequestTier::ALL {
            for private in [false, true] {
                let class = InferenceRequestClass::new(tier, private);
                for inner in [None, Some("openai:prompt_cache_key:tier:local".to_owned())] {
                    let namespace = class.cache_namespace(inner);
                    assert_eq!(
                        InferenceRequestClass::from_cache_namespace(namespace.as_deref()),
                        class
                    );
                }
            }
        }
        assert_eq!(
            InferenceRequestClass::from_cache_namespace(Some("tier:owner:x")).tier(),
            InferenceRequestTier::Mesh
        );
    }

    #[test]
    fn codes_round_trip() {
        for tier in InferenceRequestTier::ALL {
            assert_eq!(InferenceRequestTier::from_code(tier.code()), Some(tier));
        }
        assert_eq!(InferenceRequestTier::from_code(3), None);
    }
}
