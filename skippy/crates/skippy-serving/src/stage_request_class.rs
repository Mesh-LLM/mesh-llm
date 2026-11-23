//! A request's tier and cache partition on the binary stage wire.
//!
//! Stage 0 stamps the class of each request on the cache-bearing messages it
//! sends downstream. A downstream stage lowers that claim to the highest tier
//! its host assigned to the stage and evicts by the result. It partitions its
//! caches by both, so the claims it merges into one tier still never share a
//! prefix: every stage keeps the same isolation as stage 0.

use skippy_inference_api::{InferenceRequestClass, InferenceRequestTier};
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageStateHeader;

/// Stamp `class` on an outgoing stage message header.
pub(crate) fn stamp_request_class(state: &mut StageStateHeader, class: InferenceRequestClass) {
    state.set_request_tier_demotion(tier_demotion(class.tier()));
    state.set_private_cache_partition(class.is_private());
}

/// The class a stage serves a message at: the stamped claim, lowered to the
/// stage's configured maximum tier.
pub(crate) fn stage_request_class(
    config: &StageConfig,
    state: StageStateHeader,
) -> InferenceRequestClass {
    claimed_request_class(state).limited_to(max_request_tier(config))
}

/// The cache namespace a stage records and looks up a message's prefixes in:
/// the served class's partition, which sets eviction order, and within it the
/// claimed one when the stage's cap lowered it.
pub(crate) fn stage_cache_namespace(
    config: &StageConfig,
    state: StageStateHeader,
) -> Option<String> {
    let served = stage_request_class(config, state);
    let claimed = claimed_request_class(state);
    if served == claimed {
        return served.cache_namespace(None);
    }
    let claimed_partition = claimed
        .cache_partition()
        .unwrap_or_else(|| InferenceRequestTier::Local.as_str().to_owned());
    served.cache_namespace(Some(format!("claimed-{claimed_partition}")))
}

fn claimed_request_class(state: StageStateHeader) -> InferenceRequestClass {
    InferenceRequestClass::new(
        tier_from_demotion(state.request_tier_demotion()),
        state.private_cache_partition(),
    )
}

/// The highest tier this stage serves. An unrecognized name is `mesh`, so a
/// malformed cap fails low.
fn max_request_tier(config: &StageConfig) -> InferenceRequestTier {
    config
        .max_request_tier
        .as_deref()
        .map_or(InferenceRequestTier::Local, |tier| {
            InferenceRequestTier::parse(tier).unwrap_or(InferenceRequestTier::Mesh)
        })
}

fn tier_demotion(tier: InferenceRequestTier) -> u8 {
    InferenceRequestTier::Local.code() - tier.code()
}

fn tier_from_demotion(demotion: u8) -> InferenceRequestTier {
    InferenceRequestTier::Local
        .code()
        .checked_sub(demotion)
        .and_then(InferenceRequestTier::from_code)
        .unwrap_or(InferenceRequestTier::Mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::WireMessageKind;

    fn header(class: InferenceRequestClass) -> StageStateHeader {
        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        stamp_request_class(&mut state, class);
        state
    }

    #[test]
    fn unstamped_messages_are_local_and_unpartitioned() {
        let state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
        let config = StageConfig::default();
        assert_eq!(
            stage_request_class(&config, state),
            InferenceRequestClass::default()
        );
        assert_eq!(stage_cache_namespace(&config, state), None);
    }

    #[test]
    fn every_class_round_trips_through_the_header() {
        let config = StageConfig::default();
        for tier in InferenceRequestTier::ALL {
            for private in [false, true] {
                let class = InferenceRequestClass::new(tier, private);
                assert_eq!(stage_request_class(&config, header(class)), class);
            }
        }
    }

    #[test]
    fn the_stage_cap_lowers_but_never_raises_a_claim() {
        let config = StageConfig {
            max_request_tier: Some("trusted".into()),
            ..StageConfig::default()
        };
        let local = InferenceRequestClass::new(InferenceRequestTier::Local, true);
        let mesh = InferenceRequestClass::new(InferenceRequestTier::Mesh, false);
        assert_eq!(
            stage_request_class(&config, header(local)),
            InferenceRequestClass::new(InferenceRequestTier::Trusted, true)
        );
        assert_eq!(stage_request_class(&config, header(mesh)), mesh);
        let malformed = StageConfig {
            max_request_tier: Some("owner".into()),
            ..StageConfig::default()
        };
        assert_eq!(
            stage_request_class(&malformed, header(local)).tier(),
            InferenceRequestTier::Mesh
        );
    }

    #[test]
    fn claims_a_cap_merges_keep_separate_partitions_and_evict_at_the_cap() {
        let config = StageConfig {
            max_request_tier: Some("mesh".into()),
            ..StageConfig::default()
        };
        let namespaces = [
            InferenceRequestClass::new(InferenceRequestTier::Local, false),
            InferenceRequestClass::new(InferenceRequestTier::Trusted, false),
            InferenceRequestClass::new(InferenceRequestTier::Mesh, false),
            InferenceRequestClass::new(InferenceRequestTier::Local, true),
            InferenceRequestClass::new(InferenceRequestTier::Mesh, true),
        ]
        .map(|class| stage_cache_namespace(&config, header(class)));
        let distinct = namespaces
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert_eq!(distinct, namespaces.len(), "{namespaces:?}");
        for namespace in &namespaces {
            assert_eq!(
                InferenceRequestClass::from_cache_namespace(namespace.as_deref()).tier(),
                InferenceRequestTier::Mesh,
                "{namespace:?}"
            );
        }
        // A claim the cap leaves alone keeps the plain partition.
        assert_eq!(namespaces[2].as_deref(), Some("tier:mesh:"));
    }
}
