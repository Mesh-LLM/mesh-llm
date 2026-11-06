//! Compare token metadata before choosing a warm alias or loading durable bytes.
use super::*;

pub(super) struct RestoreCandidate<'a> {
    pub(super) identity: &'a PrefillKvIdentity,
    pub(super) location: Option<&'a skippy_cache::L3Location>,
}

pub(super) struct DurableRestoreLookup<'a> {
    pub(super) started: Instant,
    pub(super) location: &'a skippy_cache::L3Location,
}

/// Durable metadata resolved before entering a scheduled runtime operation.
/// A missing location is a completed lookup, not permission to probe again.
pub(crate) struct PreparedExactStateRestore {
    identities: Vec<PrefillKvIdentity>,
    locations: Vec<Option<skippy_cache::L3Location>>,
}

impl PreparedExactStateRestore {
    /// The exact identity order paired with these completed durable probes.
    pub(crate) fn identities(&self) -> &[PrefillKvIdentity] {
        &self.identities
    }
}

impl KvStageIntegration {
    /// Locate durable candidates without holding the scheduler's runtime lock.
    /// Resident eligibility is deliberately checked later, during restore.
    pub(crate) fn prepare_exact_state_restore(
        &self,
        identities: &[PrefillKvIdentity],
        shared_durable_fallback: bool,
    ) -> PreparedExactStateRestore {
        let locations = identities
            .iter()
            .map(|identity| {
                if !self.should_lookup() || self.exact_state_payload().is_none() {
                    return None;
                }
                let l3 = self.l3.as_ref()?;
                let tokens = if shared_durable_fallback {
                    self.durable_exact_lookup_token_ids(&identity.token_ids)
                } else {
                    &identity.token_ids
                };
                // Corrupt/unavailable storage remains a cache miss. The
                // prepared None must never trigger a retry under runtime lock.
                l3.locate_longest(&identity.namespace, tokens, MAX_PREFIX_PROBES)
                    .ok()
                    .flatten()
            })
            .collect();
        PreparedExactStateRestore {
            identities: identities.to_vec(),
            locations,
        }
    }

    pub(super) fn ordered_restore_candidates<'a>(
        &self,
        prepared: &'a PreparedExactStateRestore,
    ) -> Vec<RestoreCandidate<'a>> {
        if !self.records_resident_prefixes() {
            return prepared
                .identities
                .iter()
                .zip(&prepared.locations)
                .map(|(identity, location)| RestoreCandidate {
                    identity,
                    location: location.as_ref(),
                })
                .collect();
        }
        let resident = prepared
            .identities
            .iter()
            .filter_map(|identity| self.probe_resident_prefix(identity))
            .map(|hit| hit.token_count as u64)
            .max()
            .unwrap_or(0);
        let mut durable = prepared
            .identities
            .iter()
            .zip(&prepared.locations)
            .filter_map(|(identity, location)| {
                let location = location.as_ref()?;
                preflight_l3_kv_location(location).ok()?;
                (location.token_count > resident).then_some((
                    location.token_count,
                    RestoreCandidate {
                        identity,
                        location: Some(location),
                    },
                ))
            })
            .collect::<Vec<_>>();
        // Never load a candidate shorter than the native fallback. Ties stay
        // in shared memory. Actual fill still owns descriptor/cost admission.
        durable.sort_by_key(|(count, _)| std::cmp::Reverse(*count));
        durable
            .into_iter()
            .map(|(_, candidate)| candidate)
            .collect()
    }
}

#[cfg(test)]
#[path = "resident_preference/tests.rs"]
mod tests;
