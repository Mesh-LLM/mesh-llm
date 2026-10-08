//! Host-owned active signing registrations and delegation revocation state.

use mesh_llm_identity::plugin_delegation::{
    IdentityEvidenceStatus, PluginSigningBinding, PluginSigningDelegation,
    PluginSigningDelegationClaim,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDelegationState {
    pub status: IdentityEvidenceStatus,
    pub active_binding: Option<PluginSigningBinding>,
    pub revoked_delegation_ids: Vec<String>,
}

impl Default for PluginDelegationState {
    fn default() -> Self {
        Self {
            status: IdentityEvidenceStatus::Missing,
            active_binding: None,
            revoked_delegation_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PluginIdentityRegistry {
    delegations: BTreeMap<String, Vec<PluginSigningDelegation>>,
    revoked: BTreeSet<String>,
    issuance_times: BTreeMap<String, Vec<u64>>,
    service_times: BTreeMap<String, Vec<u64>>,
    service_active: BTreeSet<String>,
}

impl PluginIdentityRegistry {
    /// Admit setup RPCs before filesystem work. One active call per plugin and
    /// thirty calls per rolling minute bound reads as well as signature requests.
    pub(crate) fn admit_service(&mut self, plugin_id: &str, now: u64) -> anyhow::Result<()> {
        if self.service_active.contains(plugin_id) {
            anyhow::bail!("identity service already active");
        }
        let times = self.service_times.entry(plugin_id.into()).or_default();
        times.retain(|issued| now.saturating_sub(*issued) < 60_000);
        if times.len() >= 30 {
            anyhow::bail!("identity service rate exceeded (thirty per minute)");
        }
        times.push(now);
        self.service_active.insert(plugin_id.into());
        Ok(())
    }
    pub(crate) fn release_service(&mut self, plugin_id: &str) {
        self.service_active.remove(plugin_id);
    }
    /// Reuse a matching certificate until its renewal window. Fresh signatures
    /// are limited independently of expiry pruning and registration rotation.
    pub(crate) fn preflight_issuance(
        &mut self,
        claim: &PluginSigningDelegationClaim,
    ) -> anyhow::Result<Option<PluginSigningDelegation>> {
        let now = claim.issued_at_unix_ms;
        let lifetime = claim.expires_at_unix_ms.saturating_sub(now);
        let entries = self
            .delegations
            .entry(claim.plugin.plugin_id.clone())
            .or_default();
        entries.retain(|entry| {
            let keep = entry.claim.expires_at_unix_ms > now;
            if !keep {
                self.revoked.remove(&entry.claim.delegation_id);
            }
            keep
        });
        let renewal_window = (lifetime / 2).min(30_000);
        if let Some(cached) = entries.iter().rev().find(|entry| {
            let previous = &entry.claim;
            previous.plugin == claim.plugin
                && previous.owner_id == claim.owner_id
                && previous.owner_sign_public_key == claim.owner_sign_public_key
                && previous.node_endpoint_id == claim.node_endpoint_id
                && previous.node_certificate_id == claim.node_certificate_id
                && previous.issued_at_unix_ms <= now
                && !self.revoked.contains(&previous.delegation_id)
                && previous.expires_at_unix_ms.saturating_sub(now) > renewal_window
                && previous.expires_at_unix_ms <= claim.expires_at_unix_ms
                && previous
                    .expires_at_unix_ms
                    .saturating_sub(previous.issued_at_unix_ms)
                    <= lifetime
        }) {
            return Ok(Some(cached.clone()));
        }
        if entries.len() >= 1024 {
            anyhow::bail!("plugin delegation registry capacity exceeded");
        }
        let times = self
            .issuance_times
            .entry(claim.plugin.plugin_id.clone())
            .or_default();
        times.retain(|issued| now.saturating_sub(*issued) < 60_000);
        if times
            .last()
            .is_some_and(|issued| now < issued.saturating_add(1000))
            || times.len() >= 5
        {
            anyhow::bail!(
                "plugin delegation issuance rate exceeded (one per second, five per minute)"
            );
        }
        Ok(None)
    }
    /// Registration change invalidates every previously issued delegation for
    /// the old key, artifact, owner or certificate. Renewal of the same binding
    /// does not revoke still-valid evidence.
    pub(crate) fn register(&mut self, delegation: PluginSigningDelegation) -> anyhow::Result<()> {
        let entries = self
            .delegations
            .entry(delegation.claim.plugin.plugin_id.clone())
            .or_default();
        entries.retain(|entry| {
            let keep = entry.claim.expires_at_unix_ms > delegation.claim.issued_at_unix_ms;
            if !keep {
                self.revoked.remove(&entry.claim.delegation_id);
            }
            keep
        });
        if entries.len() >= 1024 {
            anyhow::bail!("plugin delegation registry capacity exceeded");
        }
        for previous in entries.iter() {
            if previous.claim.plugin != delegation.claim.plugin
                || previous.claim.owner_id != delegation.claim.owner_id
                || previous.claim.node_endpoint_id != delegation.claim.node_endpoint_id
                || previous.claim.node_certificate_id != delegation.claim.node_certificate_id
            {
                self.revoked.insert(previous.claim.delegation_id.clone());
            }
        }
        self.issuance_times
            .entry(delegation.claim.plugin.plugin_id.clone())
            .or_default()
            .push(delegation.claim.issued_at_unix_ms);
        entries.push(delegation);
        Ok(())
    }

    pub(crate) fn invalidate(
        &mut self,
        plugin_id: &str,
        owner_id: Option<&str>,
        node_endpoint_id: &str,
        certificate_id: Option<&str>,
        artifact_sha256: &str,
    ) {
        if let Some(entries) = self.delegations.get(plugin_id) {
            for delegation in entries {
                if Some(delegation.claim.owner_id.as_str()) != owner_id
                    || delegation.claim.node_endpoint_id != node_endpoint_id
                    || Some(delegation.claim.node_certificate_id.as_str()) != certificate_id
                    || delegation.claim.plugin.artifact_sha256 != artifact_sha256
                {
                    self.revoked.insert(delegation.claim.delegation_id.clone());
                }
            }
        }
    }

    pub(crate) fn revoke_plugin(&mut self, plugin_id: &str) {
        if let Some(entries) = self.delegations.get(plugin_id) {
            self.revoked.extend(
                entries
                    .iter()
                    .map(|entry| entry.claim.delegation_id.clone()),
            );
        }
    }

    pub(crate) fn state(&self, plugin_id: &str, now: u64) -> PluginDelegationState {
        let entries = self.delegations.get(plugin_id);
        let mut status = match entries.and_then(|entries| entries.last()) {
            None => IdentityEvidenceStatus::Missing,
            Some(entry) if self.revoked.contains(&entry.claim.delegation_id) => {
                IdentityEvidenceStatus::Revoked
            }
            Some(entry) if entry.claim.expires_at_unix_ms <= now => IdentityEvidenceStatus::Expired,
            Some(_) => IdentityEvidenceStatus::Verified,
        };
        let active_binding = entries
            .and_then(|entries| {
                entries.iter().rev().find(|entry| {
                    entry.claim.expires_at_unix_ms > now
                        && !self.revoked.contains(&entry.claim.delegation_id)
                })
            })
            .map(|entry| entry.claim.plugin.clone());
        if active_binding.is_some() {
            status = IdentityEvidenceStatus::Verified;
        }
        let revoked_delegation_ids = entries
            .into_iter()
            .flatten()
            .filter(|entry| self.revoked.contains(&entry.claim.delegation_id))
            .map(|entry| entry.claim.delegation_id.clone())
            .collect();
        PluginDelegationState {
            status,
            active_binding,
            revoked_delegation_ids,
        }
    }
}

impl super::PluginManager {
    /// Called by host grant/key lifecycle management, never by a plugin claim.
    pub(crate) async fn revoke_plugin_delegations(&self, plugin_id: &str) {
        self.inner
            .identity_registry
            .lock()
            .await
            .revoke_plugin(plugin_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_identity::OwnerKeypair;
    use mesh_llm_identity::plugin_delegation::{
        PluginSigningDelegationClaim, PluginSigningScope, sign_plugin_delegation,
    };

    fn delegation(id: &str, key: u8, certificate: &str) -> PluginSigningDelegation {
        let owner = OwnerKeypair::from_bytes(&[1; 32], &[2; 32]).unwrap();
        let plugin = OwnerKeypair::from_bytes(&[key; 32], &[4; 32]).unwrap();
        sign_plugin_delegation(
            &owner,
            PluginSigningDelegationClaim {
                version: 1,
                delegation_id: id.into(),
                owner_id: owner.owner_id(),
                owner_sign_public_key: hex::encode(owner.verifying_key().as_bytes()),
                node_endpoint_id: hex::encode([5; 32]),
                node_certificate_id: certificate.into(),
                plugin: PluginSigningBinding {
                    plugin_id: "observer".into(),
                    artifact_sha256: hex::encode([6; 32]),
                    signing_public_key: hex::encode(plugin.verifying_key().as_bytes()),
                    scope: PluginSigningScope::OpenAiExchangeEvidence,
                },
                issued_at_unix_ms: 100,
                expires_at_unix_ms: 1000,
            },
        )
        .unwrap()
    }

    #[test]
    fn key_rotation_and_revocation_remove_active_authority() {
        let mut registry = PluginIdentityRegistry::default();
        registry
            .register(delegation("old", 3, "certificate"))
            .unwrap();
        let new = delegation("new", 7, "certificate");
        registry.register(new.clone()).unwrap();
        let state = registry.state("observer", 200);
        assert_eq!(state.revoked_delegation_ids, vec!["old"]);
        assert_eq!(state.active_binding, Some(new.claim.plugin));
        registry.revoke_plugin("observer");
        assert_eq!(registry.state("observer", 200).active_binding, None);
        assert_eq!(
            registry.state("observer", 200).revoked_delegation_ids,
            vec!["old", "new"]
        );
    }

    #[test]
    fn certificate_and_artifact_changes_invalidate_registered_keys() {
        let mut registry = PluginIdentityRegistry::default();
        let signed = delegation("first", 3, "certificate");
        registry.register(signed.clone()).unwrap();
        registry.invalidate(
            "observer",
            Some(&signed.claim.owner_id),
            &signed.claim.node_endpoint_id,
            Some("renewed"),
            &signed.claim.plugin.artifact_sha256,
        );
        assert_eq!(registry.state("observer", 200).active_binding, None);
        registry
            .register(delegation("second", 3, "renewed"))
            .unwrap();
        registry.invalidate(
            "observer",
            Some(&signed.claim.owner_id),
            &signed.claim.node_endpoint_id,
            Some("renewed"),
            &hex::encode([8; 32]),
        );
        assert_eq!(registry.state("observer", 200).active_binding, None);
        assert_eq!(
            registry.state("observer", 200).revoked_delegation_ids,
            vec!["first", "second"]
        );
    }

    #[test]
    fn expired_authority_is_not_advertised_and_restart_is_unverified() {
        let mut registry = PluginIdentityRegistry::default();
        registry
            .register(delegation("first", 3, "certificate"))
            .unwrap();
        assert_eq!(registry.state("observer", 1000).active_binding, None);
        assert_eq!(
            PluginIdentityRegistry::default()
                .state("observer", 200)
                .active_binding,
            None
        );
    }

    fn timed(id: &str, key: u8, issued: u64, expires: u64) -> PluginSigningDelegation {
        let mut claim = delegation(id, key, "certificate").claim;
        claim.issued_at_unix_ms = issued;
        claim.expires_at_unix_ms = expires;
        sign_plugin_delegation(
            &OwnerKeypair::from_bytes(&[1; 32], &[2; 32]).unwrap(),
            claim,
        )
        .unwrap()
    }
    #[test]
    fn setup_certificate_reuses_binding_until_renewal_window() {
        let mut registry = PluginIdentityRegistry::default();
        let first = timed("first", 3, 1000, 61000);
        registry.register(first.clone()).unwrap();
        let request = timed("candidate", 3, 2000, 62000).claim;
        assert_eq!(
            registry
                .preflight_issuance(&request)
                .unwrap()
                .unwrap()
                .claim
                .delegation_id,
            "first"
        );
        let shorter = timed("candidate", 3, 2000, 12000).claim;
        assert!(
            registry.preflight_issuance(&shorter).unwrap().is_none(),
            "cache exceeded requested lifetime"
        );
        let renewal = timed("candidate", 3, 40000, 100000).claim;
        assert!(registry.preflight_issuance(&renewal).unwrap().is_none());
        registry.revoke_plugin("observer");
        assert!(registry.preflight_issuance(&request).unwrap().is_none());
    }
    #[test]
    fn rotation_and_expiry_cannot_bypass_owner_signature_rate_limit() {
        let mut registry = PluginIdentityRegistry::default();
        registry.register(timed("first", 3, 1000, 2000)).unwrap();
        assert!(
            registry
                .preflight_issuance(&timed("rotation", 7, 1500, 2500).claim)
                .unwrap_err()
                .to_string()
                .contains("rate")
        );
        for index in 1..5 {
            let issued = 1000 + index * 1000;
            let entry = timed(&format!("entry-{index}"), 3, issued, issued + 1000);
            assert!(registry.preflight_issuance(&entry.claim).unwrap().is_none());
            registry.register(entry).unwrap();
        }
        assert!(
            registry
                .preflight_issuance(&timed("sixth", 3, 6000, 7000).claim)
                .unwrap_err()
                .to_string()
                .contains("rate")
        );
        assert!(
            registry
                .preflight_issuance(&timed("later", 3, 61000, 62000).claim)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn capacity_is_checked_before_owner_signature() {
        let mut registry = PluginIdentityRegistry::default();
        let entry = timed("first", 3, 1000, 61000);
        registry
            .delegations
            .insert("observer".into(), vec![entry; 1024]);
        assert!(
            registry
                .preflight_issuance(&timed("rotation", 7, 3000, 63000).claim)
                .unwrap_err()
                .to_string()
                .contains("capacity")
        );
    }
    #[test]
    fn filesystem_work_admission_bounds_concurrency_and_read_rate() {
        let mut registry = PluginIdentityRegistry::default();
        registry.admit_service("observer", 1000).unwrap();
        assert!(
            registry
                .admit_service("observer", 1001)
                .unwrap_err()
                .to_string()
                .contains("active")
        );
        registry.release_service("observer");
        for now in 1001..1030 {
            registry.admit_service("observer", now).unwrap();
            registry.release_service("observer");
        }
        assert!(
            registry
                .admit_service("observer", 1030)
                .unwrap_err()
                .to_string()
                .contains("rate")
        );
        registry.admit_service("healthy", 1030).unwrap();
        registry.release_service("healthy");
        registry.admit_service("observer", 61000).unwrap();
    }
}
