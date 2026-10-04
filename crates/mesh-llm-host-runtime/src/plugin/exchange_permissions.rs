//! Active subscription checks distinguish deliberate removal from missing permissions.
use mesh_llm_config::OpenAiExchangeGrant;

use super::PluginManager;
use super::proto;

impl PluginManager {
    pub(in crate::plugin) async fn exchange_permissions_preflight(
        &self,
        endpoint: &str,
    ) -> super::PhaseResult {
        let mut result = super::PhaseResult::default();
        for (name, grant) in self.exchange_grant_snapshot() {
            if !grant
                .endpoints
                .iter()
                .any(|subscribed| subscribed == endpoint)
                || grant.phases.is_empty()
            {
                continue;
            }
            let manifest = match self.inner.plugins.get(&name) {
                Some(plugin) => plugin.manifest_snapshot().await,
                None => None,
            };
            let declaration = manifest
                .as_ref()
                .and_then(|manifest| manifest.openai_exchange_hook.as_deref());
            let Some(declaration) = declaration else {
                result.required_failure |=
                    grant.failure_policy == mesh_llm_config::OpenAiExchangeFailurePolicy::Required;
                result.incomplete = true;
                result.evidence_unavailable = true;
                self.inner
                    .exchange_health
                    .lock()
                    .unwrap()
                    .entry(name)
                    .or_default()
                    .permissions_unavailable = true;
                continue;
            };
            let active = declaration
                .phases
                .iter()
                .any(|phase| subscribes(Some(declaration), Some(&grant), endpoint, phase));
            if !active {
                continue;
            }
            self.refresh_exchange_permissions_status(&name, Some(declaration));
            if mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
                Some(declaration),
                Some(&grant),
            )
            .is_err()
            {
                result.required_failure = true;
                result.incomplete = true;
                result.evidence_unavailable = true;
            }
        }
        result
    }
    pub(in crate::plugin) fn refresh_exchange_permissions_status(
        &self,
        name: &str,
        declaration: Option<&proto::OpenAiExchangeHookManifest>,
    ) {
        let grant = self.effective_exchange_grant(name);
        let active = declaration
            .zip(grant.as_ref())
            .is_some_and(|(declaration, grant)| {
                declaration
                    .endpoints
                    .iter()
                    .any(|endpoint| grant.endpoints.contains(endpoint))
                    && declaration
                        .phases
                        .iter()
                        .any(|phase| grant.phases.contains(phase))
            });
        let unavailable = active
            && mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
                declaration,
                grant.as_ref(),
            )
            .is_err();
        self.inner
            .exchange_health
            .lock()
            .unwrap()
            .entry(name.into())
            .or_default()
            .permissions_unavailable = unavailable;
    }

    pub(in crate::plugin) fn granted_exchange_subscription(
        &self,
        name: &str,
        declaration: Option<&proto::OpenAiExchangeHookManifest>,
        endpoint: &str,
        phase: &str,
    ) -> anyhow::Result<Option<OpenAiExchangeGrant>> {
        let grant = self.effective_exchange_grant(name);
        if !subscribes(declaration, grant.as_ref(), endpoint, phase) {
            return Ok(None);
        }
        let negotiated = mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
            declaration,
            grant.as_ref(),
        );
        self.inner
            .exchange_health
            .lock()
            .unwrap()
            .entry(name.into())
            .or_default()
            .permissions_unavailable = negotiated.is_err();
        negotiated
    }
}

pub(crate) fn subscribes(
    declaration: Option<&proto::OpenAiExchangeHookManifest>,
    grant: Option<&OpenAiExchangeGrant>,
    endpoint: &str,
    phase: &str,
) -> bool {
    declaration.zip(grant).is_some_and(|(declaration, grant)| {
        declaration.endpoints.iter().any(|value| value == endpoint)
            && declaration.phases.iter().any(|value| value == phase)
            && grant.endpoints.iter().any(|value| value == endpoint)
            && grant.phases.iter().any(|value| value == phase)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_plugin::openai_exchange::{negotiate_openai_exchange, openai_exchange_hook};

    #[test]
    fn reduced_permissions_remain_active_but_complete_removal_disables() {
        let mut declaration = openai_exchange_hook("observe");
        declaration.required = true;
        declaration.request_body = true;
        declaration.metadata = true;
        let full = OpenAiExchangeGrant {
            endpoints: declaration.endpoints.clone(),
            phases: declaration.phases.clone(),
            request_body: true,
            metadata: true,
            deadline_ms: 100,
            max_body_bytes: 1024,
            max_queue_bytes: 1024,
            max_in_flight: 1,
            ..Default::default()
        };
        assert!(
            negotiate_openai_exchange(Some(&declaration), Some(&full))
                .unwrap()
                .is_some()
        );
        for reduced in [
            OpenAiExchangeGrant {
                request_body: false,
                ..full.clone()
            },
            OpenAiExchangeGrant {
                metadata: false,
                ..full.clone()
            },
        ] {
            assert!(subscribes(
                Some(&declaration),
                Some(&reduced),
                "chat_completions",
                "request_received"
            ));
            assert!(negotiate_openai_exchange(Some(&declaration), Some(&reduced)).is_err());
        }
        assert!(!subscribes(
            Some(&declaration),
            None,
            "chat_completions",
            "request_received"
        ));
        let removed = OpenAiExchangeGrant {
            endpoints: vec!["responses".into()],
            ..full
        };
        assert!(!subscribes(
            Some(&declaration),
            Some(&removed),
            "chat_completions",
            "request_received"
        ));
    }
}
