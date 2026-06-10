use crate::mesh;

const UNKNOWN_CONTEXT_FALLBACK_MAX_TOKENS: u32 = 8_192;

pub(in crate::network::openai) fn context_can_satisfy(
    required_tokens: Option<u32>,
    context_length: Option<u32>,
) -> bool {
    match (required_tokens, context_length) {
        (Some(required), Some(context)) => context >= required,
        (Some(required), None) => required <= UNKNOWN_CONTEXT_FALLBACK_MAX_TOKENS,
        _ => true,
    }
}

pub(in crate::network::openai) async fn select_remote_host(
    node: &mesh::Node,
    model: &str,
    required_tokens: Option<u32>,
    hosts: Vec<iroh::EndpointId>,
) -> Option<iroh::EndpointId> {
    let Some(required_tokens) = required_tokens else {
        return hosts.into_iter().next();
    };

    let mut unknown = None;
    for host in hosts {
        match node.peer_model_context_length(host, model).await {
            Some(context) if context >= required_tokens => return Some(host),
            Some(context) => {
                tracing::info!(
                    "MoA: skipping remote worker {model} on {}; context {context} cannot fit {required_tokens} required tokens",
                    host.fmt_short()
                );
            }
            None if context_can_satisfy(Some(required_tokens), None) => {
                unknown.get_or_insert(host);
            }
            None => {
                tracing::info!(
                    "MoA: skipping remote worker {model} on {}; context is unknown and request needs {required_tokens} tokens",
                    host.fmt_short()
                );
            }
        }
    }
    unknown
}

pub(in crate::network::openai) fn virtual_mesh_context_length_from_known_contexts<I>(
    contexts: I,
) -> Option<u32>
where
    I: IntoIterator<Item = (String, u32)>,
{
    let mut by_model = std::collections::BTreeMap::<String, u32>::new();
    for (model_key, context) in contexts {
        by_model
            .entry(model_key)
            .and_modify(|existing| *existing = (*existing).max(context))
            .or_insert(context);
    }
    let mut contexts_by_model = by_model.into_values().collect::<Vec<_>>();
    contexts_by_model.sort_unstable_by(|left, right| right.cmp(left));
    contexts_by_model.get(1).copied()
}

pub(in crate::network::openai) fn should_advertise_virtual_mesh(models: &[String]) -> bool {
    models
        .iter()
        .filter(|model| model.as_str() != mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
        .take(2)
        .count()
        >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_can_satisfy_keeps_unknown_as_fallback() {
        assert!(context_can_satisfy(Some(4_096), None));
        assert!(!context_can_satisfy(Some(16_384), None));
        assert!(context_can_satisfy(None, Some(4096)));
        assert!(context_can_satisfy(Some(16_384), Some(32_768)));
        assert!(!context_can_satisfy(Some(16_384), Some(4096)));
    }

    #[test]
    fn virtual_mesh_context_is_minimum_when_only_two_known_contributors_fit() {
        let contexts = vec![("small".to_string(), 8192), ("large".to_string(), 65_536)];
        assert_eq!(
            virtual_mesh_context_length_from_known_contexts(contexts),
            Some(8192)
        );
    }

    #[test]
    fn virtual_mesh_context_uses_second_highest_known_model_context() {
        let contexts = vec![
            ("small".to_string(), 32_768),
            ("large-a".to_string(), 131_072),
            ("large-b".to_string(), 131_072),
        ];
        assert_eq!(
            virtual_mesh_context_length_from_known_contexts(contexts),
            Some(131_072)
        );
    }

    #[test]
    fn virtual_mesh_context_counts_each_model_once() {
        let contexts = vec![
            ("large".to_string(), 131_072),
            ("large".to_string(), 131_072),
            ("small".to_string(), 16_384),
        ];
        assert_eq!(
            virtual_mesh_context_length_from_known_contexts(contexts),
            Some(16_384)
        );
    }

    #[test]
    fn virtual_mesh_context_needs_two_known_contributor_contexts() {
        let contexts = vec![("known".to_string(), 32_768)];
        assert_eq!(
            virtual_mesh_context_length_from_known_contexts(contexts),
            None
        );
    }

    #[test]
    fn virtual_mesh_requires_two_concrete_models() {
        assert!(!should_advertise_virtual_mesh(&["only".to_string()]));
        assert!(should_advertise_virtual_mesh(&[
            "a".to_string(),
            "b".to_string(),
        ]));
    }
}
