use crate::mesh;

pub(in crate::network::openai) fn context_can_satisfy(
    required_tokens: Option<u32>,
    context_length: Option<u32>,
) -> bool {
    match (required_tokens, context_length) {
        (Some(required), Some(context)) => context >= required,
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
            None => {
                unknown.get_or_insert(host);
            }
        }
    }
    unknown
}

/// Capabilities the automatic directive can serve: the union across the mesh.
///
/// A committee is text-only by construction, but the directive is not the
/// committee — a media request is served by a modality-capable model instead.
/// So what the directive can accept is what *any* served model can accept.
pub(in crate::network::openai) fn virtual_mesh_capabilities(
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> crate::models::ModelCapabilities {
    let mut union = crate::models::ModelCapabilities {
        // `moe` describes one model's architecture, not something a caller can
        // request, so it is not part of an admission union.
        moe: false,
        ..Default::default()
    };
    for model in models {
        if model == mesh_mixture_of_agents::VIRTUAL_MODEL_NAME {
            continue;
        }
        let (base_model, _) = crate::network::openai::ingress::parse_model_with_profile(model);
        let caps =
            crate::network::openai::transport::capabilities_for_model(base_model, descriptors);
        union.multimodal |= caps.multimodal;
        union.vision = union.vision.max(caps.vision);
        union.audio = union.audio.max(caps.audio);
        union.reasoning = union.reasoning.max(caps.reasoning);
        union.tool_use = union.tool_use.max(caps.tool_use);
    }
    union
}

pub(in crate::network::openai) fn virtual_mesh_context_length(
    models: &[String],
    runtimes: &[mesh::ModelRuntimeDescriptor],
) -> Option<u32> {
    let mut contexts_by_model = Vec::new();
    for model in models {
        if model == mesh_mixture_of_agents::VIRTUAL_MODEL_NAME {
            continue;
        }
        let context = runtimes
            .iter()
            .filter(|runtime| runtime.model_name == *model)
            .filter_map(mesh::ModelRuntimeDescriptor::advertised_context_length)
            .max();
        if let Some(context) = context {
            contexts_by_model.push(context);
        }
    }
    contexts_by_model.sort_unstable_by(|left, right| right.cmp(left));
    contexts_by_model.get(1).copied()
}

pub(in crate::network::openai) fn should_advertise_virtual_mesh(models: &[String]) -> bool {
    // Advertise the automatic directive whenever the node can serve anything at
    // all. It used to require two models, back when the name promised "a
    // committee will form". It now means "serve this request as well as you
    // can", which a single-model node does by serving that model — so hiding
    // the name would remove the one name clients are told to send, and break
    // any client that validates against `/v1/models`.
    models
        .iter()
        .any(|model| model.as_str() != mesh_mixture_of_agents::VIRTUAL_MODEL_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(model_name: &str, context_length: Option<u32>) -> mesh::ModelRuntimeDescriptor {
        mesh::ModelRuntimeDescriptor {
            model_name: model_name.to_string(),
            identity_hash: None,
            context_length,
            ready: true,
        }
    }

    #[test]
    fn context_can_satisfy_keeps_unknown_as_fallback() {
        assert!(context_can_satisfy(Some(16_384), None));
        assert!(context_can_satisfy(None, Some(4096)));
        assert!(context_can_satisfy(Some(16_384), Some(32_768)));
        assert!(!context_can_satisfy(Some(16_384), Some(4096)));
    }

    #[test]
    fn virtual_mesh_context_is_minimum_when_only_two_known_contributors_fit() {
        let models = vec![
            "small".to_string(),
            "large".to_string(),
            mesh_mixture_of_agents::VIRTUAL_MODEL_NAME.to_string(),
        ];
        let runtimes = vec![runtime("small", Some(8192)), runtime("large", Some(65_536))];
        assert_eq!(virtual_mesh_context_length(&models, &runtimes), Some(8192));
    }

    #[test]
    fn virtual_mesh_context_uses_second_highest_known_model_context() {
        let models = vec![
            "small".to_string(),
            "large-a".to_string(),
            "large-b".to_string(),
        ];
        let runtimes = vec![
            runtime("small", Some(32_768)),
            runtime("large-a", Some(131_072)),
            runtime("large-b", Some(131_072)),
        ];
        assert_eq!(
            virtual_mesh_context_length(&models, &runtimes),
            Some(131_072)
        );
    }

    #[test]
    fn virtual_mesh_context_counts_each_model_once() {
        let models = vec!["small".to_string(), "large".to_string()];
        let runtimes = vec![
            runtime("large", Some(131_072)),
            runtime("large", Some(131_072)),
            runtime("small", Some(16_384)),
        ];
        assert_eq!(
            virtual_mesh_context_length(&models, &runtimes),
            Some(16_384)
        );
    }

    #[test]
    fn virtual_mesh_context_needs_two_known_contributor_contexts() {
        let models = vec!["unknown".to_string(), "known".to_string()];
        let runtimes = vec![runtime("unknown", None), runtime("known", Some(32_768))];
        assert_eq!(virtual_mesh_context_length(&models, &runtimes), None);
    }

    #[test]
    fn virtual_mesh_is_advertised_whenever_any_model_is_served() {
        // The directive is the one name clients are told to send, so it must be
        // listed on a single-model node too — it previously required two models
        // and vanished from `/v1/models` below that, breaking clients that
        // validate the model list before sending.
        assert!(should_advertise_virtual_mesh(&["only".to_string()]));
        assert!(should_advertise_virtual_mesh(&[
            "a".to_string(),
            "b".to_string(),
        ]));
    }

    #[test]
    fn virtual_mesh_is_not_advertised_when_nothing_is_served() {
        // Nothing to route to, so the directive would 503. Do not offer it.
        assert!(!should_advertise_virtual_mesh(&[]));
        assert!(!should_advertise_virtual_mesh(&[
            mesh_mixture_of_agents::VIRTUAL_MODEL_NAME.to_string(),
        ]));
    }
}
