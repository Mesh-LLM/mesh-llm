//! Required safety admission, independent of optional exchange observers.
use super::{client_stream::ClientStream, exchange_admission::SelectedRouteTarget, transport};
use crate::inference::{election::InferenceTarget, image_safety};
use crate::mesh::{Node, ServedModelDescriptor, model_identity::descriptor_matches_routable_name};

pub(super) async fn admit(
    node: &Node,
    stream: &mut ClientStream,
    request: &transport::BufferedHttpRequest,
    model: Option<&str>,
    target: &SelectedRouteTarget<'_>,
) -> Result<(), image_safety::ImageSafetyResponseError> {
    if let Ok(body) = request.effective_http_entity()
        && image_safety::request_has_duplicate_keys(&body)
    {
        stream
            .require_image_safety(super::client_stream::ImageResponseSafetyPlan {
                guard_plugins: Vec::new(),
            })
            .map_err(|_| image_safety::ImageSafetyResponseError::Unavailable)?;
        return Err(image_safety::ImageSafetyResponseError::Unavailable);
    }
    let generation_endpoint = matches!(
        request.client_path.split('?').next(),
        Some("/v1/chat/completions" | "/v1/completions" | "/v1/responses" | "/v1/messages")
    );
    let generation_path = request.method == "POST" && generation_endpoint;
    let image_output = selected_outputs_images(node, model, target).await;
    if !image_output
        && !generation_endpoint
        && (request.is_tokenize_request()
            || request.is_anthropic_count_tokens_request()
            || super::workload_routing::required_request_workload(&request.client_path).is_some())
    {
        return Ok(());
    }
    let requesting = !stream.is_mesh_peer();
    let independent = if requesting {
        match image_safety::independent_requester_screening(node).await {
            Ok(enabled) => enabled,
            Err(error) => {
                stream
                    .require_image_safety(super::client_stream::ImageResponseSafetyPlan {
                        guard_plugins: Vec::new(),
                    })
                    .map_err(|_| image_safety::ImageSafetyResponseError::Unavailable)?;
                return Err(error);
            }
        }
    } else {
        false
    };
    if !image_output && !independent {
        return Ok(());
    }
    // Mark the response and its raw logging artifacts private before calling
    // any guard, so even admission failures cannot persist the prompt.
    stream
        .require_image_safety(super::client_stream::ImageResponseSafetyPlan {
            guard_plugins: Vec::new(),
        })
        .map_err(|_| image_safety::ImageSafetyResponseError::Unavailable)?;
    if !generation_path || model.is_none() {
        return Err(image_safety::ImageSafetyResponseError::Unavailable);
    }
    let serving = image_output
        && !matches!(
            target,
            SelectedRouteTarget::Mesh(InferenceTarget::Remote(_))
        );
    let body = request
        .effective_http_entity()
        .map_err(|_| image_safety::ImageSafetyResponseError::Unavailable)?;
    let plan = image_safety::admit(node, &body, requesting, serving).await?;
    stream
        .require_image_safety(plan)
        .map_err(|_| image_safety::ImageSafetyResponseError::Unavailable)
}

async fn selected_outputs_images(
    node: &Node,
    model: Option<&str>,
    target: &SelectedRouteTarget<'_>,
) -> bool {
    match target {
        SelectedRouteTarget::Mesh(InferenceTarget::Remote(id)) => node
            .peers()
            .await
            .iter()
            .find(|peer| peer.id == *id)
            .is_none_or(|peer| descriptors_output_images(model, &peer.served_model_descriptors)),
        SelectedRouteTarget::Mesh(InferenceTarget::Local(_)) => {
            descriptors_output_images(model, &node.served_model_descriptors().await)
        }
        SelectedRouteTarget::Mesh(InferenceTarget::None) => false,
        SelectedRouteTarget::Url(url) => {
            if descriptors_output_images(model, &node.served_model_descriptors().await) {
                return true;
            }
            let Some(manager) = node.plugin_manager().await else {
                return true;
            };
            if let Some(model) = model
                && let Ok(Some(route)) = manager.virtual_model_for_model(model).await
                && route.output_modalities.iter().any(|value| value == "image")
            {
                return true;
            }
            let Ok(endpoints) = manager.inference_endpoints().await else {
                return true;
            };
            let mut matched = false;
            for endpoint in endpoints {
                if endpoint.address == *url {
                    matched = true;
                    let Some(modalities) = manager
                        .endpoint_output_modalities_snapshot(
                            &endpoint.plugin_name,
                            &endpoint.endpoint_id,
                        )
                        .await
                    else {
                        return true;
                    };
                    if modalities.iter().any(|(name, values)| {
                        model.is_none_or(|model| name == model)
                            && values.iter().any(|value| value == "image")
                    }) {
                        return true;
                    }
                }
            }
            !matched
        }
    }
}

fn descriptors_output_images(model: Option<&str>, descriptors: &[ServedModelDescriptor]) -> bool {
    descriptors.iter().any(|descriptor| {
        model.is_none_or(|model| descriptor_matches_routable_name(descriptor, model))
            && descriptor.metadata.as_ref().is_some_and(|metadata| {
                metadata
                    .output_modalities
                    .iter()
                    .any(|value| value == "image")
            })
    })
}

#[cfg(test)]
mod tests;
