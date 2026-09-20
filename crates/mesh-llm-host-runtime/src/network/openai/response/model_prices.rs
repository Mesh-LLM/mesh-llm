//! Advertised economics are descriptive, never payment authorization.
use crate::{
    mesh,
    network::openai::{request_parse::public_model_id, routing_rank::descriptor_for_model},
};
use serde_json::{Value, json};

pub(super) async fn attach_prices(
    body: &mut Value,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
    node: &mesh::Node,
) {
    let Some(items) = body.get_mut("data").and_then(Value::as_array_mut) else {
        return;
    };
    let peers = node
        .state
        .lock()
        .await
        .peers
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let local_models = node.hosted_models.lock().await.clone();
    let local_prices = node.advertised_payment_offers().await;
    for model in models {
        let (base, profile) = crate::network::openai::ingress::parse_model_with_profile(model);
        let id = public_model_id(base, descriptor_for_model(descriptors, base), profile);
        let mut offers = Vec::new();
        for peer in &peers {
            if peer.is_admitted() && peer.routes_http_model(base) {
                offers.push(offer(
                    &peer.id.to_string(),
                    peer.lightning_offers.get(base),
                    Some(peer.last_seen.elapsed().as_secs()),
                    false,
                ));
            }
        }
        if local_models.iter().any(|model| model == base)
            && let Ok(prices) = &local_prices
        {
            offers.push(offer(
                &node.id().to_string(),
                prices.get(base),
                Some(0),
                true,
            ));
        }
        if let Some(item) = items
            .iter_mut()
            .find(|item| item["id"].as_str() == Some(&id))
        {
            item["payment"] = json!({
                "binding_quote": false,
                "free_available": offers.iter().any(|offer| offer["paid"] == false),
                "paid_available": offers.iter().any(|offer| offer["paid"] == true),
                "offers": offers,
            });
        }
    }
}

fn offer(
    provider: &str,
    price: Option<&mesh_llm_payments::pricing::Pricing>,
    age: Option<u64>,
    local: bool,
) -> Value {
    json!({
        "provider_id": provider,
        "paid": price.is_some(),
        "pricing": price,
        "rate_unit": "msat_per_million_tokens",
        "peer_last_seen_seconds_ago": age,
        "local": local,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_model_can_describe_free_and_paid_providers() {
        let price = mesh_llm_payments::pricing::Pricing {
            input_msat_per_million: 10,
            output_msat_per_million: 20,
            minimum_invoice_msat: 1000,
        };
        let free = offer("free", None, Some(3), false);
        let paid = offer("paid", Some(&price), Some(1), false);
        assert_eq!(free["paid"], false);
        assert_eq!(paid["paid"], true);
        assert_eq!(paid["pricing"]["minimum_invoice_msat"], 1000);
        assert_eq!(paid["pricing"]["input_msat_per_million"], 10);
    }
}
