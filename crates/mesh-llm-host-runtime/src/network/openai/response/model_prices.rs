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
    price: Option<&mesh_llm_payments_types::pricing::Pricing>,
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
    #[tokio::test]
    async fn listing_aggregates_mixed_offers_without_changing_other_fields() -> anyhow::Result<()> {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let free = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 0 }).await?;
        let paid = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 0 }).await?;
        for (provider, charged) in [(&free, false), (&paid, true)] {
            provider.set_models(vec!["test".into()]).await;
            provider.set_hosted_models(vec!["test".into()]).await;
            provider.set_serving_models(vec!["test".into()]).await;
            let mut announcement = provider
                .build_local_announcement(provider.snapshot_local_announcement_data().await);
            if charged {
                announcement.lightning_offers.insert(
                    "test".into(),
                    mesh_llm_payments_types::pricing::Pricing {
                        input_msat_per_million: 10,
                        output_msat_per_million: 20,
                        minimum_invoice_msat: 1000,
                    },
                );
            }
            node.add_peer_after_direct_requirements_validated(
                provider.id(),
                provider.endpoint.addr(),
                &announcement,
                Some(1),
            )
            .await;
        }
        let mut body = json!({"data":[{"id":"test","sentinel":7},{"id":"unrelated"}]});
        attach_prices(&mut body, &["test".into()], &[], &node).await;
        assert_eq!(body["data"][0]["sentinel"], 7);
        assert_eq!(body["data"][0]["payment"]["free_available"], true);
        assert_eq!(body["data"][0]["payment"]["paid_available"], true);
        assert_eq!(
            body["data"][0]["payment"]["offers"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(body["data"][1].get("payment").is_none());
        assert!(node.payments.get().is_none());
        node.endpoint.close().await;
        free.endpoint.close().await;
        paid.endpoint.close().await;
        Ok(())
    }

    #[test]
    fn same_model_can_describe_free_and_paid_providers() {
        let price = mesh_llm_payments_types::pricing::Pricing {
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
