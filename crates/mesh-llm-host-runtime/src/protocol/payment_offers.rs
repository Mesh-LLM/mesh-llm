#[cfg(feature = "payments")]
use std::collections::BTreeMap;

#[cfg(feature = "payments")]
use mesh_llm_payments::pricing::Pricing;

use crate::proto::node::LightningOffer;

const MAX_OFFERS: usize = 128;
const MAX_MODEL_LEN: usize = 1024;

#[cfg(feature = "payments")]
pub(super) fn encode(prices: &BTreeMap<String, Pricing>) -> Vec<LightningOffer> {
    prices
        .iter()
        .map(|(model, price)| LightningOffer {
            model: model.clone(),
            input_msat_per_million: price.input_msat_per_million,
            output_msat_per_million: price.output_msat_per_million,
            minimum_invoice_msat: price.minimum_invoice_msat,
        })
        .collect()
}

/// Reject malformed advertisements instead of accidentally treating an invalid
/// paid offer as a free provider. Older readers ignore this additive field.
#[cfg(feature = "payments")]
pub(super) fn decode(offers: &[LightningOffer]) -> Option<BTreeMap<String, Pricing>> {
    if offers.len() > MAX_OFFERS {
        return None;
    }
    let mut result = BTreeMap::new();
    for offer in offers {
        if offer.model.is_empty() || offer.model.len() > MAX_MODEL_LEN {
            return None;
        }
        let price = Pricing {
            input_msat_per_million: offer.input_msat_per_million,
            output_msat_per_million: offer.output_msat_per_million,
            minimum_invoice_msat: offer.minimum_invoice_msat,
        };
        price.validate().ok()?;
        if result.insert(offer.model.clone(), price).is_some() {
            return None;
        }
    }
    Some(result)
}

/// A build without the payments feature keeps only which models a peer
/// charges for, so routing can skip them. Structural checks match `decode`;
/// price fields are not interpreted, so any listed model counts as paid.
#[cfg(not(feature = "payments"))]
pub(super) fn decode_priced_models(
    offers: &[LightningOffer],
) -> Option<std::collections::BTreeSet<String>> {
    if offers.len() > MAX_OFFERS {
        return None;
    }
    let mut result = std::collections::BTreeSet::new();
    for offer in offers {
        if offer.model.is_empty() || offer.model.len() > MAX_MODEL_LEN {
            return None;
        }
        if !result.insert(offer.model.clone()) {
            return None;
        }
    }
    Some(result)
}

#[cfg(all(test, feature = "payments"))]
mod tests {
    use super::*;
    #[test]
    fn payments_old_announcements_are_free_and_invalid_offers_are_not() {
        assert!(decode(&[]).unwrap().is_empty());
        let offer = LightningOffer {
            model: "model".into(),
            input_msat_per_million: 500,
            output_msat_per_million: 1500,
            minimum_invoice_msat: 1,
        };
        let decoded = decode(std::slice::from_ref(&offer)).unwrap();
        assert_eq!(encode(&decoded), vec![offer.clone()]);
        assert!(decode(&[offer.clone(), offer.clone()]).is_none());
        let mut invalid = offer;
        invalid.minimum_invoice_msat = 0;
        assert!(decode(&[invalid]).is_none());
    }
}

#[cfg(all(test, not(feature = "payments")))]
mod priced_model_tests {
    use super::*;

    fn offer(model: &str) -> LightningOffer {
        LightningOffer {
            model: model.into(),
            input_msat_per_million: 500,
            output_msat_per_million: 1500,
            minimum_invoice_msat: 1,
        }
    }

    #[test]
    fn priced_models_are_kept_without_payments_and_malformed_offers_rejected() {
        assert!(decode_priced_models(&[]).unwrap().is_empty());
        let models = decode_priced_models(&[offer("a"), offer("b")]).unwrap();
        assert_eq!(models.into_iter().collect::<Vec<_>>(), vec!["a", "b"]);
        assert!(decode_priced_models(&[offer("a"), offer("a")]).is_none());
        assert!(decode_priced_models(&[offer("")]).is_none());
    }
}
