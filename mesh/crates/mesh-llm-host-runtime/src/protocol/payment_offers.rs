use std::collections::BTreeMap;

use mesh_llm_payments_types::pricing::Pricing;

use crate::proto::node::LightningOffer;

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
pub(super) fn decode(offers: &[LightningOffer]) -> Option<BTreeMap<String, Pricing>> {
    if offers.len() > 128 {
        return None;
    }
    let mut result = BTreeMap::new();
    for offer in offers {
        if offer.model.is_empty() || offer.model.len() > 1024 {
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

#[cfg(test)]
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
