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
            // Deprecated: always 1 so older buyers (which reject 0) still accept.
            minimum_invoice_msat: 1,
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
        // Keep a legacy (v0.78.x) seller's advertised invoice quantum so the
        // buyer's request frame and charges match what that seller enforces.
        // Current sellers always advertise 1; an absent field reads as 0.
        let price = Pricing {
            minimum_invoice_msat: offer.minimum_invoice_msat.max(1),
            ..Pricing::exact(offer.input_msat_per_million, offer.output_msat_per_million)
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
        // A legacy seller's minimum is kept on read so requests echo it,
        // but this node never re-advertises anything but 1.
        let mut padded = offer.clone();
        padded.minimum_invoice_msat = 1000;
        let legacy = decode(std::slice::from_ref(&padded)).unwrap();
        assert_eq!(legacy["model"].minimum_invoice_msat, 1000);
        assert_eq!(encode(&legacy), vec![offer.clone()]);
        let mut absent = offer.clone();
        absent.minimum_invoice_msat = 0;
        assert_eq!(decode(&[absent]).unwrap(), decoded);
        let mut invalid = offer;
        invalid.input_msat_per_million = 0;
        assert!(decode(&[invalid]).is_none());
    }
}
