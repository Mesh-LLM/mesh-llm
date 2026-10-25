use std::collections::BTreeMap;

use mesh_llm_payments_types::{offers::LightningOffers, pricing::Pricing};

use crate::proto::node::LightningOffer;

pub(super) fn encode(offers: &LightningOffers) -> Vec<LightningOffer> {
    offers
        .iter()
        .map(|(model, price)| LightningOffer {
            model: model.clone(),
            input_msat_per_million: price.input_msat_per_million,
            output_msat_per_million: price.output_msat_per_million,
            // Deprecated: always 1 so older buyers (which reject 0) still accept.
            minimum_invoice_msat: 1,
            cache_discount_percent: u32::from(price.cache_discount_percent),
            payment_protocol: offers.protocol,
        })
        .collect()
}

/// Reject malformed advertisements instead of accidentally treating an invalid
/// paid offer as a free provider. Older readers ignore this additive field.
/// The payment protocol is kept, not checked: an offer on a protocol this node
/// cannot pay still marks the peer paid, so it is neither paid nor used free.
pub(super) fn decode(offers: &[LightningOffer]) -> Option<LightningOffers> {
    if offers.len() > 128 {
        return None;
    }
    let protocol = offers.first().map_or(0, |offer| offer.payment_protocol);
    if offers
        .iter()
        .any(|offer| offer.payment_protocol != protocol)
    {
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
            cache_discount_percent: offer.cache_discount_percent.try_into().ok()?,
        };
        price.validate().ok()?;
        if result.insert(offer.model.clone(), price).is_some() {
            return None;
        }
    }
    Some(LightningOffers {
        protocol,
        prices: result,
    })
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
            cache_discount_percent: 0,
            payment_protocol: mesh_llm_payments_types::metered_wire::PAYMENT_PROTOCOL,
        };
        let decoded = decode(std::slice::from_ref(&offer)).unwrap();
        assert_eq!(encode(&decoded), vec![offer.clone()]);
        assert!(decode(&[offer.clone(), offer.clone()]).is_none());
        // The deprecated minimum is ignored on read, even a padded one.
        let mut padded = offer.clone();
        padded.minimum_invoice_msat = 1000;
        assert_eq!(decode(&[padded]).unwrap(), decoded);
        let mut discounted = offer.clone();
        discounted.cache_discount_percent = 75;
        assert_eq!(
            encode(&decode(&[discounted.clone()]).unwrap()),
            vec![discounted]
        );
        let mut invalid_discount = offer.clone();
        invalid_discount.cache_discount_percent = 101;
        assert!(decode(&[invalid_discount]).is_none());
        // A paid offer from before versioning still decodes, as paid but
        // unpayable, so it is never mistaken for a free provider.
        let mut legacy = offer.clone();
        legacy.payment_protocol = 0;
        let legacy_decoded = decode(std::slice::from_ref(&legacy)).unwrap();
        assert_eq!(legacy_decoded.prices, decoded.prices);
        assert!(decoded.payable() && !legacy_decoded.payable());
        legacy.model = "other".into();
        assert!(
            decode(&[offer.clone(), legacy]).is_none(),
            "one node, one protocol"
        );
        let mut invalid = offer;

        invalid.input_msat_per_million = 0;
        assert!(decode(&[invalid]).is_none());
    }
}
