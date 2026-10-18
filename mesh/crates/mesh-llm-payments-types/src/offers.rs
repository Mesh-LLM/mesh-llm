//! A peer's advertised paid offers and the payment protocol they are sold under.

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

use crate::{metered_wire::PAYMENT_PROTOCOL, pricing::Pricing};

/// Prices a peer advertises per model, and the payment protocol it accepts
/// payment over. Dereferences to the price map; protocol `0` means the peer
/// advertised prices without a version (a release before metered v3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LightningOffers {
    pub protocol: u32,
    pub prices: BTreeMap<String, Pricing>,
}

impl LightningOffers {
    /// Whether this node can pay for these offers. A peer advertising no
    /// prices is free, never unpayable.
    pub fn payable(&self) -> bool {
        self.prices.is_empty() || self.protocol == PAYMENT_PROTOCOL
    }
}

impl Default for LightningOffers {
    fn default() -> Self {
        BTreeMap::new().into()
    }
}

/// This node's own prices, sold under the payment protocol it speaks.
impl From<BTreeMap<String, Pricing>> for LightningOffers {
    fn from(prices: BTreeMap<String, Pricing>) -> Self {
        Self {
            protocol: PAYMENT_PROTOCOL,
            prices,
        }
    }
}

impl Deref for LightningOffers {
    type Target = BTreeMap<String, Pricing>;
    fn deref(&self) -> &Self::Target {
        &self.prices
    }
}

impl DerefMut for LightningOffers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.prices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_current_protocol_paid_offers_are_payable() {
        let mut offers = LightningOffers::default();
        assert!(offers.payable(), "free peers are always usable");
        offers.insert(
            "m".into(),
            Pricing {
                input_msat_per_million: 500,
                output_msat_per_million: 1500,
                cache_discount_percent: 0,
            },
        );
        assert!(offers.payable());
        offers.protocol = 0;
        assert!(!offers.payable(), "unversioned paid sellers predate v3");
        offers.protocol = PAYMENT_PROTOCOL + 1;
        assert!(!offers.payable());
        offers.clear();
        assert!(offers.payable());
    }
}
