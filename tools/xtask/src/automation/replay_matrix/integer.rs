use num_bigint::BigUint;
use std::fmt;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PositiveInteger(BigUint);

impl PositiveInteger {
    pub(super) fn from_int(value: i128) -> Option<Self> {
        let value = u128::try_from(value).ok()?;
        (value > 0).then(|| Self(BigUint::from(value)))
    }

    pub(super) fn from_decimal(text: &str) -> Option<Self> {
        let value = BigUint::parse_bytes(text.as_bytes(), 10)?;
        (value.bits() > 0).then_some(Self(value))
    }

    pub(super) fn product(&self, other: &Self) -> Self {
        Self(&self.0 * &other.0)
    }

    pub(super) fn below(&self, minimum: u32) -> bool {
        self.0 < BigUint::from(minimum)
    }
}

impl fmt::Display for PositiveInteger {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
