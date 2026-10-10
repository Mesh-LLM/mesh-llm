use super::document::Json;
use super::fields::PlanResult;
use super::projection::ToJson;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Integer(u128);

#[derive(Debug, thiserror::Error)]
#[error("{field} must be an unsigned 64-bit integer >= {minimum}")]
struct InvalidInteger<'a> {
    field: &'a str,
    minimum: u64,
}

impl Integer {
    pub(super) fn parse(value: Option<&Json>, field: &str, minimum: u64) -> PlanResult<Self> {
        let number = value
            .and_then(Json::as_integer)
            .and_then(|value| u64::try_from(value).ok());
        match number {
            Some(number) if number >= minimum => Ok(Self(u128::from(number))),
            _ => Err(InvalidInteger { field, minimum }.to_string()),
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        self.0 == 0
    }

    pub(super) fn sum(&self, other: &Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl ToJson for Integer {
    fn to_json(&self) -> Json {
        Json::Integer(self.0.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct WorkBytes(Integer);

impl WorkBytes {
    pub(super) fn parse(value: Option<&Json>, field: &str) -> PlanResult<Self> {
        Integer::parse(value, field, 1).map(Self)
    }

    pub(super) fn integer(&self) -> &Integer {
        &self.0
    }
}

impl ToJson for WorkBytes {
    fn to_json(&self) -> Json {
        self.0.to_json()
    }
}

#[cfg(test)]
mod tests {
    use super::WorkBytes;
    use crate::ci_plan::family::document;

    #[test]
    fn model_bytes_reject_input_above_u64_maximum() {
        let value = document::parse("18446744073709551616").expect("valid JSON");
        let result = WorkBytes::parse(Some(&value), "estimated_model_bytes");
        assert_eq!(
            result.expect_err("out-of-range input"),
            "estimated_model_bytes must be an unsigned 64-bit integer >= 1"
        );
    }
}
