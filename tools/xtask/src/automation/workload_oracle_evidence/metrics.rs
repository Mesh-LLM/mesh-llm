use super::Error;
#[cfg(test)]
use crate::automation::codepoint_json::parser;
use crate::automation::codepoint_json::value::Value;

pub(super) struct PcmMetrics(Value);

impl PcmMetrics {
    #[cfg(test)]
    pub(super) fn parse(bytes: &[u8]) -> Result<Self, Error> {
        Self::from_value(parser::parse(bytes).map_err(Error::Json)?)
    }

    pub(super) fn from_value(value: Value) -> Result<Self, Error> {
        if !matches!(value, Value::Object(_)) {
            return Err(Error::Metrics);
        }
        for field in ["sample_rate_hz", "channels", "sample_count"] {
            let positive = match value.get(field) {
                Some(Value::Int(integer)) => *integer > 0,
                Some(Value::BigInt(integer)) => !integer.starts_with('-'),
                Some(
                    Value::Null
                    | Value::Bool(_)
                    | Value::Float(_)
                    | Value::Str(_)
                    | Value::Array(_)
                    | Value::Object(_),
                )
                | None => false,
            };
            if !positive {
                return Err(Error::PositiveInteger(field));
            }
        }
        for (field, minimum, maximum, minimum_text, maximum_text) in [
            ("relative_rms_error", 0.0, 0.02, "0.0", "0.02"),
            ("waveform_cosine", 0.9995, 1.0, "0.9995", "1.0"),
        ] {
            let number = match value.get(field) {
                Some(Value::Int(integer)) => match *integer {
                    0 => Some(0.0),
                    1 => Some(1.0),
                    _ => None,
                },
                Some(Value::Float(float)) => Some(*float),
                Some(
                    Value::Null
                    | Value::Bool(_)
                    | Value::BigInt(_)
                    | Value::Str(_)
                    | Value::Array(_)
                    | Value::Object(_),
                )
                | None => None,
            };
            if !number
                .is_some_and(|number| number.is_finite() && (minimum..=maximum).contains(&number))
            {
                return Err(Error::MetricRange {
                    field,
                    minimum: minimum_text,
                    maximum: maximum_text,
                });
            }
        }
        Ok(Self(value))
    }

    pub(super) fn into_value(self) -> Value {
        self.0
    }
}
