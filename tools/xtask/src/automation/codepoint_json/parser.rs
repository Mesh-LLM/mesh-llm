use super::value::Value;
use crate::ci_plan::document::Json;

pub(in crate::automation) fn parse(raw: &[u8]) -> Result<Value, String> {
    convert(Json::parse(raw).map_err(|error| error.to_string())?)
}

fn convert(value: Json) -> Result<Value, String> {
    Ok(match value {
        Json::Null => Value::Null,
        Json::Bool(value) => Value::Bool(value),
        Json::Number(value) => match value.as_i64() {
            Some(number) => Value::Int(i128::from(number)),
            None => match value.as_u64() {
                Some(number) => Value::Int(i128::from(number)),
                None => Value::Float(value.as_f64().ok_or("invalid JSON number")?),
            },
        },
        Json::String(value) => Value::Str(value.as_str().into()),
        Json::Array(values) => {
            Value::Array(values.into_iter().map(convert).collect::<Result<_, _>>()?)
        }
        Json::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| convert(value).map(|value| (key.as_str().into(), value)))
                .collect::<Result<_, _>>()?,
        ),
    })
}
