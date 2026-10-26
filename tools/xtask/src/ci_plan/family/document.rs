use super::text::FamilyString;
use crate::ci_plan::document::Json as Scanned;
use num_bigint::BigInt;

#[derive(Debug)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Integer(BigInt),
    Float(f64),
    String(FamilyString),
    Array(Vec<Json>),
    Object(Vec<(FamilyString, Json)>),
}

pub(super) fn parse(text: &str) -> Result<Json, String> {
    convert(Scanned::parse(text.as_bytes()).map_err(|error| error.to_string())?)
}

fn convert(value: Scanned) -> Result<Json, String> {
    Ok(match value {
        Scanned::Null => Json::Null,
        Scanned::Bool(flag) => Json::Bool(flag),
        Scanned::Number(number) => {
            if number.is_f64() {
                Json::Float(
                    number
                        .to_string()
                        .parse::<f64>()
                        .map_err(|error| error.to_string())?,
                )
            } else {
                Json::Integer(
                    number
                        .to_string()
                        .parse::<BigInt>()
                        .map_err(|error| error.to_string())?,
                )
            }
        }
        Scanned::String(text) => Json::String(text.into()),
        Scanned::Array(items) => {
            Json::Array(items.into_iter().map(convert).collect::<Result<_, _>>()?)
        }
        Scanned::Object(entries) => Json::Object(
            entries
                .into_iter()
                .map(|(key, value)| convert(value).map(|value| (key.into(), value)))
                .collect::<Result<_, _>>()?,
        ),
    })
}

impl Json {
    pub(super) fn get(&self, key: &str) -> Option<&Self> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub(super) fn get_key(&self, key: &FamilyString) -> Option<&Self> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub(super) fn as_object(&self) -> Option<&[(FamilyString, Self)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub(super) fn as_array(&self) -> Option<&[Self]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(super) fn as_text(&self) -> Option<&FamilyString> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    pub(super) fn as_integer(&self) -> Option<&BigInt> {
        match self {
            Self::Integer(number) => Some(number),
            _ => None,
        }
    }

    pub(super) fn equals_one(&self) -> bool {
        matches!(self, Self::Integer(number) if number == &BigInt::from(1_u8))
    }
}
