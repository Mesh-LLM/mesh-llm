use super::document::Json;
use super::text::FamilyString;
use std::collections::BTreeMap;

pub(super) trait ToJson {
    fn to_json(&self) -> Json;
}

pub(super) fn object<const COUNT: usize>(entries: [(&str, Json); COUNT]) -> Json {
    Json::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect(),
    )
}

impl ToJson for str {
    fn to_json(&self) -> Json {
        Json::String(self.into())
    }
}

impl ToJson for String {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

impl ToJson for FamilyString {
    fn to_json(&self) -> Json {
        Json::String(self.clone())
    }
}

impl<T: ToJson + ?Sized> ToJson for &T {
    fn to_json(&self) -> Json {
        (*self).to_json()
    }
}

impl<T: ToJson> ToJson for Option<T> {
    fn to_json(&self) -> Json {
        self.as_ref().map_or(Json::Null, ToJson::to_json)
    }
}

impl<T: ToJson> ToJson for [T] {
    fn to_json(&self) -> Json {
        Json::Array(self.iter().map(ToJson::to_json).collect())
    }
}

impl<Key: Clone + Into<FamilyString>, Value: ToJson> ToJson for BTreeMap<Key, Value> {
    fn to_json(&self) -> Json {
        Json::Object(
            self.iter()
                .map(|(key, value)| (key.clone().into(), value.to_json()))
                .collect(),
        )
    }
}

impl<T: ToJson> ToJson for Vec<T> {
    fn to_json(&self) -> Json {
        self.as_slice().to_json()
    }
}

impl ToJson for u8 {
    fn to_json(&self) -> Json {
        Json::Integer((*self).into())
    }
}

impl ToJson for u64 {
    fn to_json(&self) -> Json {
        Json::Integer((*self).into())
    }
}

impl ToJson for usize {
    fn to_json(&self) -> Json {
        Json::Integer((*self).into())
    }
}
