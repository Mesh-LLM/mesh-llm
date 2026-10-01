use crate::automation::codepoint_json::{strings::JsonString, value::Value};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::automation::replay_matrix) struct ReplayString(Vec<u32>);

impl From<&str> for ReplayString {
    fn from(text: &str) -> Self {
        Self(text.chars().map(u32::from).collect())
    }
}

impl From<&JsonString> for ReplayString {
    fn from(text: &JsonString) -> Self {
        Self(text.codepoints().collect())
    }
}

impl FromIterator<u32> for ReplayString {
    fn from_iter<T: IntoIterator<Item = u32>>(codes: T) -> Self {
        Self(codes.into_iter().collect())
    }
}

impl PartialEq<str> for ReplayString {
    fn eq(&self, other: &str) -> bool {
        self.codepoints().eq(other.chars().map(u32::from))
    }
}

impl ReplayString {
    pub(super) fn codepoints(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().copied()
    }
}

pub(super) fn repr(value: &Value) -> String {
    super::super::serialization::render(value)
        .trim_end_matches('\n')
        .to_owned()
}
