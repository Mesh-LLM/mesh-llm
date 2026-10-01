#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::automation) struct JsonString(String);

impl PartialEq<str> for JsonString {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl JsonString {
    pub(in crate::automation) fn codepoints(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.chars().map(u32::from)
    }
}

impl From<&str> for JsonString {
    fn from(text: &str) -> Self {
        Self(text.to_owned())
    }
}
