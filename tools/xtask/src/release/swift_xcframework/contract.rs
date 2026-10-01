use super::Error;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    HostOnly,
    Full,
}

impl Mode {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::HostOnly => "host-only",
            Self::Full => "full",
        }
    }

    pub(super) fn keys(self) -> BTreeSet<Key> {
        let keys: &[(&str, &str)] = match self {
            Self::HostOnly => &[("macos", "")],
            Self::Full => &[
                ("ios", ""),
                ("ios", "maccatalyst"),
                ("ios", "simulator"),
                ("macos", ""),
            ],
        };
        keys.iter()
            .map(|(platform, variant)| Key {
                platform: (*platform).to_owned(),
                variant: (*variant).to_owned(),
            })
            .collect()
    }

    pub(super) fn architectures(self, _key: &Key) -> BTreeSet<String> {
        let names: &[&str] = match self {
            Self::HostOnly | Self::Full => &["arm64"],
        };
        names.iter().map(|name| (*name).to_owned()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    pub(super) platform: String,
    pub(super) variant: String,
}

impl Key {
    pub(super) fn is_macos(&self) -> bool {
        self.platform == "macos" && self.variant.is_empty()
    }
}

#[derive(Debug)]
pub(super) struct Component(String);

impl Component {
    pub(super) fn parse(value: Option<&plist::Value>, field: &str) -> Result<Self, Error> {
        let value = value
            .and_then(plist::Value::as_string)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                Error::Contract(format!("XCFramework {field} must be a non-empty string"))
            })?;
        let parts = value
            .split('/')
            .filter(|part| !part.is_empty() && *part != ".")
            .collect::<Vec<_>>();
        if value.starts_with('/') || parts.len() != 1 || parts.first() == Some(&"..") {
            return Err(Error::Contract(format!(
                "XCFramework {field} must be one safe path component: {value:?}"
            )));
        }
        Ok(Self(value.to_owned()))
    }

    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}
