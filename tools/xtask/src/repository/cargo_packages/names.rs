use super::Error;
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct PackageName(String);

impl TryFrom<String> for PackageName {
    type Error = Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let mut bytes = value.bytes();
        if !bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
            || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(Error::InvalidName);
        }
        Ok(Self(value))
    }
}

impl PackageName {
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "Vec<PackageName>")]
pub(crate) struct PackageList(Vec<PackageName>);

impl TryFrom<Vec<PackageName>> for PackageList {
    type Error = Error;

    fn try_from(names: Vec<PackageName>) -> Result<Self, Self::Error> {
        if names.is_empty() {
            return Err(Error::EmptyPackages);
        }
        if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
            return Err(Error::DuplicatePackage);
        }
        Ok(Self(names))
    }
}

impl PackageList {
    pub(crate) fn parse(json: &str) -> Result<Self, Error> {
        Ok(serde_json::from_str(json)?)
    }

    pub(crate) fn names(&self) -> &[PackageName] {
        &self.0
    }

    pub(crate) fn python_json_line(&self) -> String {
        let names = self
            .0
            .iter()
            .map(|name| format!("\"{}\"", name.as_str()))
            .collect::<Vec<_>>();
        format!("[{}]\n", names.join(", "))
    }
}
