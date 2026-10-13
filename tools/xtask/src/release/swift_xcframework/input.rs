use super::{
    Error,
    contract::{Key, Mode},
};
use plist::{Dictionary, Value};
use std::{collections::BTreeSet, path::Path};

pub(super) struct Document(Value);
pub(super) struct Entry<'a> {
    pub(super) key: Key,
    fields: &'a Dictionary,
}

impl Document {
    pub(super) fn read(root: &Path) -> Result<Self, Error> {
        let info = root.join("Info.plist");
        if !root.is_dir() || !info.is_file() {
            return Err(Error::Contract(format!(
                "XCFramework or Info.plist is missing: {}",
                root.display()
            )));
        }
        let file = std::fs::File::open(&info).map_err(|error| Error::io(&info, error))?;
        Ok(Self(Value::from_reader(file)?))
    }

    pub(super) fn entries(&self, mode: Option<Mode>) -> Result<Vec<Entry<'_>>, Error> {
        let libraries = self
            .0
            .as_dictionary()
            .and_then(|info| info.get("AvailableLibraries"))
            .and_then(Value::as_array)
            .filter(|libraries| !libraries.is_empty())
            .ok_or_else(|| {
                Error::Contract("XCFramework AvailableLibraries must be a non-empty array".into())
            })?;
        let mut keys = BTreeSet::new();
        let mut entries = Vec::new();
        for library in libraries {
            let fields = library.as_dictionary().ok_or_else(|| {
                Error::Contract(format!("invalid XCFramework library entry: {library:?}"))
            })?;
            let platform = fields
                .get("SupportedPlatform")
                .and_then(Value::as_string)
                .filter(|platform| !platform.is_empty())
                .ok_or_else(|| {
                    Error::Contract(format!(
                        "invalid SupportedPlatform in XCFramework entry: {library:?}"
                    ))
                })?;
            let variant = match fields.get("SupportedPlatformVariant") {
                None => "",
                Some(value) => value.as_string().ok_or_else(|| {
                    Error::Contract(format!(
                        "invalid SupportedPlatformVariant in XCFramework entry: {library:?}"
                    ))
                })?,
            };
            let key = Key {
                platform: platform.to_owned(),
                variant: variant.to_owned(),
            };
            if !keys.insert(key.clone()) {
                return Err(Error::Contract(format!(
                    "XCFramework contains a duplicate platform slice: {key:?}"
                )));
            }
            entries.push(Entry { key, fields });
        }
        if !keys.iter().any(Key::is_macos) {
            return Err(Error::Contract(
                "XCFramework does not contain a macOS framework slice".into(),
            ));
        }
        if let Some(mode) = mode
            && keys != mode.keys()
        {
            return Err(Error::Contract(format!(
                "{} Swift SDK input has an unexpected platform matrix: {keys:?}; expected {:?}",
                mode.name(),
                mode.keys()
            )));
        }
        Ok(entries)
    }
}

impl Entry<'_> {
    pub(super) fn architectures(&self, mode: Option<Mode>) -> Result<BTreeSet<String>, Error> {
        let values = self
            .fields
            .get("SupportedArchitectures")
            .and_then(Value::as_array)
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                Error::Contract(format!(
                    "XCFramework slice {:?} must declare SupportedArchitectures",
                    self.key
                ))
            })?;
        let declared: BTreeSet<String> = values
            .iter()
            .map(|value| {
                value
                    .as_string()
                    .filter(|name| !name.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        Error::Contract(format!(
                            "XCFramework slice {:?} has invalid SupportedArchitectures",
                            self.key
                        ))
                    })
            })
            .collect::<Result<_, _>>()?;
        if declared.len() != values.len() {
            return Err(Error::Contract(format!(
                "XCFramework slice {:?} declares duplicate architectures",
                self.key
            )));
        }
        if let Some(mode) = mode
            && declared != mode.architectures(&self.key)
        {
            return Err(Error::Contract(format!(
                "{} Swift SDK slice {:?} has an unexpected architecture contract: {declared:?}; expected {:?}",
                mode.name(),
                self.key,
                mode.architectures(&self.key)
            )));
        }
        Ok(declared)
    }

    pub(super) fn location(
        &self,
    ) -> Result<(super::contract::Component, super::contract::Component), Error> {
        use super::contract::Component;
        let identifier =
            Component::parse(self.fields.get("LibraryIdentifier"), "LibraryIdentifier")?;
        let library = Component::parse(self.fields.get("LibraryPath"), "LibraryPath")?;
        if !library.as_str().ends_with(".framework") {
            return Err(Error::Contract(format!(
                "XCFramework LibraryPath must name a framework: {:?}",
                library.as_str()
            )));
        }
        Ok((identifier, library))
    }
}
