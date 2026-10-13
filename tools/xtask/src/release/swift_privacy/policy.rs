use super::PrivacyError;
use plist::{Dictionary, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

pub(super) const EXPECTED: [(&str, &[&str]); 3] = [
    (
        "NSPrivacyAccessedAPICategoryFileTimestamp",
        &["C617.1", "3B52.1"],
    ),
    ("NSPrivacyAccessedAPICategoryDiskSpace", &["E174.1"]),
    ("NSPrivacyAccessedAPICategorySystemBootTime", &["35F9.1"]),
];

pub(super) fn validate(bytes: &[u8]) -> Result<(), PrivacyError> {
    let value = Value::from_reader(Cursor::new(bytes))?;
    let manifest = value
        .as_dictionary()
        .ok_or(PrivacyError::Shape("manifest must be a dictionary"))?;
    if manifest
        .get("NSPrivacyTracking")
        .and_then(Value::as_boolean)
        != Some(false)
    {
        return Err(PrivacyError::Tracking);
    }
    if !empty_array(manifest.get("NSPrivacyCollectedDataTypes")) {
        return Err(PrivacyError::CollectedData);
    }
    if !empty_array(manifest.get("NSPrivacyTrackingDomains")) {
        return Err(PrivacyError::TrackingDomains);
    }
    let actual = categories(manifest)?;
    for (category, reasons) in EXPECTED {
        let expected: BTreeSet<&str> = reasons.iter().copied().collect();
        let found = actual.get(category).cloned().unwrap_or_default();
        if found != expected {
            return Err(PrivacyError::Reasons {
                category,
                actual: found.into_iter().map(str::to_owned).collect(),
                expected: expected.into_iter().collect(),
            });
        }
    }
    let unexpected: Vec<String> = actual
        .keys()
        .filter(|category| !EXPECTED.iter().any(|(expected, _)| **category == *expected))
        .map(|category| (*category).to_owned())
        .collect();
    if !unexpected.is_empty() {
        return Err(PrivacyError::UnexpectedCategories(unexpected));
    }
    Ok(())
}

fn empty_array(value: Option<&Value>) -> bool {
    value.and_then(Value::as_array).is_some_and(Vec::is_empty)
}

fn categories(manifest: &Dictionary) -> Result<BTreeMap<&str, BTreeSet<&str>>, PrivacyError> {
    let entries = match manifest.get("NSPrivacyAccessedAPITypes") {
        None => &[][..],
        Some(Value::Array(entries)) => entries.as_slice(),
        Some(Value::Dictionary(entries)) if entries.is_empty() => &[],
        Some(Value::String(entries)) if entries.is_empty() => &[],
        Some(Value::Data(entries)) if entries.is_empty() => &[],
        Some(_) => return Err(PrivacyError::Shape("API types must be iterable entries")),
    };
    let mut actual = BTreeMap::new();
    for (index, entry) in entries.iter().enumerate() {
        let entry = entry
            .as_dictionary()
            .ok_or(PrivacyError::InvalidEntry(index))?;
        let category = entry
            .get("NSPrivacyAccessedAPIType")
            .and_then(Value::as_string)
            .filter(|category| !category.is_empty())
            .ok_or(PrivacyError::InvalidEntry(index))?;
        let reasons = reasons(entry.get("NSPrivacyAccessedAPITypeReasons"), index)?;
        if reasons.is_empty() {
            return Err(PrivacyError::InvalidEntry(index));
        }
        if actual.insert(category, reasons).is_some() {
            return Err(PrivacyError::DuplicateCategory(category.to_owned()));
        }
    }
    Ok(actual)
}

fn reasons(value: Option<&Value>, index: usize) -> Result<BTreeSet<&str>, PrivacyError> {
    match value {
        None => Ok(BTreeSet::new()),
        Some(Value::Array(reasons)) => reasons
            .iter()
            .map(|reason| reason.as_string().ok_or(PrivacyError::InvalidEntry(index)))
            .collect(),
        Some(Value::Dictionary(reasons)) => Ok(reasons.keys().map(String::as_str).collect()),
        Some(Value::String(reasons)) if reasons.is_empty() => Ok(BTreeSet::new()),
        Some(Value::Data(reasons)) if reasons.is_empty() => Ok(BTreeSet::new()),
        Some(_) => Err(PrivacyError::InvalidEntry(index)),
    }
}
