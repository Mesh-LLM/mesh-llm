use super::document::Json;
use super::text::FamilyString;
use std::collections::BTreeSet;

pub(super) type PlanResult<T> = Result<T, String>;

pub(super) fn object<'a>(value: Option<&'a Json>, field: &str) -> PlanResult<&'a Json> {
    value
        .filter(|item| item.as_object().is_some())
        .ok_or_else(|| format!("{field} must be an object"))
}

pub(super) fn exact(value: &Json, allowed: &[&str], field: &str) -> PlanResult<()> {
    let unknown = value
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(key, _)| !allowed.iter().any(|allowed| key == *allowed))
        .map(|(key, _)| key)
        .collect::<BTreeSet<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{field} contains unknown fields: {}",
            unknown
                .into_iter()
                .map(FamilyString::diagnostic)
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

pub(super) fn string(value: Option<&Json>, field: &str) -> PlanResult<FamilyString> {
    value
        .and_then(Json::as_text)
        .filter(|text| {
            !text.codes().is_empty()
                && !text
                    .codes()
                    .iter()
                    .any(|code| [0x0d, 0x0a, 0x09, 0x7c].contains(code))
        })
        .cloned()
        .ok_or_else(|| format!("{field} must be a non-empty single-line string"))
}

pub(super) fn strings(value: Option<&Json>, field: &str) -> PlanResult<Vec<FamilyString>> {
    let items = value
        .and_then(Json::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    let result = items
        .iter()
        .enumerate()
        .map(|(index, item)| string(Some(item), &format!("{field}[{index}]")))
        .collect::<PlanResult<Vec<_>>>()?;
    if result.iter().collect::<BTreeSet<_>>().len() != result.len() {
        return Err(format!("{field} must not contain duplicates"));
    }
    Ok(result)
}

pub(super) fn number(
    value: Option<&Json>,
    field: &str,
    range: std::ops::RangeInclusive<u64>,
) -> PlanResult<u64> {
    let bounds = format!("between {} and {}", range.start(), range.end());
    let error = || format!("{field} must be an integer {bounds}");
    let number = value
        .and_then(Json::as_integer)
        .and_then(|number| u64::try_from(number).ok())
        .ok_or_else(error)?;
    if !range.contains(&number) {
        return Err(error());
    }
    Ok(number)
}

pub(super) fn choice(value: Option<&Json>, field: &str, allowed: &[&str]) -> PlanResult<String> {
    let text = string(value, field)?;
    if let Some(allowed) = allowed.iter().find(|allowed| &text == **allowed) {
        Ok((*allowed).to_owned())
    } else {
        Err(format!("{field} must be one of: {}", allowed.join(", ")))
    }
}

pub(super) fn label(value: Option<&Json>, field: &str) -> PlanResult<String> {
    let text = string(value, field)?;
    if let Some(scalar) = text.scalar_text().filter(|text| valid_label(text)) {
        Ok(scalar)
    } else {
        Err(format!(
            "{field} has an invalid label: {}",
            text.diagnostic()
        ))
    }
}

pub(super) fn valid_label(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        && chars.all(|ch| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '_' | '-')
        })
}

pub(super) fn hex_sha(value: &str, min: usize, max: usize) -> bool {
    (min..=max).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
