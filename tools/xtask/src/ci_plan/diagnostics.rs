//! Planner diagnostics in the legacy `PlanError` wording, and the typed field
//! readers that produce them. Messages embed Python `repr` for names and
//! lists because callers and reviewers match the legacy text.

use crate::ci_plan::document::Json;
use crate::repository::python_text;
use std::collections::BTreeSet;
use std::fmt;

/// Raised when the plan input or checked-in CI manifests are invalid.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct PlanError(pub(super) String);

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

pub(super) type PlanResult<T> = Result<T, PlanError>;

pub(super) fn fail<T>(message: impl Into<String>) -> PlanResult<T> {
    Err(PlanError(message.into()))
}

/// Python `repr(str)`.
pub(super) fn repr(text: &str) -> String {
    python_text::repr(text)
}

/// Python `repr(list[str])`, e.g. `['a', 'b']`.
pub(super) fn repr_list<S: AsRef<str>>(items: &[S]) -> String {
    let rendered = items
        .iter()
        .map(|item| repr(item.as_ref()))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{rendered}]")
}

/// `sorted(set(selected) - known)`.
pub(super) fn sorted_unknown(selected: &[String], known: &BTreeSet<&str>) -> Vec<String> {
    selected
        .iter()
        .filter(|item| !known.contains(item.as_str()))
        .map(String::clone)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// `_nonempty_string`: a string with at least one character.
pub(super) fn nonempty_string(value: Option<&Json>, field: &str) -> PlanResult<String> {
    match value.and_then(Json::as_str) {
        Some(text) if !text.is_empty() => Ok(text.to_owned()),
        _ => fail(format!("{field} must be a non-empty string")),
    }
}

/// `_string_list`: an array of non-empty, unique strings.
pub(super) fn string_list(value: Option<&Json>, field: &str) -> PlanResult<Vec<String>> {
    let items = value
        .and_then(Json::as_array)
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_str().filter(|text| !text.is_empty()))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| PlanError(format!("{field} must be an array of non-empty strings")))?;
    let unique = items.iter().collect::<BTreeSet<_>>();
    if unique.len() != items.len() {
        return fail(format!("{field} must not contain duplicates"));
    }
    Ok(items.into_iter().map(str::to_owned).collect())
}

/// `type(value) is int and value >= 1`.
pub(super) fn positive_int(value: Option<&Json>) -> Option<i128> {
    value.and_then(Json::as_int).filter(|number| *number >= 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_plan_repr_list_matches_python() {
        assert_eq!(repr_list(&["a", "it's"]), "['a', \"it's\"]");
        assert_eq!(repr_list::<&str>(&[]), "[]");
    }

    #[test]
    fn migration_ci_plan_string_list_rejects_like_legacy() {
        let parse = |text: &str| Json::parse(text.as_bytes()).expect("valid JSON");
        let bad_type = string_list(Some(&parse("[\"a\", \"\"]")), "f");
        assert_eq!(bad_type, fail("f must be an array of non-empty strings"));
        let duplicate = string_list(Some(&parse("[\"a\", \"a\"]")), "f");
        assert_eq!(duplicate, fail("f must not contain duplicates"));
        assert_eq!(
            string_list(None, "f"),
            fail("f must be an array of non-empty strings")
        );
    }
}
