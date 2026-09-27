//! `classify`, `subgroups` and `build_plan` of
//! `scripts/release-notes-classify.py`: the Conventional Commits type,
//! scope and trailers of a commit choose its Keep a Changelog section.
//! Commit records are JSON-shaped [`Value`]s because `--links` supplies
//! them verbatim; their Python access errors are reproduced.

use crate::ci_operations::ci_metrics_value::Value;
use crate::release::classify_subject::{
    HYGIENE, internal_scope, parse_subject, title, type_section,
};
use crate::release::link_commits::pr_suffix;
use crate::release::link_gh::{truthy, type_name};
use crate::release::python_failure::Uncaught;
use crate::repository::python_text::{repr, strip};
use std::cmp::Ordering;
use std::collections::HashMap;

const SECTION_ORDER: [&str; 7] = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Other changes",
];
const OTHER: &str = "Other changes";
const INTERNAL_GROUPS: [(&str, &[&str]); 4] = [
    ("CI and release engineering", &["ci"]),
    ("Build and dependencies", &["build", "deps", "chore"]),
    ("Tests", &["test"]),
    (HYGIENE, &["refactor", "style", "docs"]),
];
const INTERNAL_SUMMARY: &str =
    "CI, build, test, and repository work with no user-facing behavior change";
const SUBGROUP_THRESHOLD: usize = 20;
const MIN_GROUP: usize = 4;

/// `commit[key]` on a JSON-shaped record.
fn index<'a>(record: &'a Value, key: &str) -> Result<&'a Value, Uncaught> {
    let message = match record {
        Value::Object(_) => {
            return record
                .get(key)
                .ok_or_else(|| Uncaught::new("KeyError", repr(key)));
        }
        Value::Str(_) => "string indices must be integers, not 'str'".to_owned(),
        Value::Array(_) => "list indices must be integers or slices, not str".to_owned(),
        other => format!("'{}' object is not subscriptable", type_name(other)),
    };
    Err(Uncaught::new("TypeError", message))
}

/// `TRAILING_PR_RE.sub("", commit["subject"])`.
fn authored(commit: &Value) -> Result<&str, Uncaught> {
    match index(commit, "subject")? {
        Value::Str(text) => Ok(pr_suffix(text).map_or(text.as_str(), |(at, _)| &text[..at])),
        other => Err(Uncaught::new(
            "TypeError",
            format!(
                "expected string or bytes-like object, got '{}'",
                type_name(other)
            ),
        )),
    }
}

/// `scope_of(commit)` and `type_of(commit)`.
fn scope_of(commit: &Value) -> Result<Option<String>, Uncaught> {
    Ok(parse_subject(authored(commit)?).and_then(|found| found.scope.map(str::to_owned)))
}

fn type_of(commit: &Value) -> Result<Option<String>, Uncaught> {
    Ok(parse_subject(authored(commit)?).map(|found| found.kind.to_owned()))
}

type Placement = Option<(String, Option<String>)>;

/// `classify(commit)`: the section and scope, or `None` when unknown.
pub(crate) fn classify(commit: Option<&Value>) -> Result<Placement, Uncaught> {
    let Some(commit) = commit.filter(|record| !matches!(record, Value::Null)) else {
        return Ok(None);
    };
    let trailers = index(commit, "trailers")?;
    let Value::Object(keys) = trailers else {
        let kind = type_name(trailers);
        let message = format!("'{kind}' object has no attribute 'get'");
        return Err(Uncaught::new("AttributeError", message));
    };
    if let Some(value) = trailers.get("release-notes").filter(|value| truthy(value)) {
        let Value::Str(text) = value else {
            let message = format!("'{}' object has no attribute 'strip'", type_name(value));
            return Err(Uncaught::new("AttributeError", message));
        };
        let wanted = title(strip(text));
        let known = SECTION_ORDER.contains(&wanted.as_str()) || wanted == "Internal";
        return Ok(if known {
            Some((wanted, scope_of(commit)?))
        } else {
            None
        });
    }
    let Some(found) = parse_subject(authored(commit)?) else {
        return Ok(None);
    };
    let Some(section) = type_section(found.kind) else {
        return Ok(None);
    };
    let has = |key: &str| keys.iter().any(|(name, _)| name == key);
    let section = match () {
        () if has("deprecated") => "Deprecated",
        () if has("removed") => "Removed",
        () if has("security") => "Security",
        () if found.breaking || has("breaking change") => "Changed",
        () if found.scope.and_then(internal_scope).is_some() => "Internal",
        () => section,
    };
    Ok(Some((section.to_owned(), found.scope.map(str::to_owned))))
}

/// Pull request numbers compare as integers: canonical decimal text.
fn numeric(left: &String, right: &String) -> Ordering {
    let negative = |text: &String| text.starts_with('-');
    let magnitude = |text: &String| (text.len(), text.clone());
    match (negative(left), negative(right)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => magnitude(right).cmp(&magnitude(left)),
        (false, false) => magnitude(left).cmp(&magnitude(right)),
    }
}

fn prs_value(prs: &[String]) -> Value {
    Value::Array(prs.iter().cloned().map(Value::BigInt).collect())
}

fn group(title: &str, prs: &[String]) -> Value {
    Value::Object(vec![
        ("title".to_owned(), Value::text(title)),
        ("prs".to_owned(), prs_value(prs)),
    ])
}

/// `subgroups(by_scope)`: named scopes of at least `MIN_GROUP` entries,
/// then a trailing `Other`; `None` below two named headings.
fn subgroups(by_scope: &[(String, Vec<String>)]) -> Option<Vec<Value>> {
    let mut ordered: Vec<&(String, Vec<String>)> = by_scope.iter().collect();
    ordered.sort_by(|(a, x), (b, y)| y.len().cmp(&x.len()).then_with(|| a.cmp(b)));
    let (mut named, mut leftover) = (Vec::new(), Vec::new());
    for (scope, prs) in ordered {
        if scope != "General" && prs.len() >= MIN_GROUP {
            named.push(group(scope, prs));
        } else {
            leftover.extend(prs.iter().cloned());
        }
    }
    if named.len() < 2 {
        return None;
    }
    if !leftover.is_empty() {
        leftover.sort_by(numeric);
        named.push(group("Other", &leftover));
    }
    Some(named)
}

fn push<K: PartialEq + Clone, V>(map: &mut Vec<(K, Vec<V>)>, key: &K, item: V) {
    match map.iter_mut().find(|(seen, _)| seen == key) {
        Some((_, items)) => items.push(item),
        None => map.push((key.clone(), vec![item])),
    }
}

type Scoped = Vec<(String, Vec<(String, Vec<String>)>)>;

/// `build_plan(prs, commits, version, date)` and its unclassified count.
pub(crate) fn build_plan(
    prs: &[String],
    commits: &HashMap<String, Value>,
    version: &str,
    date: &str,
) -> Result<(Value, usize), Uncaught> {
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    let mut scopes: Scoped = Vec::new();
    let mut internal: Vec<(String, Vec<String>)> = Vec::new();
    let mut unclassified = 0;
    for pr in prs {
        let commit = commits.get(pr);
        let Some((section, scope)) = classify(commit)? else {
            push(&mut sections, &OTHER.to_owned(), pr.clone());
            unclassified += 1;
            continue;
        };
        if section == "Internal" {
            let kind = match commit {
                Some(record) => type_of(record)?,
                None => None,
            };
            let by_type = INTERNAL_GROUPS
                .iter()
                .find(|(_, kinds)| kind.as_deref().is_some_and(|k| kinds.contains(&k)))
                .map_or(HYGIENE, |(name, _)| name);
            let name = scope.as_deref().and_then(internal_scope).unwrap_or(by_type);
            push(&mut internal, &name.to_owned(), pr.clone());
            continue;
        }
        push(&mut sections, &section, pr.clone());
        let scope = scope.unwrap_or_else(|| "General".to_owned());
        match scopes.iter_mut().find(|(seen, _)| *seen == section) {
            Some((_, by_scope)) => push(by_scope, &scope, pr.clone()),
            None => scopes.push((section, vec![(scope, vec![pr.clone()])])),
        }
    }
    let mut rendered = Vec::new();
    for title in SECTION_ORDER {
        let Some((_, members)) = sections.iter().find(|(name, _)| name == title) else {
            continue;
        };
        let by_scope = scopes.iter().find(|(name, _)| name == title);
        let groups = (members.len() > SUBGROUP_THRESHOLD && title != OTHER)
            .then(|| by_scope.and_then(|(_, by_scope)| subgroups(by_scope)))
            .flatten();
        let body = match groups {
            Some(groups) => ("groups".to_owned(), Value::Array(groups)),
            None => ("prs".to_owned(), prs_value(members)),
        };
        rendered.push(Value::Object(vec![
            ("title".to_owned(), Value::text(title)),
            body,
        ]));
    }
    let mut plan = vec![
        ("version".to_owned(), Value::text(version)),
        ("date".to_owned(), Value::text(date)),
        ("sections".to_owned(), Value::Array(rendered)),
    ];
    if !internal.is_empty() {
        let groups = INTERNAL_GROUPS
            .iter()
            .filter_map(|(name, _)| internal.iter().find(|(seen, _)| seen == name))
            .map(|(name, prs)| group(name, prs))
            .collect();
        plan.push((
            "internal".to_owned(),
            Value::Object(vec![
                ("summary".to_owned(), Value::text(INTERNAL_SUMMARY)),
                ("groups".to_owned(), Value::Array(groups)),
            ]),
        ));
    }
    Ok((Value::Object(plan), unclassified))
}
