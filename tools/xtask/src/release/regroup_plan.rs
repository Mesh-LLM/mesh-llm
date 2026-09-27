//! The plan side of `scripts/release-notes-regroup.py`: `validate_metadata`,
//! `plan_groups` + `validate`, and `render`. A plan is agent-authorable, so
//! every Python failure mode of these functions is kept.

use crate::ci_operations::ci_metrics_value::Value;
use crate::release::link_gh::truthy;
use crate::release::python_failure::Uncaught;
use crate::release::regroup_body::Body;
use crate::release::regroup_date::is_iso_date;
use crate::release::regroup_py::{
    contains, get, get_or_null, get_truthy, item, iterate, key, length, pr_key, quoted, text,
};
use std::collections::HashSet;

const ALLOWED_SECTIONS: [&str; 7] = [
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Other changes",
];

/// Why the script stops: an uncaught exception or a `sys.exit(message)`.
pub(crate) enum Stop {
    Raised(Uncaught),
    Exit(String),
}

impl From<Uncaught> for Stop {
    fn from(uncaught: Uncaught) -> Self {
        Self::Raised(uncaught)
    }
}

/// `re.match(^first[rest]{0,max}$)`; `$` also matches before a final `\n`.
fn plain(text: &str, max: usize, rest: &dyn Fn(char) -> bool) -> bool {
    let text = text.strip_suffix('\n').unwrap_or(text);
    let mut chars = text.chars();
    chars.next().is_some_and(|ch| ch.is_ascii_alphanumeric())
        && chars.clone().count() <= max
        && chars.all(|ch| ch.is_ascii_alphanumeric() || rest(ch))
}

fn is_title(value: &Value) -> bool {
    matches!(value, Value::Str(title) if plain(title, 79, &|ch| " ,.:&/()'-".contains(ch)))
}

/// `check_title(label, value)`.
fn check_title(problems: &mut Vec<String>, label: &str, value: &Value) {
    if !is_title(value) {
        problems.push(format!("{label} is not a plain heading: {}", quoted(value)));
    }
}

/// `plan.get(name, [])` iterated.
fn items(value: &Value, name: &str) -> Result<Vec<Value>, Uncaught> {
    get(value, name)?.map_or_else(|| Ok(Vec::new()), iterate)
}

/// `validate_metadata(plan)`.
pub(crate) fn validate_metadata(plan: &Value) -> Result<(), Stop> {
    let mut problems = Vec::new();
    let version = get_or_null(plan, "version")?;
    let version_ok = |value: &str| plain(value, 63, &|ch| ".+-".contains(ch));
    if !matches!(version, Value::Null) && !version_ok(&text(&version)) {
        problems.push(format!(
            "version is not a plain version string: {}",
            quoted(&version)
        ));
    }
    let date = get_or_null(plan, "date")?;
    if !matches!(date, Value::Null) && !is_iso_date(&text(&date)) {
        problems.push(format!(
            "date is not a real YYYY-MM-DD date: {}",
            quoted(&date)
        ));
    }
    for section in items(plan, "sections")? {
        let title = get_or_null(&section, "title")?;
        if !matches!(&title, Value::Str(name) if ALLOWED_SECTIONS.contains(&name.as_str())) {
            problems.push(format!(
                "unknown section {}; allowed: {}",
                quoted(&title),
                ALLOWED_SECTIONS.join(", ")
            ));
        }
        for group in items(&section, "groups")? {
            check_title(&mut problems, "group title", &get_or_null(&group, "title")?);
        }
    }
    let internal = get_or_null(plan, "internal")?;
    if truthy(&internal) {
        let summary = get_or_null(&internal, "summary")?;
        check_title(&mut problems, "internal summary", &summary);
        for group in items(&internal, "groups")? {
            let title = get_or_null(&group, "title")?;
            check_title(&mut problems, "internal group title", &title);
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    let lines: Vec<String> = problems.iter().map(|line| format!("  {line}")).collect();
    Err(Stop::Exit(format!(
        "error: plan metadata rejected\n{}",
        lines.join("\n")
    )))
}

/// `plan_groups(plan)`: `visit` sees each assigned pull request as the
/// generator yields it, so a failure lands where Python's would.
fn plan_groups(
    plan: &Value,
    visit: &mut dyn FnMut(&Value) -> Result<(), Uncaught>,
) -> Result<(), Uncaught> {
    let mut each = |prs: &Value| -> Result<(), Uncaught> {
        for pr in iterate(prs)? {
            visit(&pr)?;
        }
        Ok(())
    };
    for section in items(plan, "sections")? {
        item(&section, "title")?;
        if contains(&section, "prs")? {
            each(item(&section, "prs")?)?;
        }
        for group in items(&section, "groups")? {
            item(&group, "title")?;
            each(item(&group, "prs")?)?;
        }
    }
    let internal = get_or_null(plan, "internal")?;
    if truthy(&internal) {
        for group in items(&internal, "groups")? {
            item(&group, "title")?;
            each(item(&group, "prs")?)?;
        }
    }
    Ok(())
}

/// `validate(plan, entries, order)`: the count of assigned entries.
pub(crate) fn validate(plan: &Value, body: &Body) -> Result<usize, Stop> {
    let mut assigned = Vec::new();
    let mut seen = HashSet::new();
    let mut dupes = Vec::new();
    plan_groups(plan, &mut |pr| {
        let pr_key = key(pr)?;
        if !seen.insert(pr_key.clone()) {
            dupes.push(pr.clone());
        }
        assigned.push((pr.clone(), pr_key));
        Ok(())
    })?;
    let entry = |pr_key: &str| {
        body.entries
            .get(pr_key.strip_prefix("int:").unwrap_or("\0"))
    };
    let mut problems = Vec::new();
    if !dupes.is_empty() {
        problems.push("assigned to more than one section:".to_owned());
        for pr in &dupes {
            let line = entry(&key(pr)?).map_or("(unknown PR)", String::as_str);
            problems.push(format!("  #{}  {line}", text(pr)));
        }
    }
    let missing: Vec<&String> = body
        .order
        .iter()
        .filter(|pr| !seen.contains(&pr_key(pr)))
        .collect();
    if !missing.is_empty() {
        problems.push("missing from the plan:".to_owned());
        for pr in missing {
            problems.push(format!("  #{pr}  {}", body.entries[pr]));
        }
    }
    let unknown: Vec<&Value> = assigned
        .iter()
        .filter(|(_, pr_key)| entry(pr_key).is_none())
        .map(|(pr, _)| pr)
        .collect();
    if !unknown.is_empty() {
        problems.push("in the plan but not in the release body:".to_owned());
        problems.extend(unknown.into_iter().map(|pr| format!("  #{}", text(pr))));
    }
    if problems.is_empty() {
        return Ok(assigned.len());
    }
    Err(Stop::Exit(format!(
        "error: plan does not cover the body exactly\n{}",
        problems.join("\n")
    )))
}

/// `[entries[pr] for pr in prs]`; `validate` proved every lookup succeeds.
fn entry_lines(body: &Body, prs: &Value) -> Result<Vec<String>, Uncaught> {
    let mut lines = Vec::new();
    for pr in iterate(prs)? {
        let pr_key = key(&pr)?;
        let canonical = pr_key.strip_prefix("int:").unwrap_or_default();
        let line = body
            .entries
            .get(canonical)
            .ok_or_else(|| Uncaught::new("KeyError", text(&pr)))?;
        lines.push(line.clone());
    }
    Ok(lines)
}

/// `render(plan, entries, tail)`.
pub(crate) fn render(plan: &Value, body: &Body) -> Result<String, Uncaught> {
    let mut out: Vec<String> = Vec::new();
    if get_truthy(plan, "version")? {
        let mut heading = format!("## [{}]", text(item(plan, "version")?));
        if get_truthy(plan, "date")? {
            heading.push_str(&format!(" - {}", text(item(plan, "date")?)));
        }
        out.extend([heading, String::new()]);
    }
    for section in items(plan, "sections")? {
        out.extend([
            format!("### {}", text(item(&section, "title")?)),
            String::new(),
        ]);
        if contains(&section, "prs")? {
            out.extend(entry_lines(body, item(&section, "prs")?)?);
            out.push(String::new());
        }
        render_groups(&mut out, body, &items(&section, "groups")?)?;
    }
    let internal = get_or_null(plan, "internal")?;
    if truthy(&internal) {
        let groups = iterate(item(&internal, "groups")?)?;
        let mut count = 0;
        for group in &groups {
            count += length(item(group, "prs")?)?;
        }
        let summary = text(item(&internal, "summary")?);
        out.extend([
            "### Internal".to_owned(),
            String::new(),
            "<details>".to_owned(),
        ]);
        out.extend([
            format!("<summary>{summary} ({count} changes)</summary>"),
            String::new(),
        ]);
        render_groups(&mut out, body, &groups)?;
        out.extend(["</details>".to_owned(), String::new()]);
    }
    out.extend(body.tail.iter().cloned());
    let joined = out.join("\n");
    Ok(format!(
        "{}\n",
        joined.trim_end_matches(crate::repository::python_text::is_space)
    ))
}

fn render_groups(out: &mut Vec<String>, body: &Body, groups: &[Value]) -> Result<(), Uncaught> {
    for group in groups {
        out.extend([
            format!("#### {}", text(item(group, "title")?)),
            String::new(),
        ]);
        out.extend(entry_lines(body, item(group, "prs")?)?);
        out.push(String::new());
    }
    Ok(())
}
