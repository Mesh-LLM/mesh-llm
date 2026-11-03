//! Deliberately bounded CODEOWNERS grammar with GitHub last-match precedence.
//! A canonical terminal block proves scope coverage, including future paths.
use super::policy::Policy;
use crate::command::DynResult;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug)]
struct Rule {
    pattern: String,
    owners: Vec<String>,
    rooted: bool,
    directory: bool,
    components: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OwnershipDecision {
    pub schema_version: u32,
    pub protected_patterns: Vec<String>,
    pub reviewer_team: String,
    pub evidence_status: &'static str,
}

pub(crate) fn canonical_patterns(policy: &Policy) -> Vec<String> {
    policy
        .scope
        .directory_prefixes
        .iter()
        .chain(&policy.scope.exact_paths)
        .map(|path| format!("/{path}"))
        .chain(
            policy
                .scope
                .root_filename_prefixes
                .iter()
                .map(|prefix| format!("/{prefix}*")),
        )
        .collect()
}

pub(crate) fn validate(policy: &Policy, codeowners: &str) -> DynResult<OwnershipDecision> {
    policy.validate()?;
    let rules = parse(codeowners)?;
    let patterns = canonical_patterns(policy);
    let suffix_start = rules
        .len()
        .checked_sub(patterns.len())
        .ok_or("CODEOWNERS missing terminal CI policy protection block")?;
    let suffix = &rules[suffix_start..];
    let expected: BTreeSet<_> = patterns.iter().map(String::as_str).collect();
    let actual: BTreeSet<_> = suffix.iter().map(|rule| rule.pattern.as_str()).collect();
    let team = format!("@{}", policy.reviewer_team);
    if actual != expected || suffix.iter().any(|rule| rule.owners != [team.clone()]) {
        return Err("CODEOWNERS must end with exactly every canonical policy pattern, team-only; later overrides are forbidden".into());
    }
    // Use the same last-match matcher for actual representative identities.
    for path in policy.scope.exact_paths.iter().map(String::as_str) {
        if effective_owners(codeowners, path)? != [team.as_str()] {
            return Err("CODEOWNERS effective control ownership is not team-only".into());
        }
    }
    Ok(OwnershipDecision {
        schema_version: 1,
        protected_patterns: patterns,
        reviewer_team: policy.reviewer_team.clone(),
        evidence_status: "local-ownership-validation-only",
    })
}

pub(crate) fn effective_owners(codeowners: &str, path: &str) -> DynResult<Vec<String>> {
    super::scope::validate_path(path)?;
    Ok(effective_owners_from_rules(&parse(codeowners)?, path)
        .into_iter()
        .map(str::to_owned)
        .collect())
}

fn effective_owners_from_rules<'a>(rules: &'a [Rule], path: &str) -> Vec<&'a str> {
    rules
        .iter()
        .rev()
        .find(|rule| rule.matches(path))
        .map(|rule| rule.owners.iter().map(String::as_str).collect())
        .unwrap_or_default()
}

fn parse(codeowners: &str) -> DynResult<Vec<Rule>> {
    if codeowners.len() >= 3_000_000 {
        return Err("CODEOWNERS exceeds GitHub's supported size".into());
    }
    let mut rules = Vec::new();
    for raw in codeowners.lines() {
        let content = raw.trim();
        if content.is_empty() || content.starts_with('#') {
            continue;
        }
        if content.contains('#') {
            return Err(
                "unsupported CODEOWNERS inline comment or literal #; use standalone comments"
                    .into(),
            );
        }
        let mut fields = content.split_whitespace();
        let pattern = fields.next().ok_or("missing CODEOWNERS pattern")?;
        let owners: Vec<_> = fields.map(str::to_owned).collect();
        if owners.iter().any(|owner| !valid_owner(owner)) {
            return Err(
                "unsupported CODEOWNERS owner syntax; use @login or @organization/team".into(),
            );
        }
        rules.push(Rule::parse(pattern, owners)?);
    }
    Ok(rules)
}

fn valid_owner(owner: &str) -> bool {
    let Some(name) = owner.strip_prefix('@') else {
        return false;
    };
    !name.is_empty()
        && name.split('/').count() <= 2
        && name.split('/').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == b'-')
        })
}

impl Rule {
    fn parse(pattern: &str, owners: Vec<String>) -> DynResult<Self> {
        if pattern
            .chars()
            .any(|ch| "![]?\\".contains(ch) || ch.is_control())
        {
            return Err(format!("unsupported CODEOWNERS pattern syntax: {pattern}").into());
        }
        let directory = pattern.ends_with('/');
        let body = pattern.trim_start_matches('/').trim_end_matches('/');
        let components: Vec<_> = body.split('/').map(str::to_owned).collect();
        if body.is_empty()
            || pattern.starts_with("//")
            || pattern.ends_with("//")
            || components.iter().any(|part| {
                matches!(part.as_str(), "" | "." | "..") || (part.contains("**") && part != "**")
            })
        {
            return Err(format!("unsupported CODEOWNERS path syntax: {pattern}").into());
        }
        Ok(Self {
            pattern: pattern.into(),
            owners,
            directory,
            rooted: pattern.starts_with('/') || body.contains('/'),
            components,
        })
    }

    fn matches(&self, path: &str) -> bool {
        let parts: Vec<_> = path.split('/').collect();
        if self.rooted {
            return self.matches_components(&parts);
        }
        (0..parts.len()).any(|offset| self.matches_components(&parts[offset..]))
    }

    fn matches_components(&self, parts: &[&str]) -> bool {
        let mut positions = vec![false; parts.len() + 1];
        positions[0] = true;
        for (component_index, component) in self.components.iter().enumerate() {
            let mut next = vec![false; parts.len() + 1];
            if component == "**" {
                let mut reachable = false;
                for (index, prior) in positions.iter().enumerate() {
                    // A trailing '/**' denotes contents, not the directory itself.
                    if component_index + 1 == self.components.len() {
                        next[index] = reachable;
                        reachable |= prior;
                    } else {
                        reachable |= prior;
                        next[index] = reachable;
                    }
                }
            } else {
                for index in 0..parts.len() {
                    next[index + 1] = positions[index] && segment_matches(component, parts[index]);
                }
            }
            positions = next;
        }
        positions
            .iter()
            .enumerate()
            .any(|(consumed, matched)| *matched && (!self.directory || consumed < parts.len()))
    }
}

// '*' is bounded to a path component. '**' is handled only as a whole component.
fn segment_matches(pattern: &str, text: &str) -> bool {
    let pattern = pattern.as_bytes();
    let text = text.as_bytes();
    let (mut p, mut t, mut star, mut retry) = (0, 0, None, 0);
    while t < text.len() {
        if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            retry = t;
        } else if p < pattern.len() && pattern[p] == text[t] {
            p += 1;
            t += 1;
        } else if let Some(index) = star {
            retry += 1;
            t = retry;
            p = index + 1;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}
