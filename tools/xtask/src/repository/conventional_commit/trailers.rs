//! Attribution-trailer policy from `scripts/check-conventional-commit.py`:
//! agent, bot and relay identities are rejected from `*-by:` trailers.

use crate::repository::python_text::{is_space, strip};

const DENIED_DOMAINS: &[&str] = &["buzz.xyz"];
const DENIED_ADDRESSES: &[&str] = &["noreply@anthropic.com", "noreply@coderabbit.ai"];
/// Matched against whole ASCII name tokens, so "Sol" is denied but not "Solomon".
const DENIED_NAMES: &[&str] = &[
    "claude",
    "anthropic",
    "chatgpt",
    "openai",
    "codex",
    "copilot",
    "sisyphus",
    "astra",
    "sol",
    "luna",
    "terra",
    "coderabbit",
    "coderabbitai",
    "devin",
    "cursor",
];

struct Trailer<'a> {
    name: &'a str,
    email: Option<&'a str>,
}

/// Problems with every trailer line in `lines`.
pub(super) fn check_trailers<'a>(lines: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    lines
        .into_iter()
        .filter_map(|line| {
            let line = strip(line);
            let trailer = parse_trailer(line)?;
            let reason = denied_identity(trailer.name, trailer.email.unwrap_or_default())?;
            Some(format!("drop '{line}': {reason}"))
        })
        .collect()
}

/// Letters the case-insensitive `[A-Za-z]` class accepts.
fn is_token_letter(ch: char) -> bool {
    ch.is_ascii_alphabetic() || matches!(ch, '\u{130}' | '\u{131}' | '\u{17f}' | '\u{212a}')
}

/// `^([A-Za-z][A-Za-z-]*-by)\s*:\s*([^<]*?)\s*(?:<([^>]*)>)?\s*$`, IGNORECASE.
fn parse_trailer(line: &str) -> Option<Trailer<'_>> {
    let token_len = line.len()
        - line
            .trim_start_matches(|ch: char| is_token_letter(ch) || ch == '-')
            .len();
    let token = &line[..token_len];
    if !token.starts_with(is_token_letter) || !token.to_lowercase().ends_with("-by") {
        return None;
    }
    let rest = line[token_len..]
        .trim_start_matches(is_space)
        .strip_prefix(':')?;
    let rest = rest.trim_start_matches(is_space);
    let Some((name, tail)) = rest.split_once('<') else {
        return Some(Trailer {
            name: rest.trim_end_matches(is_space),
            email: None,
        });
    };
    let (email, after) = tail.split_once('>')?;
    after.chars().all(is_space).then_some(Trailer {
        name: name.trim_end_matches(is_space),
        email: Some(email),
    })
}

/// Why an identity is denied, or `None` when it is allowed.
fn denied_identity(name: &str, email: &str) -> Option<String> {
    let email = strip(email).to_lowercase();
    let name = strip(name);
    if !email.is_empty() {
        if DENIED_ADDRESSES.contains(&email.as_str()) {
            return Some(format!("'{email}' is an agent attribution address"));
        }
        let domain = email
            .rsplit_once('@')
            .map_or(email.as_str(), |(_, domain)| domain);
        if DENIED_DOMAINS
            .iter()
            .any(|denied| domain == *denied || domain.ends_with(&format!(".{denied}")))
        {
            return Some(format!("'{domain}' is a relay identity domain"));
        }
        if email.contains("[bot]") {
            return Some(format!("'{email}' is a bot account"));
        }
    }
    let lowered = name.to_lowercase();
    let mut hits = lowered
        .split(|ch: char| !(ch.is_ascii_lowercase() || ch.is_ascii_digit()))
        .filter(|token| DENIED_NAMES.contains(token))
        .collect::<Vec<_>>();
    hits.sort_unstable();
    hits.dedup();
    if !hits.is_empty() {
        return Some(format!(
            "'{name}' names an agent or bot ({})",
            hits.join(", ")
        ));
    }
    lowered
        .contains("[bot]")
        .then(|| format!("'{name}' is a bot account"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_repository_commits_trailer_grammar() {
        let parsed = parse_trailer("Co-authored-by : X Y <a@b> ").expect("trailer");
        assert_eq!((parsed.name, parsed.email), ("X Y", Some("a@b")));
        assert!(parse_trailer("X-by: a <b> c").is_none());
        assert!(parse_trailer("-by: a").is_none());
        assert!(parse_trailer("Acked-BY: Real").is_some_and(|t| t.name == "Real"));
        assert!(parse_trailer("Co-authored-by X").is_none());
    }

    #[test]
    fn migration_repository_commits_identity_policy() {
        assert_eq!(
            denied_identity("Sol-Luna", "").as_deref(),
            Some("'Sol-Luna' names an agent or bot (luna, sol)")
        );
        assert_eq!(
            denied_identity("", "A@X.BUZZ.XYZ ").as_deref(),
            Some("'x.buzz.xyz' is a relay identity domain")
        );
        assert_eq!(denied_identity("Solomon", "real@example.com"), None);
        assert_eq!(denied_identity("Real", "abuzz.xyz"), None);
    }
}
