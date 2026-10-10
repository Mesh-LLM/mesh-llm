//! Remote identity evidence omits URL credentials; it never obtains remote history.
use super::provenance::{Error, Git, Result};
fn redact(url: &str) -> String {
    let mut text = url.to_owned();
    if let Some(scheme) = text.find("://") {
        let start = scheme + 3;
        let end = text[start..]
            .find(['/', '?', '#'])
            .map_or(text.len(), |n| start + n);
        if let Some(at) = text[start..end].rfind('@') {
            text.replace_range(start..start + at + 1, "");
        }
        if let Some(query) = text.find(['?', '#']) {
            text.truncate(query);
            text.push_str("?[redacted]");
        }
    } else if let Some(colon) = text.find(':')
        && !text[..colon].contains(['/', '\\'])
        && let Some(at) = text[..colon].rfind('@')
    {
        text.replace_range(..at + 1, "");
    }
    text
}
pub(crate) fn urls(git: &mut impl Git) -> Result<Vec<String>> {
    let out = git.read(&["remote", "-v"])?;
    if out.code != 0 {
        return Err(Error("Git remote evidence unavailable".into()));
    }
    let text = String::from_utf8(out.stdout)
        .map_err(|_| Error("Git remote evidence is not UTF-8".into()))?;
    text.lines()
        .map(|line| {
            let (name, rest) = line
                .split_once('\t')
                .ok_or_else(|| Error("Git remote evidence record invalid".into()))?;
            let (url, role) = rest
                .rsplit_once(" (")
                .ok_or_else(|| Error("Git remote evidence role absent".into()))?;
            if name.is_empty() || !matches!(role, "fetch)" | "push)") {
                return Err(Error("Git remote evidence role invalid".into()));
            }
            Ok(format!("{name}\t{} ({role}", redact(url)))
        })
        .collect()
}
