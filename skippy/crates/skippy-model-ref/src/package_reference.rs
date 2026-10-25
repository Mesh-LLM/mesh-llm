//! Hugging Face package references used by local package-acquisition callers.
//! Parsing preserves revision names. It does not resolve them to commits.
use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackageReference {
    repo: String,
    revision: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackageReferenceError;

impl fmt::Display for PackageReferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected hf://namespace/repo optionally followed by @revision or :revision")
    }
}
impl Error for PackageReferenceError {}

impl PackageReference {
    pub fn parse(input: &str) -> Result<Self, PackageReferenceError> {
        if input.len() > 4096 {
            return Err(PackageReferenceError);
        }
        let rest = input.strip_prefix("hf://").ok_or(PackageReferenceError)?;
        let delimiters = rest
            .bytes()
            .filter(|byte| matches!(byte, b'@' | b':'))
            .count();
        if delimiters > 1 {
            return Err(PackageReferenceError);
        }
        let (repo, revision) = rest.split_once(['@', ':']).unwrap_or((rest, "main"));
        let components: Vec<_> = repo.split('/').collect();
        if components.len() != 2
            || components.iter().any(|part| !safe_component(part))
            || !safe_revision(revision)
        {
            return Err(PackageReferenceError);
        }
        Ok(Self {
            repo: repo.to_owned(),
            revision: revision.to_owned(),
        })
    }
    pub fn repo(&self) -> &str {
        &self.repo
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
}

fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn safe_revision(value: &str) -> bool {
    !value.is_empty()
        && !value.contains("..")
        && !value.contains("@{")
        && value.split('/').all(|part| {
            !part.is_empty()
                && !part.starts_with('.')
                && !part.ends_with('.')
                && !part.ends_with(".lock")
                && part.chars().all(|character| {
                    !character.is_control()
                        && !character.is_whitespace()
                        && !matches!(character, '~' | '^' | ':' | '?' | '*' | '[' | '\\' | '@')
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_revisions_preserve_branch_names_and_default() {
        for (input, revision) in [
            ("hf://org/repo", "main"),
            ("hf://org/repo:release/one", "release/one"),
            ("hf://org/repo@release/one", "release/one"),
            ("hf://org/repo@abc123", "abc123"),
            ("hf://org/repo:release=v2", "release=v2"),
            ("hf://org/repo@release/étape", "release/étape"),
            ("hf://org/repo@release#v2", "release#v2"),
            ("hf://org/repo@%2e%2e", "%2e%2e"),
        ] {
            let parsed = PackageReference::parse(input).unwrap();
            assert_eq!(parsed.repo(), "org/repo");
            assert_eq!(parsed.revision(), revision);
        }
        assert_eq!(
            crate::ModelRef::parse("org/repo:Q4")
                .unwrap()
                .selector
                .as_deref(),
            Some("Q4")
        );
    }
    #[test]
    fn malformed_references_never_become_cache_paths() {
        for input in [
            "org/repo",
            "https://org/repo",
            "hf://repo",
            "hf://a/b/c",
            "hf:///b",
            "hf://a/..",
            "hf://a/b:",
            "hf://a/b@",
            "hf://a/b:x@y",
            "hf://a/b@x@y",
            "hf://a/b:x:y",
            "hf://a/b@../x",
            "hf://a/b@x//y",
            "hf://a/b@/x",
            "hf://a/b@x/",
            "hf://a/b@x\\y",
            "hf://a/b@x\ny",
            "hf://a/b@x y",
            "hf://a/b@x?y",
            "hf://a/b@x~y",
            "hf://a/b@x^y",
            "hf://a/b@x*y",
            "hf://a/b@x[y",
            "hf://a/b@x\u{7f}y",
            "hf://a/b@x\u{85}y",
            "hf://a/b@x/../y",
            "hf://a/b@.hidden/x",
            "hf://a/b@x./y",
            "hf://a/b@x.lock/y",
            "hf://a/b@x.lock",
        ] {
            assert!(PackageReference::parse(input).is_err(), "{input:?}");
        }
        assert!(PackageReference::parse(&format!("hf://a/b@{}", "x".repeat(4096))).is_err());
    }
}
