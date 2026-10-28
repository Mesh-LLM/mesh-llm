//! Installation diagnostics only: package declarations never rename an install.
use std::{fs, path::Path};

use serde::Deserialize;

#[derive(Deserialize)]
struct Declaration {
    name: String,
}

pub(crate) fn mismatch_hint(directory: &Path, requested: &str, suffix: &str) -> Option<String> {
    let text = fs::read_to_string(directory.join("plugin.toml")).ok()?;
    let declaration: Declaration = toml::from_str(&text).ok()?;
    let name = declaration.name;
    // Do not echo untrusted shell syntax or suggest an absent executable.
    if name == requested
        || name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        || !directory.join(format!("{name}{suffix}")).is_file()
    {
        return None;
    }
    Some(format!(
        "this package installs as '{name}'; use `plugins install {name}`"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggests_declared_identity_without_changing_install() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("plugin.toml"), "name = 'pay-lightning'\n").unwrap();
        fs::write(root.path().join("pay-lightning"), "fixture").unwrap();
        assert_eq!(
            mismatch_hint(root.path(), "lexe-wallet", "").as_deref(),
            Some("this package installs as 'pay-lightning'; use `plugins install pay-lightning`")
        );
        assert!(mismatch_hint(root.path(), "pay-lightning", "").is_none());
        assert!(mismatch_hint(root.path(), "lexe-wallet", ".exe").is_none());
    }

    #[test]
    fn unsafe_or_invalid_declarations_do_not_generate_commands() {
        let root = tempfile::tempdir().unwrap();
        for text in [
            "name = '../other'",
            "name = 'bad;command'",
            "not toml",
            "name = ''",
        ] {
            fs::write(root.path().join("plugin.toml"), text).unwrap();
            assert!(mismatch_hint(root.path(), "requested", "").is_none());
        }
    }
}
