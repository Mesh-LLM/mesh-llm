//! The fixed-shape patterns the identity catalog validates, written as
//! character-class checks equivalent to the legacy `re.fullmatch` calls.

pub(crate) const REPOSITORY: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner";
pub(crate) const EPOCH_PREFIX: &str = "mesh-llm-cuda-runner-sha256-";

pub(crate) fn lower_hex(text: &str, len: usize) -> bool {
    text.len() == len
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// `REPOSITORY@sha256:<64 lowercase hex>`.
pub(crate) fn is_reference(text: &str) -> bool {
    text.strip_prefix(REPOSITORY)
        .and_then(|rest| rest.strip_prefix("@sha256:"))
        .is_some_and(|hex| lower_hex(hex, 64))
}

/// `[a-z][a-z0-9_-]*`.
pub(crate) fn is_identifier(text: &str) -> bool {
    let mut bytes = text.bytes();
    bytes.next().is_some_and(|first| first.is_ascii_lowercase())
        && bytes.all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
}

/// `[a-z0-9_-]+\.ya?ml`.
pub(crate) fn is_workflow_name(text: &str) -> bool {
    let stem = text
        .strip_suffix(".yml")
        .or_else(|| text.strip_suffix(".yaml"));
    stem.is_some_and(|stem| {
        !stem.is_empty()
            && stem
                .bytes()
                .all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
    })
}

/// `[A-Za-z0-9_./*?-]+`.
pub(crate) fn is_recipe_input(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_./*?-".contains(&byte))
}

/// `mesh-llm-sccache-seed-linux-x86_64-img-{d}-epoch-{d}-v[0-9]+-`.
pub(crate) fn is_seed_prefix(text: &str, short_digest: &str) -> bool {
    let head =
        format!("mesh-llm-sccache-seed-linux-x86_64-img-{short_digest}-epoch-{short_digest}-v");
    text.strip_prefix(&head)
        .and_then(|rest| rest.strip_suffix('-'))
        .is_some_and(|version| {
            !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit())
        })
}

/// `dtolnay/rust-toolchain@[0-9a-f]{40}`.
pub(crate) fn is_pinned_rust_action(text: &str) -> bool {
    text.strip_prefix("dtolnay/rust-toolchain@")
        .is_some_and(|sha| lower_hex(sha, 40))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_operations_identity_patterns_match_legacy_regexes() {
        assert!(is_reference(&format!(
            "{REPOSITORY}@sha256:{}",
            "a".repeat(64)
        )));
        assert!(!is_reference(&format!("{REPOSITORY}:latest")));
        assert!(is_identifier("rust-clippy_1") && !is_identifier("1a") && !is_identifier("A"));
        assert!(
            is_workflow_name("ci.yml")
                && is_workflow_name("a.yaml")
                && !is_workflow_name("../x.yml")
        );
        assert!(is_recipe_input("Cargo.lock") && !is_recipe_input("a b"));
        assert!(is_seed_prefix(
            "mesh-llm-sccache-seed-linux-x86_64-img-abc-epoch-abc-v3-",
            "abc"
        ));
        assert!(!is_seed_prefix(
            "mesh-llm-sccache-seed-linux-x86_64-img-abc-epoch-abc-v-",
            "abc"
        ));
        assert!(is_pinned_rust_action(&format!(
            "dtolnay/rust-toolchain@{}",
            "0".repeat(40)
        )));
    }
}
