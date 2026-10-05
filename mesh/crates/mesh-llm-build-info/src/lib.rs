pub const BUILD_VERSION: &str = match option_env!("MESH_LLM_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

pub const RELEASE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn is_sha_build(version: &str) -> bool {
    let Some((_, metadata)) = version.split_once('+') else {
        return false;
    };

    let sha = if let Some(sha) = metadata.strip_suffix(".dirty") {
        sha
    } else {
        metadata
    };

    let Some(hex) = sha.strip_prefix('g') else {
        return false;
    };

    hex.len() >= 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_version_uses_override_when_present() {
        let expected = option_env!("MESH_LLM_BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"));
        assert_eq!(BUILD_VERSION, expected);
    }

    #[test]
    fn release_version_is_plain_package_version() {
        assert_eq!(RELEASE_VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn recognizes_clean_sha_build() {
        assert!(is_sha_build("0.68.0+gABCDEF"));
    }

    #[test]
    fn recognizes_dirty_sha_build() {
        assert!(is_sha_build("0.68.0+gABCDEF.dirty"));
    }

    #[test]
    fn recognizes_lowercase_sha_build() {
        assert!(is_sha_build("0.68.0+gabcdef"));
    }

    #[test]
    fn rejects_malformed_metadata() {
        assert!(!is_sha_build("0.68.0+gABCDEF.dirty.extra"));
    }

    #[test]
    fn rejects_missing_plus() {
        assert!(!is_sha_build("0.68.0gABCDEF"));
    }

    #[test]
    fn rejects_short_sha() {
        assert!(!is_sha_build("0.68.0+gABCD"));
    }

    #[test]
    fn rejects_non_hex_sha() {
        assert!(!is_sha_build("0.68.0+gABCDEX"));
    }
}
