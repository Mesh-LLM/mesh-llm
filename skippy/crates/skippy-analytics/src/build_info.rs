//! The version this build reports about itself.

/// `SKIPPY_BUILD_VERSION` when the build sets it, else the crate version.
/// The same rule `skippy runtime` uses to name the build.
pub const BUILD_VERSION: &str = match option_env!("SKIPPY_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

/// Whether `version` is a commit build, such as `0.80.0+gabc1234` or
/// `0.80.0+gabc1234.dirty`.
#[must_use]
pub fn is_sha_build(version: &str) -> bool {
    let Some((_, metadata)) = version.split_once('+') else {
        return false;
    };
    let sha = metadata.strip_suffix(".dirty").unwrap_or(metadata);
    let Some(hex) = sha.strip_prefix('g') else {
        return false;
    };
    hex.len() >= 6 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_builds_are_recognized() {
        assert!(is_sha_build("0.80.0+gabc1234"));
        assert!(is_sha_build("0.80.0+gabc1234.dirty"));
        assert!(!is_sha_build("0.80.0"));
        assert!(!is_sha_build("0.80.0-rc1"));
        assert!(!is_sha_build("0.80.0+build.5"));
    }
}
