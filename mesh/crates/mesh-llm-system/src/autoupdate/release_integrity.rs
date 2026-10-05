use anyhow::{Result, bail};
use sha2::Digest;

#[derive(Clone, Debug)]
pub(super) struct ReleaseAsset {
    pub(super) name: String,
    pub(super) sha256: String,
}

pub(super) fn github_release_asset_sha256(asset: &serde_json::Value) -> Option<String> {
    normalize_github_sha256(asset["digest"].as_str()?).ok()
}

fn normalize_github_sha256(value: &str) -> Result<String> {
    let Some(digest) = value.trim().strip_prefix("sha256:") else {
        bail!("GitHub release asset digest must use sha256");
    };
    let digest = digest.to_ascii_lowercase();
    if digest.len() != 64 || !digest.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("GitHub release asset has an invalid sha256 digest");
    }
    Ok(digest)
}

pub(super) fn verify_release_asset_bytes(bytes: &[u8], expected_sha256: &str) -> Result<()> {
    let expected = normalize_github_sha256(&format!("sha256:{expected_sha256}"))?;
    let actual = hex::encode(sha2::Sha256::digest(bytes));
    if actual != expected {
        bail!("release asset checksum mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

pub(super) fn release_asset<'a>(
    release: &'a super::ReleaseInfo,
    name: &str,
) -> Option<&'a ReleaseAsset> {
    release.assets.iter().find(|asset| asset.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_release_asset_digest_is_required_and_validated() {
        let digest = "a".repeat(64);
        let asset = serde_json::json!({
            "name": "mesh-llm-aarch64-apple-darwin.tar.gz",
            "digest": format!("sha256:{digest}")
        });
        assert_eq!(github_release_asset_sha256(&asset), Some(digest));

        let missing = serde_json::json!({
            "name": "mesh-llm-aarch64-apple-darwin.tar.gz"
        });
        assert_eq!(github_release_asset_sha256(&missing), None);
    }

    #[test]
    fn release_metadata_excludes_assets_without_valid_github_digests() {
        let valid_digest = "b".repeat(64);
        let release = super::super::release_info_from_json(&serde_json::json!({
            "tag_name": "v0.73.1",
            "assets": [
                {
                    "name": "verified.tar.gz",
                    "digest": format!("sha256:{valid_digest}")
                },
                {
                    "name": "missing.tar.gz"
                },
                {
                    "name": "invalid.tar.gz",
                    "digest": "sha256:not-a-digest"
                }
            ]
        }))
        .unwrap();

        assert_eq!(release.assets.len(), 1);
        assert_eq!(release.assets[0].name, "verified.tar.gz");
        assert_eq!(release.assets[0].sha256, valid_digest);
    }

    #[test]
    fn modified_release_asset_is_rejected_before_execution() {
        let expected = hex::encode(sha2::Sha256::digest(b"expected release archive"));
        let error = verify_release_asset_bytes(b"modified release archive", &expected)
            .expect_err("modified release asset must fail verification");
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }
}
