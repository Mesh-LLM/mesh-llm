#![cfg(feature = "automation-contracts")]
use std::{error::Error, fs, path::Path};

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn release_publication_keeps_its_approval_and_content_boundaries() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("root")?;
    let script = fs::read_to_string(root.join("scripts/release-notes-generate.sh"))?;
    let approval = script.find("RELEASE_NOTES_APPROVED").ok_or("approval")?;
    let publication = script.find("gh release edit").ok_or("publication")?;
    assert!(approval < publication);
    assert!(!script.contains("GITHUB_ACTIONS"));
    assert!(
        script
            .find("cargo xtool release notes-link")
            .ok_or("link")?
            < script
                .find("cargo xtool release notes-classify")
                .ok_or("classify")?
    );
    assert!(
        script
            .find("the link pass dropped a published entry")
            .ok_or("loss gate")?
            < publication
    );
    assert!(script.contains("body.github.md"));
    Ok(())
}

#[path = "lane_l2_quality_contracts/cli_website.rs"]
mod cli_website;

#[cfg(unix)]
#[path = "lane_l2_quality_contracts/skippy_cpp_api.rs"]
mod skippy_cpp_api;
