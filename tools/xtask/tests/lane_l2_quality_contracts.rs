use std::{error::Error, fs, path::Path};

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn commit_validation_and_publication_keep_their_trust_and_approval_boundaries() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("root")?;
    let workflow = fs::read_to_string(root.join(".github/workflows/ci-quality-slice.yml"))?;
    let job = workflow
        .split("  commit_convention:\n")
        .nth(1)
        .ok_or("job")?
        .split("  runner_policy:\n")
        .next()
        .ok_or("job end")?;
    assert!(!job.contains("needs:"));
    assert!(job.contains("github.event_name == 'pull_request'"));
    assert!(job.contains("ref: ${{ github.event.repository.default_branch }}"));
    assert!(job.contains("persist-credentials: false"));
    assert!(job.contains("PR_TITLE: ${{ github.event.pull_request.title }}"));
    assert!(job.contains("PR_BODY: ${{ github.event.pull_request.body }}"));
    assert!(job.contains("repository conventional-commits --message \"$PR_TITLE\""));
    assert!(job.contains("--trailers-only"));
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
    for path in [
        "scripts/build-development-product.sh",
        "scripts/build-windows.ps1",
    ] {
        assert!(fs::read_to_string(root.join(path))?.contains("core.hooksPath scripts/hooks"));
    }
    assert!(
        fs::read_to_string(root.join("scripts/hooks/commit-msg"))?
            .contains("repository conventional-commits")
    );
    Ok(())
}

#[path = "lane_l2_quality_contracts/cli_website.rs"]
mod cli_website;
