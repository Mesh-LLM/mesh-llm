//! Complete protected-audit and product-restore sequence with disjoint owned staging.
use super::{
    support::{self, Fixture},
    verified_pr_authority::{SOURCE, command, digest, run, script, staged},
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeSet, fs, path::Path, process::Command};

const PRODUCT_SOURCE: &str = "1123456789012345678901234567890123456789";

fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn action_steps(name: &str) -> Vec<Node> {
    let Some(Node::Seq(steps)) = support::action(name)
        .get("runs")
        .unwrap()
        .get("steps")
        .cloned()
    else {
        panic!("{name} steps")
    };
    steps
}
fn sentinel_steps() -> Vec<Node> {
    let path = support::root().join(".github/workflows/ci-quality-slice.yml");
    let document = workflow_yaml::parse(&fs::read_to_string(path).unwrap()).unwrap();
    let job = document
        .get("jobs")
        .unwrap()
        .get("authority_sentinel")
        .unwrap();
    let Some(Node::Seq(steps)) = job.get("steps").cloned() else {
        panic!("sentinel steps")
    };
    steps
}

/// The directory a role downloads into, after checking every shell reference uses it.
fn owned_stage(role: &str, steps: &[Node]) -> String {
    let downloads: Vec<_> = steps
        .iter()
        .filter(|s| text(s, "uses").starts_with("actions/download-artifact@"))
        .collect();
    let [download] = downloads.as_slice() else {
        panic!("{role}: one download")
    };
    let stage = text(download.get("with").unwrap(), "path")
        .strip_prefix("${{ runner.temp }}/")
        .unwrap_or_else(|| panic!("{role}: runner temp stage"))
        .to_owned();
    let runs: Vec<_> = steps.iter().map(|s| text(s, "run")).collect();
    assert!(
        runs.iter().any(|r| r
            .lines()
            .any(|l| l.trim() == format!("mkdir \"$RUNNER_TEMP/{stage}\""))),
        "{role}: admission creates its own stage"
    );
    for reference in runs
        .iter()
        .flat_map(|r| r.split("$RUNNER_TEMP/immutable-").skip(1))
    {
        let used = format!("immutable-{}", reference.split(['"', '/']).next().unwrap());
        assert_eq!(used, stage, "{role}: foreign stage reference");
    }
    stage
}

#[test]
fn protected_audit_sentinel_and_product_restore_own_disjoint_stages() {
    let stages = [
        owned_stage("audit", &action_steps("audit-pr-authority-verified")),
        owned_stage("sentinel", &sentinel_steps()),
        owned_stage("restore", &action_steps("restore-automation")),
    ];
    let distinct: BTreeSet<_> = stages.iter().collect();
    assert_eq!(distinct.len(), stages.len(), "{stages:?}");
}

#[test]
fn every_job_audits_before_checkout_and_restores_product_bytes_after_it() {
    let mut checked = 0;
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        let document = workflow_yaml::parse(&fs::read_to_string(&path).unwrap()).unwrap();
        let Some(jobs) = document.get("jobs") else {
            continue;
        };
        for (name, job) in jobs.entries() {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                continue;
            };
            let find = |pattern: &dyn Fn(&str) -> bool| {
                steps.iter().position(|s| pattern(text(s, "uses")))
            };
            let Some(restore) = find(&|u| u == "./.github/actions/restore-automation") else {
                continue;
            };
            let audit = find(&|u| {
                u.contains("/audit-depot-pr-isolation@")
                    || u.contains("/audit-pr-authority-verified")
            });
            let checkout = find(&|u| u.starts_with("actions/checkout@")).unwrap();
            let file = path.file_name().unwrap().to_string_lossy();
            if let Some(audit) = audit {
                assert!(audit < checkout, "{file}/{name}: audit before checkout");
            }
            assert!(
                checkout < restore,
                "{file}/{name}: product restore after checkout"
            );
            checked += 1;
        }
    }
    assert!(checked > 0);
}

fn bundle(stage: &Path, source: &str, binary: &[u8]) {
    fs::write(stage.join("xtask"), binary).unwrap();
    fs::write(stage.join("source.txt"), format!("{source}\n")).unwrap();
    fs::write(
        stage.join("SHA256SUMS"),
        format!(
            "{}  xtask\n{}  source.txt\n",
            digest(binary),
            digest(format!("{source}\n").as_bytes())
        ),
    )
    .unwrap();
}
fn restore(
    directory: &Path,
    step: usize,
    artifact: &str,
    binary_sha256: &str,
) -> std::process::Output {
    let body = text(&action_steps("restore-automation")[step], "run").to_owned();
    let mut command = Command::new("/bin/bash");
    command
        .env_clear()
        .current_dir(directory)
        .args(["-c", &body])
        .env("PATH", "/usr/bin:/bin:/opt/homebrew/bin")
        .env("RUNNER_TEMP", directory)
        .env("RUNNER_OS", "macOS")
        .env("RUNNER_ARCH", "ARM64")
        .env("AUTOMATION_SOURCE_SHA", PRODUCT_SOURCE)
        .env("AUTOMATION_PRODUCT_SOURCE_SHA", PRODUCT_SOURCE)
        .env("AUTOMATION_SOURCE_PROFILE", "product-clean")
        .env("AUTOMATION_REQUIRE_PRODUCT", "true")
        .env("AUTOMATION_ARTIFACT_ID", artifact)
        .env("AUTOMATION_BINARY_SHA256", binary_sha256)
        .env("AUTOMATION_PRODUCER_OS", "macOS")
        .env("AUTOMATION_PRODUCER_ARCH", "ARM64")
        .env("GITHUB_ENV", directory.join("github.env"));
    Fixture::new().run(command)
}

#[test]
fn audited_authority_and_restored_product_coexist_in_one_job_without_substitution() {
    let authority = fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap();
    let product = b"#!/bin/bash\necho product-bytes\n".to_vec();
    // Protected admission/download, then the actual native audit.
    let directory = staged(&authority);
    let mut audit = command(directory.path());
    audit.env("AUTOMATION_BINARY_SHA256", digest(&authority));
    let audited = run(&mut audit, &format!("{}\n{}", script(2), script(3)));
    assert!(audited.status.success(), "{audited:?}");
    // Checkout happens here; then product admission must create its own stage.
    assert!(
        restore(directory.path(), 0, "5678", &digest(&product))
            .status
            .success()
    );
    let product_stage = directory.path().join("immutable-automation-restored");
    bundle(&product_stage, PRODUCT_SOURCE, &product);
    let restored = restore(directory.path(), 2, "5678", &digest(&product));
    assert!(restored.status.success(), "{restored:?}");
    assert_eq!(
        fs::read_to_string(directory.path().join("github.env")).unwrap(),
        format!(
            "MESH_LLM_AUTOMATION_BIN={}/xtask\n",
            product_stage.display()
        )
    );
    // Product restore left the audited authority bytes untouched.
    let audit_stage = directory.path().join("immutable-authority-audit");
    assert_eq!(fs::read(audit_stage.join("xtask")).unwrap(), authority);
    // Neither role admits the other's bytes or identity.
    let mut wrong = command(directory.path());
    wrong.env("AUTOMATION_BINARY_SHA256", digest(&product));
    bundle(&audit_stage, PRODUCT_SOURCE, &product);
    assert!(
        !run(&mut wrong, &format!("{}\n{}", script(2), script(3)))
            .status
            .success()
    );
    bundle(&product_stage, SOURCE, &authority);
    assert!(
        !restore(directory.path(), 2, "5678", &digest(&product))
            .status
            .success()
    );
    assert!(
        !restore(directory.path(), 2, "5678", &digest(&authority))
            .status
            .success()
    );
}
