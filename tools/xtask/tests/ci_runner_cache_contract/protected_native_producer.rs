//! Native protected automation producer boundaries, without compilation or upload.
use super::{
    support::{self, Fixture},
    workflow_yaml::{self, Node},
};
use std::{fs, process::Command};

const WORKFLOW: &str = ".github/workflows/protected-automation-artifact.yml";
const RUNNERS: &str = "${{ inputs.platform == 'linux-arm64' && 'ubuntu-24.04-arm' || inputs.platform == 'macos-arm64' && 'macos-15' || inputs.platform == 'windows-x64' && 'windows-2022' || 'ubuntu-24.04' }}";

fn current() -> Node {
    workflow_yaml::parse(&fs::read_to_string(support::root().join(WORKFLOW)).unwrap()).unwrap()
}

fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}

fn producer(document: &Node) -> &Node {
    document.get("jobs").unwrap().get("producer").unwrap()
}

fn steps(document: &Node) -> &[Node] {
    let Node::Seq(steps) = producer(document).get("steps").unwrap() else {
        panic!("steps")
    };
    steps
}

fn admission() -> String {
    text(&steps(&current())[0], "run").to_owned()
}

fn check(document: &Node) -> Result<(), &'static str> {
    let inputs = document
        .get("on")
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"))
        .ok_or("inputs")?;
    if inputs.entries().len() != 3
        || inputs.get("platform").is_none()
        || inputs.get("lane").is_none()
        || inputs.get("protected_source_sha").is_none()
    {
        return Err("semantic inputs only");
    }
    let job = producer(document);
    if text(job, "runs-on") != RUNNERS {
        return Err("closed hosted native runner map");
    }
    let permissions = job.get("permissions").ok_or("permissions")?;
    if permissions.entries().len() != 1 || text(permissions, "contents") != "read" {
        return Err("read-only producer");
    }
    check_steps(document)
}

fn check_steps(document: &Node) -> Result<(), &'static str> {
    let job = producer(document);
    let steps = steps(document);
    if steps.len() != 4 || text(&steps[0], "id") != "identity" || text(&steps[0], "run").is_empty()
    {
        return Err("identity admission before checkout");
    }
    if text(&steps[1], "uses") != "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09" {
        return Err("pinned checkout");
    }
    let checkout = steps[1].get("with").ok_or("checkout inputs")?;
    if text(checkout, "ref") != "${{ github.event.repository.default_branch }}"
        || text(checkout, "persist-credentials") != "false"
    {
        return Err("protected checkout without credentials");
    }
    if text(&steps[2], "id") != "source"
        || !text(&steps[2], "run").contains("source_sha=\"$(git rev-parse HEAD)\"")
    {
        return Err("actual checked-out source identity");
    }
    let source_env = steps[2].get("env").ok_or("protected source environment")?;
    if text(source_env, "AUTOMATION_PROTECTED_SOURCE_SHA") != "${{ inputs.protected_source_sha }}"
        || !text(&steps[2], "run")
            .contains("git merge-base --is-ancestor \"$AUTOMATION_PROTECTED_SOURCE_SHA\" HEAD")
        || !text(&steps[2], "run")
            .contains("git checkout --detach \"$AUTOMATION_PROTECTED_SOURCE_SHA\"")
    {
        return Err("reviewed default-branch ancestry before source selection");
    }
    if text(&steps[3], "uses") != "./.github/actions/upload-automation"
        || text(&steps[3], "id") != "upload"
    {
        return Err("owned producer");
    }
    let upload = steps[3].get("with").ok_or("upload inputs")?;
    for (name, binding) in [
        ("source_sha", "${{ steps.source.outputs.source_sha }}"),
        ("source-profile", "protected-clean"),
        (
            "runner-profile",
            "${{ runner.os == 'Linux' && 'hosted-bare' || 'image' }}",
        ),
        ("ci-lane", "${{ inputs.lane }}"),
    ] {
        if text(upload, name) != binding {
            return Err("source/profile/scope projection");
        }
    }
    check_outputs(job)
}

fn check_outputs(job: &Node) -> Result<(), &'static str> {
    let outputs = job.get("outputs").ok_or("outputs")?;
    for (name, binding) in [
        ("source_sha", "${{ steps.upload.outputs.source_sha }}"),
        ("artifact_id", "${{ steps.upload.outputs.artifact_id }}"),
        ("binary_sha256", "${{ steps.upload.outputs.binary_sha256 }}"),
        ("producer_os", "${{ steps.identity.outputs.producer_os }}"),
        (
            "producer_arch",
            "${{ steps.identity.outputs.producer_arch }}",
        ),
    ] {
        if text(outputs, name) != binding {
            return Err("independent producer output");
        }
    }
    Ok(())
}

fn replace(node: &mut Node, key: &str, value: Node) {
    let Node::Map(fields) = node else {
        panic!("mapping")
    };
    if let Some((_, old)) = fields.iter_mut().find(|(name, _)| name == key) {
        *old = value;
    } else {
        fields.push((key.into(), value));
    }
}

#[test]
fn protected_native_producer_declares_read_only_closed_runners_and_independent_identity() {
    check(&current()).unwrap();
}

#[test]
fn protected_native_producer_rejects_pr_source_runner_injection_and_digest_substitution() {
    for attack in [
        "runner",
        "permission",
        "pr-source",
        "source-profile",
        "digest",
        "scope",
    ] {
        let mut document = current();
        let Node::Map(root) = &mut document else {
            panic!()
        };
        let jobs = &mut root.iter_mut().find(|(name, _)| name == "jobs").unwrap().1;
        let Node::Map(jobs) = jobs else { panic!() };
        let job = &mut jobs
            .iter_mut()
            .find(|(name, _)| name == "producer")
            .unwrap()
            .1;
        match attack {
            "runner" => replace(
                job,
                "runs-on",
                Node::Scalar("${{ inputs.platform }}".into()),
            ),
            "permission" => replace(
                job,
                "permissions",
                Node::Map(vec![("contents".into(), Node::Scalar("write".into()))]),
            ),
            "digest" => {
                let Node::Map(fields) = job else { panic!() };
                let outputs = &mut fields
                    .iter_mut()
                    .find(|(name, _)| name == "outputs")
                    .unwrap()
                    .1;
                replace(
                    outputs,
                    "binary_sha256",
                    Node::Scalar("${{ inputs.binary_sha256 }}".into()),
                );
            }
            _ => {
                let Node::Map(fields) = job else { panic!() };
                let Node::Seq(steps) = &mut fields
                    .iter_mut()
                    .find(|(name, _)| name == "steps")
                    .unwrap()
                    .1
                else {
                    panic!()
                };
                let step = if attack == "pr-source" {
                    &mut steps[1]
                } else {
                    &mut steps[3]
                };
                let Node::Map(fields) = step else { panic!() };
                let inputs = &mut fields
                    .iter_mut()
                    .find(|(name, _)| name == "with")
                    .unwrap()
                    .1;
                let (name, value) = match attack {
                    "pr-source" => ("ref", "${{ github.event.pull_request.head.sha }}"),
                    "source-profile" => ("source-profile", "release-prepared"),
                    _ => ("ci-lane", ""),
                };
                replace(inputs, name, Node::Scalar(value.into()));
            }
        }
        assert!(check(&document).is_err(), "{attack}");
    }
}

#[test]
fn native_identity_admission_covers_every_topic_and_refuses_mismatched_or_nonhosted_runners() {
    for (platform, os, arch) in [
        ("linux-x64", "Linux", "X64"),
        ("linux-arm64", "Linux", "ARM64"),
        ("macos-arm64", "macOS", "ARM64"),
        ("windows-x64", "Windows", "X64"),
    ] {
        for lane in ["quality", "website", "linux", "macos", "windows"] {
            let fixture = Fixture::new();
            let mut command = Command::new("/bin/bash");
            command
                .env_clear()
                .args(["-c", &admission()])
                .env("PATH", "/usr/bin:/bin");
            command
                .env("AUTOMATION_PLATFORM", platform)
                .env("AUTOMATION_LANE", lane);
            command
                .env("RUNNER_OS", os)
                .env("RUNNER_ARCH", arch)
                .env("RUNNER_ENVIRONMENT", "github-hosted");
            command
                .env("RUNNER_TEMP", fixture.path())
                .env("GITHUB_OUTPUT", fixture.path().join("outputs"));
            command.env("GITHUB_ENV", fixture.path().join("env"));
            let result = fixture.run(command);
            assert!(result.status.success(), "{platform}/{lane}: {result:?}");
            assert_eq!(
                fs::read_to_string(fixture.path().join("outputs")).unwrap(),
                format!("producer_os={os}\nproducer_arch={arch}\n")
            );
        }
    }
    for (name, value) in [
        ("AUTOMATION_PLATFORM", "self-hosted"),
        ("AUTOMATION_PLATFORM", ""),
        ("AUTOMATION_LANE", "release"),
        ("AUTOMATION_LANE", ""),
        ("RUNNER_OS", "Windows"),
        ("RUNNER_ARCH", "ARM64"),
        ("RUNNER_ENVIRONMENT", "self-hosted"),
    ] {
        let fixture = Fixture::new();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .args(["-c", &admission()])
            .env("PATH", "/usr/bin:/bin");
        command
            .env("AUTOMATION_PLATFORM", "linux-x64")
            .env("AUTOMATION_LANE", "linux");
        command
            .env("RUNNER_OS", "Linux")
            .env("RUNNER_ARCH", "X64")
            .env("RUNNER_ENVIRONMENT", "github-hosted");
        command
            .env("RUNNER_TEMP", fixture.path())
            .env("GITHUB_OUTPUT", fixture.path().join("outputs"));
        command
            .env("GITHUB_ENV", fixture.path().join("env"))
            .env(name, value);
        assert!(!fixture.run(command).status.success(), "{name}");
        assert!(!fixture.path().join("outputs").exists());
        assert!(!fixture.path().join("env").exists());
    }
}

#[test]
fn native_upload_scopes_ci_artifacts_and_initializes_protected_macos_without_changing_release_retention()
 {
    let action = support::action("upload-automation");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!()
    };
    let prepare = steps
        .iter()
        .find(|step| text(step, "id") == "prepare")
        .unwrap();
    assert_eq!(
        text(prepare, "if"),
        "runner.os == 'Windows' || inputs.runner-profile == 'hosted-bare' || (runner.os == 'macOS' && inputs.source-profile == 'protected-clean')"
    );
    let upload = steps
        .iter()
        .find(|step| text(step, "id") == "upload")
        .unwrap()
        .get("with")
        .unwrap();
    assert_eq!(
        text(upload, "retention-days"),
        "${{ inputs.ci-lane != '' && 1 || 90 }}"
    );
    assert_eq!(
        text(upload, "name"),
        "automation-${{ runner.os }}-${{ runner.arch }}-${{ inputs.source_sha }}${{ inputs.ci-lane != '' && format('-{0}', inputs.ci-lane) || '' }}"
    );
    let admission = text(&steps[0], "run");
    for lane in [
        "quality",
        "website",
        "linux",
        "macos",
        "windows",
        "release",
        "../escape",
    ] {
        let fixture = Fixture::new();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .args(["-c", admission])
            .env("PATH", "/usr/bin:/bin");
        command
            .env("AUTOMATION_CI_LANE", lane)
            .env("AUTOMATION_SOURCE_PROFILE", "release-prepared");
        command.env("AUTOMATION_PROFILE", "image").env(
            "AUTOMATION_SOURCE_SHA",
            "0123456789012345678901234567890123456789",
        );
        // Every scoped input must refuse the release claim before Git or build.
        assert!(!fixture.run(command).status.success(), "{lane}");
    }
}

#[test]
fn ci_scope_admission_accepts_clean_protected_topics_and_refuses_unknown_topics_before_git() {
    let action = support::action("upload-automation");
    let Node::Seq(steps) = action.get("runs").unwrap().get("steps").unwrap() else {
        panic!()
    };
    let stub = r#"
git() {
  touch "$FIXTURE_GIT_TOUCHED"
  case "$1" in
    rev-parse) printf '%s\n' "$AUTOMATION_SOURCE_SHA" ;;
    status) return 0 ;;
    *) return 92 ;;
  esac
}

"#;
    for lane in [
        "quality",
        "website",
        "linux",
        "macos",
        "windows",
        "release",
        "../escape",
    ] {
        let fixture = Fixture::new();
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .args(["-c", &format!("{stub}\n{}", text(&steps[0], "run"))]);
        command.env("PATH", "/usr/bin:/bin");
        command
            .env("AUTOMATION_CI_LANE", lane)
            .env("AUTOMATION_SOURCE_PROFILE", "protected-clean");
        command.env("AUTOMATION_PROFILE", "image").env(
            "AUTOMATION_SOURCE_SHA",
            "0123456789012345678901234567890123456789",
        );
        command.env("FIXTURE_GIT_TOUCHED", fixture.path().join("git-touched"));
        let result = fixture.run(command);
        let allowed = !matches!(lane, "release" | "../escape");
        assert_eq!(result.status.success(), allowed, "{lane}: {result:?}");
        assert_eq!(
            fixture.path().join("git-touched").exists(),
            allowed,
            "{lane}"
        );
    }
}

fn fixture_git(fixture: &Fixture, checkout: &std::path::Path, arguments: &[&str]) -> String {
    let mut command = Command::new("/usr/bin/git");
    command.env_clear().current_dir(checkout).args(arguments);
    command.env("PATH", "/usr/bin:/bin").env("GIT_MASTER", "1");
    command
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null");
    command
        .env("GIT_AUTHOR_NAME", "fixture")
        .env("GIT_COMMITTER_NAME", "fixture");
    command
        .env("GIT_AUTHOR_EMAIL", "fixture@example.invalid")
        .env("GIT_COMMITTER_EMAIL", "fixture@example.invalid");
    let output = fixture.run(command);
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[test]
fn protected_source_selection_uses_real_git_ancestry_and_refuses_an_unmerged_branch() {
    let fixture = Fixture::new();
    let checkout = fixture.path().join("checkout");
    fs::create_dir(&checkout).unwrap();
    fixture_git(&fixture, &checkout, &["init", "-q"]);
    let mut revisions = Vec::new();
    for contents in ["base", "protected"] {
        fs::write(checkout.join("source"), contents).unwrap();
        fixture_git(&fixture, &checkout, &["add", "source"]);
        fixture_git(
            &fixture,
            &checkout,
            &["-c", "core.hooksPath=/dev/null", "commit", "-qm", contents],
        );
        revisions.push(fixture_git(&fixture, &checkout, &["rev-parse", "HEAD"]));
    }
    fixture_git(
        &fixture,
        &checkout,
        &["checkout", "--detach", &revisions[0]],
    );
    fs::write(checkout.join("source"), "unmerged").unwrap();
    fixture_git(&fixture, &checkout, &["add", "source"]);
    fixture_git(
        &fixture,
        &checkout,
        &[
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "unmerged",
        ],
    );
    let unmerged = fixture_git(&fixture, &checkout, &["rev-parse", "HEAD"]);
    let source_step = text(&steps(&current())[2], "run").to_owned();
    for (index, source) in [&revisions[0], &revisions[1], &unmerged, &"0".repeat(40)]
        .into_iter()
        .enumerate()
    {
        fixture_git(
            &fixture,
            &checkout,
            &["checkout", "--detach", &revisions[1]],
        );
        let output_path = fixture.path().join(format!("output-{index}"));
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .current_dir(&checkout)
            .args(["-c", &source_step]);
        command.env("PATH", "/usr/bin:/bin").env("GIT_MASTER", "1");
        command
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null");
        command
            .env("AUTOMATION_PROTECTED_SOURCE_SHA", source)
            .env("GITHUB_OUTPUT", &output_path);
        let result = fixture.run(command);
        let allowed = index < 2;
        assert_eq!(
            result.status.success(),
            allowed,
            "source {index}: {result:?}"
        );
        if allowed {
            assert_eq!(
                fs::read_to_string(&output_path).unwrap(),
                format!("source_sha={source}\n")
            );
            assert_eq!(
                fixture_git(&fixture, &checkout, &["rev-parse", "HEAD"]),
                *source
            );
        } else {
            assert!(!output_path.exists());
            assert_eq!(
                fixture_git(&fixture, &checkout, &["rev-parse", "HEAD"]),
                revisions[1]
            );
        }
    }
}
