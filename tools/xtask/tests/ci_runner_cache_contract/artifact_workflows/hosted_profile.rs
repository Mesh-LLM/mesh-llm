//! Refuse unclassified explicit preparation profiles; source proof only.
use super::{Node, document, input, job, named, steps, text};
use std::collections::BTreeSet;

const LINUX_PROFILE: &str = "${{ runner.os == 'Linux' && 'hosted-bare' || 'image' }}";

fn literal_hosted_jobs_are_hosted(workflow: &Node) -> bool {
    workflow
        .get("jobs")
        .unwrap()
        .entries()
        .iter()
        .all(|(_, job)| {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                return true;
            };
            !steps
                .iter()
                .any(|step| input(step, "runner-profile") == Some("hosted-bare"))
                || matches!(
                    text(job, "runs-on"),
                    Some("ubuntu-24.04" | "ubuntu-24.04-arm")
                )
        })
}

#[test]
fn explicit_preparation_profiles_have_a_classified_runner_path() {
    let directory = super::support::root().join(".github/workflows");
    let mut literal_count = 0;
    let mut dynamic = BTreeSet::new();
    for entry in std::fs::read_dir(directory).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if !file.ends_with(".yml") && !file.ends_with(".yaml") {
            continue;
        }
        let workflow = document(&file);
        assert!(literal_hosted_jobs_are_hosted(&workflow), "{file}");
        for (job_name, definition) in workflow.get("jobs").unwrap().entries() {
            let Some(Node::Seq(job_steps)) = definition.get("steps") else {
                continue;
            };
            for step in job_steps {
                let Some(profile) = input(step, "runner-profile") else {
                    continue;
                };
                if profile == "hosted-bare" {
                    literal_count += 1;
                } else {
                    assert!(
                        dynamic.insert((file.clone(), job_name.clone(), profile.to_owned())),
                        "duplicate explicit profile in {file}/{job_name}"
                    );
                }
            }
        }
    }
    assert!(
        literal_count >= 10,
        "hosted profile census must not be empty"
    );
    let expected = [
        ("product-automation-artifact.yml", "producer", LINUX_PROFILE),
        (
            "protected-automation-artifact.yml",
            "producer",
            LINUX_PROFILE,
        ),
        ("release.yml", "build_native_runtime", LINUX_PROFILE),
        (
            "llama-upstream-canary.yml",
            "preflight",
            "${{ needs.resolve.outputs.runner_profile }}",
        ),
        (
            "llama-canary-family-pass.yml",
            "build",
            "${{ inputs.runner_profile }}",
        ),
        (
            "llama-canary-family-pass.yml",
            "family",
            "${{ inputs.runner_profile }}",
        ),
        (
            "llama-canary-family-pass.yml",
            "retry_family",
            "${{ inputs.runner_profile }}",
        ),
    ]
    .into_iter()
    .map(|(file, job, profile)| (file.to_owned(), job.to_owned(), profile.to_owned()))
    .collect();
    assert_eq!(
        dynamic, expected,
        "new dynamic profiles require an explicit decision"
    );

    let product = document("product-automation-artifact.yml");
    assert_eq!(
        text(job(&product, "producer"), "runs-on"),
        Some(
            "${{ inputs.platform == 'macos-arm64' && 'macos-15' || inputs.platform == 'windows-x64' && 'windows-2022' || 'ubuntu-24.04' }}"
        )
    );
    let authority = document("protected-automation-artifact.yml");
    assert_eq!(
        text(job(&authority, "producer"), "runs-on"),
        Some(
            "${{ inputs.platform == 'linux-arm64' && 'ubuntu-24.04-arm' || inputs.platform == 'macos-arm64' && 'macos-15' || inputs.platform == 'windows-x64' && 'windows-2022' || 'ubuntu-24.04' }}"
        )
    );

    let release = document("release.yml");
    let metadata = job(&release, "metadata");
    let selector = steps(metadata)
        .iter()
        .find(|step| text(step, "id") == Some("runners"))
        .unwrap();
    assert_eq!(input(selector, "force_hosted"), Some("true"));
    for name in ["runner_8", "runner_arm_8"] {
        assert_eq!(
            text(metadata.get("outputs").unwrap(), name),
            Some(format!("${{{{ steps.runners.outputs.{name} }}}}").as_str())
        );
    }
    let runtime = job(&release, "build_native_runtime");
    assert_eq!(
        text(runtime, "runs-on"),
        Some(
            "${{ matrix.target == 'x86_64-unknown-linux-gnu' && needs.metadata.outputs.runner_8 || matrix.target == 'aarch64-unknown-linux-gnu' && needs.metadata.outputs.runner_arm_8 || matrix.os }}"
        )
    );
    let Node::Seq(rows) = runtime
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("include")
        .unwrap()
    else {
        panic!("release runtime rows")
    };
    let pairs = rows
        .iter()
        .map(|row| (text(row, "target").unwrap(), text(row, "os").unwrap()))
        .collect::<BTreeSet<_>>();
    assert_eq!(rows.len(), pairs.len(), "duplicate release runtime row");
    assert_eq!(
        pairs,
        BTreeSet::from([
            ("aarch64-apple-darwin", "macos-15"),
            ("x86_64-unknown-linux-gnu", "ubuntu-24.04"),
            ("aarch64-unknown-linux-gnu", "ubuntu-24.04-arm"),
        ])
    );

    let action = super::workflow_yaml::parse(
        &std::fs::read_to_string(
            super::support::root().join(".github/actions/prepare-automation/action.yml"),
        )
        .unwrap(),
    )
    .unwrap();
    let admission = named(action.get("runs").unwrap(), "Validate preparation profile");
    let script = text(admission, "run").unwrap();
    assert!(script.contains("test \"$RUNNER_OS\" = Linux"));
    assert!(script.contains("test \"$RUNNER_ENVIRONMENT\" = github-hosted"));
    super::assert_mutation_rejected(
        "docker-precheck.yml",
        "runs-on: ubuntu-24.04",
        "runs-on: depot-ubuntu-24.04",
        literal_hosted_jobs_are_hosted,
    );
}
