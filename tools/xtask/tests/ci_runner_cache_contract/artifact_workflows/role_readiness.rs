//! Source-only provider-role inventory; no claim about live runner qualification.
use super::{document, input, job, steps, text};
use std::collections::BTreeSet;

#[test]
fn primary_role_selectors_match_the_recorded_trusted_main_roster() {
    let eligible = [
        "ci-quality-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
        "static-abi-artifact.yml",
        "native-sdk-artifact.yml",
    ];
    let retained_hosted = [
        "ci-rust-tests-slice.yml",
        "ci-linux-product-slice.yml",
        "ci-ui-artifact-slice.yml",
        "ci-web-slice.yml",
        "ci-macos-host-slice.yml",
        "ci-macos-runtime-slice.yml",
        "ci-macos-product-slice.yml",
        "ci-windows-host-slice.yml",
        "ci-windows-runtime-slice.yml",
        "ci-windows-product-slice.yml",
        "ci-platform-checks-slice.yml",
        "swift-sdk-artifact.yml",
    ];
    let mut expected = BTreeSet::new();
    for (files, gate) in [
        (
            eligible.as_slice(),
            "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
        ),
        (retained_hosted.as_slice(), "false"),
    ] {
        for file in files {
            assert!(expected.insert(*file));
            let workflow = document(file);
            let selectors = steps(job(&workflow, "runner_policy"))
                .iter()
                .filter(|step| text(step, "id") == Some("policy"))
                .collect::<Vec<_>>();
            assert_eq!(selectors.len(), 1, "{file}");
            let policy = selectors[0];
            assert_eq!(
                text(policy, "uses"),
                Some("./.github/actions/select-ci-runners")
            );
            assert_eq!(input(policy, "depot_main_enabled"), Some(gate), "{file}");
            assert_eq!(
                input(policy, "depot_pr_enabled"),
                Some("${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}"),
                "{file}"
            );
            assert_eq!(
                input(policy, "force_hosted"),
                Some("${{ inputs.force_hosted }}"),
                "{file}"
            );
        }
    }
    let root = super::support::root().join(".github/workflows");
    let observed = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|file| file.ends_with(".yml") || file.ends_with(".yaml"))
        .filter(|file| {
            let workflow = document(file);
            workflow
                .get("jobs")
                .and_then(|jobs| jobs.get("runner_policy"))
                .is_some_and(|policy| {
                    let mut primary = false;
                    for step in steps(policy).iter().filter(|step| {
                        text(step, "uses") == Some("./.github/actions/select-ci-runners")
                    }) {
                        match (file.as_str(), text(step, "id")) {
                            (_, Some("policy")) => primary = true,
                            ("ci-linux-runtime-slice.yml", Some("cpu_policy"))
                            | ("ci-quality-slice.yml", Some("sentinel_policy")) => (),
                            (_, id) => panic!("unclassified runner policy {file}: {id:?}"),
                        }
                    }
                    primary
                })
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(observed, expected.into_iter().map(str::to_owned).collect());

    let runtime = document("ci-linux-runtime-slice.yml");
    let cpu = steps(job(&runtime, "runner_policy"))
        .iter()
        .find(|step| text(step, "id") == Some("cpu_policy"))
        .unwrap();
    assert_eq!(input(cpu, "force_hosted"), Some("true"));
    let release = document("release.yml");
    let selector = steps(job(&release, "metadata"))
        .iter()
        .find(|step| text(step, "id") == Some("runners"))
        .unwrap();
    assert_eq!(input(selector, "force_hosted"), Some("true"));
    for job_name in [
        "release_ui",
        "build_native_sdk_runtime",
        "build_swift_sdk_artifact",
    ] {
        assert_eq!(
            input(job(&release, job_name), "force_hosted"),
            Some("true"),
            "{job_name}"
        );
    }
}
