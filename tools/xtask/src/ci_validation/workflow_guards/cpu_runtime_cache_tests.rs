use super::super::workflow_yaml;
use super::super::{cache_callers, cache_consumers};
use super::*;
use std::{fs, path::Path};

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn source() -> String {
    fs::read_to_string(root().join(".github/workflows/ci-linux-runtime-slice.yml")).unwrap()
}
fn actual() -> BTreeMap<String, Node> {
    BTreeMap::from([(
        "ci-linux-runtime-slice.yml".into(),
        workflow_yaml::parse(&source()).unwrap(),
    )])
}
fn mutated(before: &str, after: &str) -> BTreeMap<String, Node> {
    let source = source();
    assert!(source.contains(before), "missing mutation source: {before}");
    BTreeMap::from([(
        "ci-linux-runtime-slice.yml".into(),
        workflow_yaml::parse(&source.replace(before, after)).unwrap(),
    )])
}
#[test]
fn actual_cpu_cache_preserves_hosted_authority_and_exact_runtime_identity() {
    check(&actual()).unwrap();
    cache_consumers::check(&actual()).unwrap();
}
#[test]
fn cpu_projection_cannot_borrow_ordinary_provider_authority() {
    for (before, after) in [
        (
            "runner_cpu: ${{ steps.cpu_policy.outputs.runner_16 }}",
            "runner_cpu: ${{ steps.policy.outputs.runner_16 }}",
        ),
        (
            "allow_native_github_cache_cpu: ${{ steps.cpu_policy.outputs.allow_native_github_cache }}",
            "allow_native_github_cache_cpu: ${{ steps.policy.outputs.allow_native_github_cache }}",
        ),
        (
            "matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16",
            "needs.runner_policy.outputs.runner_16",
        ),
    ] {
        assert!(check(&mutated(before, after)).is_err(), "accepted: {after}");
    }
}
#[test]
fn cpu_cache_restore_requires_cpu_authority_hosted_placement_and_exact_key() {
    for (before, after) in [
        (
            RESTORE_GATE,
            "${{ needs.runner_policy.outputs.allow_native_github_cache_cpu == 'true' }}",
        ),
        (
            RESTORE_GATE,
            "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.allow_native_github_cache == 'true' }}",
        ),
        ("id: runtime_cache", "id: unrelated_cache"),
        (
            "actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25",
            "actions/cache/restore@main",
        ),
        (
            "key: skippy-runtime-linux-",
            "restore-keys: skippy-runtime-linux-\n          key: skippy-runtime-linux-",
        ),
    ] {
        assert!(check(&mutated(before, after)).is_err(), "accepted: {after}");
    }
    for component in [
        "${{ matrix.runtime.target }}-",
        "${{ matrix.runtime.toolchain_epoch }}-",
        "'skippy/**', ",
        "'Cargo.lock', ",
        "'.github/cache-version.txt'",
    ] {
        let key = KEY.replace(component, "");
        assert!(check(&mutated(KEY, &key)).is_err(), "omitted: {component}");
    }
}
#[test]
fn cpu_cache_save_requires_trusted_main_miss_and_the_restored_primary_key() {
    for clause in [
        "matrix.runtime.backend == 'cpu' && ",
        "github.event_name == 'push' && ",
        "github.ref == 'refs/heads/main' && ",
        "needs.runner_policy.outputs.allow_native_github_cache_cpu == 'true' && ",
        "!startsWith(needs.runner_policy.outputs.runner_cpu, 'depot-') && ",
        " && steps.runtime_cache.outputs.cache-hit != 'true'",
    ] {
        assert!(
            check(&mutated(SAVE_GATE, &SAVE_GATE.replace(clause, ""))).is_err(),
            "omitted: {clause}"
        );
    }
    assert!(
        check(&mutated(
            "key: ${{ steps.runtime_cache.outputs.cache-primary-key }}",
            "key: unrelated-runtime"
        ))
        .is_err()
    );
}
#[test]
fn cpu_cache_publication_cannot_precede_runtime_verification_or_upload() {
    let mut workflows = actual();
    let Node::Map(jobs) = workflows
        .get_mut("ci-linux-runtime-slice.yml")
        .unwrap()
        .get_mut_jobs()
    else {
        panic!("jobs")
    };
    let runtime = &mut jobs
        .iter_mut()
        .find(|(name, _)| name == "linux_runtime")
        .unwrap()
        .1;
    let Node::Map(fields) = runtime else {
        panic!("runtime")
    };
    let Node::Seq(steps) = &mut fields
        .iter_mut()
        .find(|(name, _)| name == "steps")
        .unwrap()
        .1
    else {
        panic!("steps")
    };
    let save = steps
        .iter()
        .position(|step| step.get("name").and_then(Node::text) == Some(SAVE))
        .unwrap();
    let upload = steps
        .iter()
        .position(|step| {
            step.get("name").and_then(Node::text) == Some("Upload immutable Linux runtime input")
        })
        .unwrap();
    steps.swap(save, upload);
    assert!(check(&workflows).is_err());
}
trait JobMap {
    fn get_mut_jobs(&mut self) -> &mut Node;
}
impl JobMap for Node {
    fn get_mut_jobs(&mut self) -> &mut Node {
        let Node::Map(fields) = self else {
            panic!("document")
        };
        &mut fields
            .iter_mut()
            .find(|(name, _)| name == "jobs")
            .unwrap()
            .1
    }
}
#[test]
fn cpu_sccache_consumer_requires_the_backend_specific_flags() {
    for (before, after) in [
        (
            DEPOT,
            "${{ needs.runner_policy.outputs.allow_depot_remote_cache }}",
        ),
        (
            NATIVE,
            "${{ needs.runner_policy.outputs.allow_native_github_cache }}",
        ),
    ] {
        assert!(cache_consumers::check(&mutated(before, after)).is_err());
    }
}
#[test]
fn cpu_selector_is_separate_unconditional_and_forced_hosted() {
    let workflows: BTreeMap<_, _> = fs::read_dir(root().join(".github/workflows"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            matches!(
                path.extension().and_then(|value| value.to_str()),
                Some("yml" | "yaml")
            )
        })
        .map(|path| {
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            let source = fs::read_to_string(path).unwrap();
            let parsed = if name == "llama-upstream-canary.yml" {
                workflow_yaml::parse_resolved_aliases(&source)
            } else {
                workflow_yaml::parse(&source)
            };
            (name, parsed.unwrap())
        })
        .collect();
    cache_callers::check(&workflows).unwrap();
    for (before, after) in [
        ("force_hosted: true", "force_hosted: false"),
        ("id: cpu_policy", "id: policy"),
        (
            "id: cpu_policy",
            "id: cpu_policy\n        if: ${{ success() }}",
        ),
        (
            "id: cpu_policy",
            "id: cpu_policy\n        continue-on-error: true",
        ),
    ] {
        let mut changed = workflows.clone();
        changed.extend(mutated(before, after));
        assert!(cache_callers::check(&changed).is_err(), "accepted: {after}");
    }
}
