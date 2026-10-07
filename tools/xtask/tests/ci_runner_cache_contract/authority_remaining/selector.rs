use super::*;
use crate::cache_callers;
#[test]
fn authority_all_eligible_selector_inputs_are_bounded_and_mutations_refused() {
    let original = workflows();
    cache_callers::check(&original).unwrap();
    let mut count = 0;
    let mut cpu_selectors = 0;
    for (name, document) in &original {
        let Some(policy) = document.get("jobs").and_then(|j| j.get("runner_policy")) else {
            continue;
        };
        let keys: Vec<_> = steps(policy)
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.get("uses").and_then(Node::text) == Some("./.github/actions/select-ci-runners")
            })
            .map(|(index, s)| {
                (
                    index,
                    s.get("with")
                        .unwrap()
                        .entries()
                        .iter()
                        .map(|(key, _)| key.clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        // The owner intentionally admits release's distinct current identity contract separately.
        if name == "release.yml" {
            continue;
        }
        if name == "ci-linux-runtime-slice.yml" {
            cpu_selectors = keys.len();
            assert_eq!(
                keys.iter()
                    .map(|(index, _)| steps(policy)[*index].get("id").and_then(Node::text))
                    .collect::<Vec<_>>(),
                [Some("policy"), Some("cpu_policy")]
            );
        }
        for (index, fields) in keys {
            count += 1;
            for key in fields {
                let mut changed = original.clone();
                let Node::Map(top) = changed.get_mut(name).unwrap() else {
                    panic!()
                };
                let (_, jobs) = top.iter_mut().find(|(k, _)| k == "jobs").unwrap();
                let Node::Map(jobs) = jobs else { panic!() };
                let (_, policy) = jobs.iter_mut().find(|(k, _)| k == "runner_policy").unwrap();
                let Node::Map(policy) = policy else { panic!() };
                let (_, Node::Seq(steps)) = policy.iter_mut().find(|(k, _)| k == "steps").unwrap()
                else {
                    panic!()
                };
                replace(
                    &mut steps[index],
                    &["with", &key],
                    Node::Scalar("unbounded-fixture".into()),
                );
                assert!(
                    cache_callers::check(&changed).is_err(),
                    "{name}/{index}/{key}"
                );
            }
        }
    }
    assert_eq!(cpu_selectors, 2); // Ordinary selector plus forced-hosted CPU policy.
    assert_eq!(count - cpu_selectors, 17); // Other sixteen workflows plus Quality sentinel.
    assert_eq!(count, 19);
}
#[test]
fn authority_protected_audit27_keep_exact_pin_and_central_runner_projection() {
    let documents = workflows();
    let mut count = 0;
    for (name, workflow) in &documents {
        for (job_name, job) in workflow.get("jobs").unwrap().entries() {
            let Some(Node::Seq(steps)) = job.get("steps") else {
                continue;
            };
            for step in steps {
                let Some(uses) = step.get("uses").and_then(Node::text) else {
                    continue;
                };
                if !uses.contains("audit-depot-pr-isolation@") {
                    continue;
                }
                count += 1;
                assert_eq!(
                    uses,
                    "Mesh-LLM/mesh-llm/.github/actions/audit-depot-pr-isolation@ed07043b84d720aab30e75ed2f038f7042576f16"
                );
                let with = step.get("with").unwrap();
                let runner = job.get("runs-on").and_then(Node::text).unwrap();
                let inner = runner
                    .strip_prefix("${{ ")
                    .unwrap()
                    .strip_suffix(" }}")
                    .unwrap();
                let expected = protected_depot_projection(name, job_name, inner)
                    .unwrap_or_else(|| panic!("{name}/{job_name}: unbounded runner {inner}"));
                assert_eq!(
                    with.get("depot_selected").and_then(Node::text),
                    Some(expected.as_str()),
                    "{name}/{job_name}"
                );
                assert!(
                    protected_cache_projection(with, name, job_name),
                    "{name}/{job_name}: cache authority projection changed"
                );
                assert_eq!(
                    with.get("original_event_name").and_then(Node::text),
                    Some("${{ inputs.original_event_name }}")
                );
            }
        }
    }
    assert_eq!(count, 27); // Protected retained component census; no hosted authenticity credit.
}
#[test]
fn authority_normal_quality_and_runtime_seed_jobs_keep_current_declared_trust_shape() {
    let documents = workflows();
    for (name, output) in [
        ("quality_contracts", "runner_4"),
        ("rust_fmt", "runner_4"),
        ("cargo_machete", "runner_4"),
        ("rust_clippy", "runner_8"),
        ("cli_docs_sync", "runner_4"),
    ] {
        let target = job(&documents, "ci-quality-slice.yml", name);
        assert_eq!(
            target.get("runs-on").and_then(Node::text),
            Some(format!("${{{{ needs.runner_policy.outputs.{output} }}}}").as_str())
        );
    }
    for name in ["runtime_seed", "runtime_seed_summary"] {
        let target = job(&documents, "depot-canary.yml", name);
        assert_eq!(
            target.get("runs-on").and_then(Node::text),
            Some("ubuntu-24.04")
        );
        let permissions = target.get("permissions").unwrap();
        assert_eq!(
            permissions.entries().len(),
            if name == "runtime_seed" { 3 } else { 2 }
        );
        for key in ["contents", "actions"] {
            assert_eq!(permissions.get(key).and_then(Node::text), Some("read"))
        }
        if name == "runtime_seed" {
            assert_eq!(
                permissions.get("packages").and_then(Node::text),
                Some("read")
            )
        }
        let checkouts: Vec<_> = steps(target)
            .iter()
            .filter(|s| {
                s.get("uses")
                    .and_then(Node::text)
                    .is_some_and(|u| u.starts_with("actions/checkout@"))
            })
            .collect();
        assert_eq!(checkouts.len(), 1);
        let with = checkouts[0].get("with").unwrap();
        assert_eq!(with.entries().len(), 2);
        assert_eq!(
            with.get("ref").and_then(Node::text),
            Some("${{ github.sha }}")
        );
        assert_eq!(
            with.get("persist-credentials").and_then(Node::text),
            Some("false")
        );
    }
}

const CPU_RUNNER: &str = "matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16";
fn protected_depot_projection(workflow: &str, job: &str, inner: &str) -> Option<String> {
    if workflow == "ci-linux-runtime-slice.yml" && job == "linux_runtime" {
        return (inner == CPU_RUNNER).then(||
            "${{ matrix.runtime.backend != 'cpu' && startsWith(needs.runner_policy.outputs.runner_16, 'depot-') }}".into()
        );
    }
    (inner.starts_with("needs.runner_policy.outputs.")
        || inner
            == "fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform]")
        .then(|| format!("${{{{ startsWith({inner}, 'depot-') }}}}"))
}
#[test]
fn protected_cpu_projection_is_exact_and_admitted_only_in_its_declared_context() {
    assert!(
        protected_depot_projection("ci-linux-runtime-slice.yml", "linux_runtime", CPU_RUNNER)
            .is_some()
    );
    for (workflow, job, runner) in [
        (
            "ci-linux-host-slice.yml",
            "linux_runtime",
            CPU_RUNNER.to_owned(),
        ),
        (
            "ci-linux-runtime-slice.yml",
            "other_job",
            CPU_RUNNER.to_owned(),
        ),
        (
            "ci-linux-runtime-slice.yml",
            "linux_runtime",
            CPU_RUNNER.replace("== 'cpu'", "!= 'cpu'"),
        ),
        (
            "ci-linux-runtime-slice.yml",
            "linux_runtime",
            CPU_RUNNER.replace("runner_cpu", "runner_16"),
        ),
        (
            "ci-linux-runtime-slice.yml",
            "linux_runtime",
            "needs.runner_policy.outputs.runner_16".into(),
        ),
    ] {
        assert!(
            protected_depot_projection(workflow, job, &runner).is_none(),
            "{workflow}/{job}/{runner}"
        );
    }
}

fn protected_cache_projection(inputs: &Node, workflow: &str, job: &str) -> bool {
    let cpu = workflow == "ci-linux-runtime-slice.yml" && job == "linux_runtime";
    let expected = if cpu {
        [
            (
                "allow_native_github_cache",
                "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.allow_native_github_cache_cpu || needs.runner_policy.outputs.allow_native_github_cache }}",
            ),
            (
                "allow_depot_remote_cache",
                "${{ matrix.runtime.backend != 'cpu' && needs.runner_policy.outputs.allow_depot_remote_cache }}",
            ),
        ]
    } else {
        [
            (
                "allow_native_github_cache",
                "${{ needs.runner_policy.outputs.allow_native_github_cache }}",
            ),
            (
                "allow_depot_remote_cache",
                "${{ needs.runner_policy.outputs.allow_depot_remote_cache }}",
            ),
        ]
    };
    expected
        .into_iter()
        .all(|(key, value)| inputs.get(key).and_then(Node::text) == Some(value))
}
#[test]
fn protected_cpu_cache_authority_refuses_cross_context_and_unconditional_cache() {
    let documents = workflows();
    let runtime = job(&documents, "ci-linux-runtime-slice.yml", "linux_runtime");
    let audit = steps(runtime)
        .iter()
        .find(|step| {
            step.get("uses")
                .and_then(Node::text)
                .is_some_and(|uses| uses.contains("audit-depot-pr-isolation@"))
        })
        .unwrap();
    let inputs = audit.get("with").unwrap();
    assert!(protected_cache_projection(
        inputs,
        "ci-linux-runtime-slice.yml",
        "linux_runtime"
    ));
    assert!(!protected_cache_projection(
        inputs,
        "ci-linux-host-slice.yml",
        "linux_runtime"
    ));
    assert!(!protected_cache_projection(
        inputs,
        "ci-linux-runtime-slice.yml",
        "other_job"
    ));
    for key in ["allow_native_github_cache", "allow_depot_remote_cache"] {
        let original = inputs.get(key).and_then(Node::text).unwrap();
        for value in [
            "true".into(),
            format!("${{{{ needs.runner_policy.outputs.{key} }}}}"),
            original
                .replace("== 'cpu'", "!= 'cpu'")
                .replace("!= 'cpu'", "== 'cuda'"),
        ] {
            let mut changed = inputs.clone();
            replace(&mut changed, &[key], Node::Scalar(value));
            assert!(
                !protected_cache_projection(
                    &changed,
                    "ci-linux-runtime-slice.yml",
                    "linux_runtime"
                ),
                "{key}"
            );
        }
    }
}
