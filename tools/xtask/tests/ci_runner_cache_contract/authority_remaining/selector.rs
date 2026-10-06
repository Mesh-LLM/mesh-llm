use super::*;
use crate::cache_callers;
#[test]
fn authority_all_eligible_selector_inputs_are_bounded_and_mutations_refused() {
    let original = workflows();
    cache_callers::check(&original).unwrap();
    let mut count = 0;
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
    assert_eq!(count, 18); // Seventeen eligible workflows plus Quality's dedicated sentinel.
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
                assert!(
                    inner.starts_with("needs.runner_policy.outputs.")
                        || inner
                            == "fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform]"
                );
                assert_eq!(
                    with.get("depot_selected").and_then(Node::text),
                    Some(format!("${{{{ startsWith({inner}, 'depot-') }}}}").as_str()),
                    "{name}/{job_name}"
                );
                for key in ["allow_native_github_cache", "allow_depot_remote_cache"] {
                    assert_eq!(
                        with.get(key).and_then(Node::text),
                        Some(format!("${{{{ needs.runner_policy.outputs.{key} }}}}").as_str())
                    )
                }
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
