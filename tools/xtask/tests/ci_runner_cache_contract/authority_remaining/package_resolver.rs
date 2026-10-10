//! Closed resolver caller custody. Synthetic native pins are tests, never publication identities.
use super::*;
const LEGACY: &str = "Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages@5c8fb4d472bc57058c8153761c561daa77dd5b94";
const AUDIT: &str = "Mesh-LLM/mesh-llm/.github/actions/audit-depot-pr-isolation@ed07043b84d720aab30e75ed2f038f7042576f16";
const EXECUTABLE: &str = "${{ runner.temp }}/immutable-automation-restored/${{ runner.os == 'Windows' && 'xtask.exe' || 'xtask' }}";
const CALLERS: &[(&str, &str, &str)] = &[
    ("ci-rust-tests-slice.yml", "rust_tests", "packages"),
    ("ci-quality-slice.yml", "rust_clippy", "packages"),
    (
        "ci-platform-checks-slice.yml",
        "platform_checks",
        "packages",
    ),
    (
        "ci-platform-checks-slice.yml",
        "platform_checks",
        "windows_packages",
    ),
    (
        "ci-platform-checks-slice.yml",
        "platform_checks",
        "windows_dynamic_packages",
    ),
];
struct NativePublication<'a> {
    resolver: &'a str,
    consumer: &'a str,
}
fn scalar<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn immutable_action(reference: &str, owner: &str) -> bool {
    reference
        .strip_prefix(owner)
        .and_then(|tail| tail.strip_prefix('@'))
        .is_some_and(|pin| {
            pin.len() == 40
                && pin
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}
fn inputs(file: &str, id: &str) -> (&'static str, &'static str, &'static str, &'static str) {
    match (file, id) {
        ("ci-rust-tests-slice.yml", "packages") => (
            "current",
            "${{ toJson(matrix.batch.crates) }}",
            "${{ inputs.rust_tests_matrix }}",
            "",
        ),
        ("ci-quality-slice.yml", "packages") => (
            "current",
            "${{ toJson(matrix.batch.crates) }}",
            "${{ inputs.clippy_matrix }}",
            "",
        ),
        (_, "packages") => (
            "legacy",
            "[\"model-artifact\",\"mesh-llm-host-runtime\",\"mesh-llm\"]",
            "",
            "${{ matrix.check.kind == 'unit' }}",
        ),
        (_, "windows_packages") => (
            "current",
            "[\"mesh-llm-analytics\",\"mesh-llm-config\",\"mesh-llm-events\",\"mesh-llm-identity\",\"mesh-llm-payments\",\"mesh-llm-plugin\",\"mesh-llm-plugin-manager\",\"mesh-llm-tui\",\"mesh-llm-ui\",\"mesh-llm-wallet\",\"skippy-cache\"]",
            "",
            "${{ matrix.check.kind == 'unit' && matrix.check.platform == 'windows' }}",
        ),
        (_, "windows_dynamic_packages") => (
            "legacy",
            "[\"mesh-llm-commands\",\"mesh-llm-system\"]",
            "",
            "${{ matrix.check.kind == 'unit' && matrix.check.platform == 'windows' }}",
        ),
        _ => panic!("closed caller"),
    }
}
fn check(
    documents: &BTreeMap<String, Node>,
    publication: Option<&NativePublication<'_>>,
) -> Result<(), String> {
    if let Some(p) = publication
        && (!immutable_action(
            p.resolver,
            "Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages",
        ) || !immutable_action(
            p.consumer,
            "Mesh-LLM/mesh-llm/.github/actions/audit-pr-authority-verified",
        ))
    {
        return Err("unresolved or mutable native publication".into());
    }
    let mut actual = Vec::new();
    for (file, doc) in documents {
        for (name, target) in doc.get("jobs").ok_or("missing jobs")?.entries() {
            let Some(Node::Seq(entries)) = target.get("steps") else {
                continue;
            };
            for (index, step) in entries.iter().enumerate() {
                if !scalar(step, "uses").contains("resolve-cargo-packages") {
                    continue;
                }
                let caller = (file.as_str(), name.as_str(), scalar(step, "id"));
                if !CALLERS.contains(&caller) {
                    return Err("undeclared resolver caller".into());
                }
                let expected = publication.map_or(LEGACY, |p| p.resolver);
                if scalar(step, "uses") != expected || step.get("continue-on-error").is_some() {
                    return Err("resolver publication or error custody changed".into());
                }
                let (generation, crates, batches, condition) = inputs(file, caller.2);
                let with = step.get("with").ok_or("missing resolver input")?;
                if with.entries().len() != if batches.is_empty() { 2 } else { 3 }
                    || scalar(with, "generation") != generation
                    || scalar(with, "crates") != crates
                    || scalar(with, "batches") != batches
                    || scalar(step, "if") != condition
                {
                    return Err(
                        "resolver generation, batch correlation, owners or platform guard changed"
                            .into(),
                    );
                }
                if let Some(p) = publication {
                    check_native(file, entries, index, p)?
                }
                actual.push(caller);
            }
        }
    }
    actual.sort_unstable();
    let mut expected = CALLERS.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err("five resolver caller census changed".into());
    }
    Ok(())
}
fn check_native(
    file: &str,
    entries: &[Node],
    index: usize,
    p: &NativePublication<'_>,
) -> Result<(), String> {
    let step = &entries[index];
    if step.get("env").is_none_or(|env| {
        env.entries().len() != 1 || scalar(env, "MESH_LLM_AUTOMATION_BIN") != EXECUTABLE
    }) {
        return Err("resolver executable is not the admitted platform binary".into());
    }
    let admissions: Vec<_> = entries
        .iter()
        .enumerate()
        .filter(|(_, s)| scalar(s, "uses") == p.consumer)
        .collect();
    let [(audit, admission)] = admissions.as_slice() else {
        return Err("expected exactly one admission".into());
    };
    let checkout = entries
        .iter()
        .position(|s| scalar(s, "uses").starts_with("actions/checkout@"))
        .ok_or("checkout")?;
    if *audit >= checkout
        || checkout >= index
        || admission.get("if").is_some()
        || admission.get("continue-on-error").is_some()
    {
        return Err("admission must succeed before checkout and resolution".into());
    }
    let identity = if file == "ci-platform-checks-slice.yml" {
        "matrix.check.platform == 'macos' && inputs.authority_macos_arm64 || inputs.authority_windows_x64"
    } else {
        "inputs.authority_linux_x64"
    };
    let with = admission.get("with").ok_or("admission identity")?;
    let depot = match file {
        "ci-platform-checks-slice.yml" => {
            "${{ startsWith(fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform], 'depot-') }}"
        }
        "ci-rust-tests-slice.yml" => {
            "${{ startsWith(needs.runner_policy.outputs.runner, 'depot-') }}"
        }
        "ci-quality-slice.yml" => {
            "${{ startsWith(needs.runner_policy.outputs.runner_8, 'depot-') }}"
        }
        _ => return Err("undeclared native platform".into()),
    };
    if with.entries().len() != 9 || scalar(with, "depot_selected") != depot {
        return Err("runner authority changed".into());
    }
    for (field, expected) in [
        ("original_event_name", "${{ inputs.original_event_name }}"),
        (
            "allow_native_github_cache",
            "${{ needs.runner_policy.outputs.allow_native_github_cache }}",
        ),
        (
            "allow_depot_remote_cache",
            "${{ needs.runner_policy.outputs.allow_depot_remote_cache }}",
        ),
    ] {
        if scalar(with, field) != expected {
            return Err("cache/event authority changed".into());
        }
    }

    for field in [
        "source_sha",
        "artifact_id",
        "binary_sha256",
        "producer_os",
        "producer_arch",
    ] {
        if scalar(with, field) != format!("${{{{ fromJson({identity}).{field} }}}}") {
            return Err(format!("admission identity correlation: {field}"));
        }
    }
    Ok(())
}
fn mutable_steps<'a>(
    docs: &'a mut BTreeMap<String, Node>,
    file: &str,
    name: &str,
) -> &'a mut Vec<Node> {
    let target = docs.get_mut(file).unwrap();
    let Node::Map(top) = target else { panic!() };
    let (_, Node::Map(jobs)) = top.iter_mut().find(|(k, _)| k == "jobs").unwrap() else {
        panic!()
    };
    let (_, Node::Map(job)) = jobs.iter_mut().find(|(k, _)| k == name).unwrap() else {
        panic!()
    };
    let (_, Node::Seq(entries)) = job.iter_mut().find(|(k, _)| k == "steps").unwrap() else {
        panic!()
    };
    entries
}
fn set(node: &mut Node, key: &str, value: Node) {
    let Node::Map(fields) = node else { panic!() };
    if let Some((_, existing)) = fields.iter_mut().find(|(k, _)| k == key) {
        *existing = value
    } else {
        fields.push((key.into(), value))
    }
}
fn mapping_child<'a>(node: &'a mut Node, key: &str) -> &'a mut Node {
    let Node::Map(fields) = node else { panic!() };
    &mut fields.iter_mut().find(|(name, _)| name == key).unwrap().1
}
fn native_fixture(p: &NativePublication<'_>) -> BTreeMap<String, Node> {
    let mut docs = workflows();
    for (file, name, id) in CALLERS {
        let entries = mutable_steps(&mut docs, file, name);
        let admission = entries
            .iter_mut()
            .find(|s| scalar(s, "uses") == AUDIT || scalar(s, "uses") == p.consumer)
            .unwrap();
        set(admission, "uses", Node::Scalar(p.consumer.into()));
        let identity = if *file == "ci-platform-checks-slice.yml" {
            "matrix.check.platform == 'macos' && inputs.authority_macos_arm64 || inputs.authority_windows_x64"
        } else {
            "inputs.authority_linux_x64"
        };
        for field in [
            "source_sha",
            "artifact_id",
            "binary_sha256",
            "producer_os",
            "producer_arch",
        ] {
            set(
                mapping_child(admission, "with"),
                field,
                Node::Scalar(format!("${{{{ fromJson({identity}).{field} }}}}")),
            );
        }
        let resolver = entries.iter_mut().find(|s| scalar(s, "id") == *id).unwrap();
        set(resolver, "uses", Node::Scalar(p.resolver.into()));
        set(
            resolver,
            "env",
            Node::Map(vec![(
                "MESH_LLM_AUTOMATION_BIN".into(),
                Node::Scalar(EXECUTABLE.into()),
            )]),
        );
    }
    docs
}
#[test]
fn current_five_legacy_resolvers_keep_closed_inputs_until_publication() {
    check(&workflows(), None).unwrap();
}
#[test]
fn five_native_resolvers_require_exact_publication_custody_and_inputs() {
    // Inert immutable fixture values; no assertion these commits exist or are published.
    let p = NativePublication {
        resolver: "Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        consumer: "Mesh-LLM/mesh-llm/.github/actions/audit-pr-authority-verified@bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };
    let original = native_fixture(&p);
    check(&original, Some(&p)).unwrap();
    for (file, name, id) in CALLERS {
        for (key, value) in [("generation", "unknown"), ("crates", "[]")] {
            let mut docs = original.clone();
            let step = mutable_steps(&mut docs, file, name)
                .iter_mut()
                .find(|s| scalar(s, "id") == *id)
                .unwrap();
            replace(step, &["with", key], Node::Scalar(value.into()));
            assert!(check(&docs, Some(&p)).is_err(), "{file}/{id}/{key}");
        }
        for value in [
            "${{ env.MESH_LLM_AUTOMATION_BIN }}",
            "${{ github.workspace }}/target/debug/xtask",
            "${{ runner.temp }}/immutable-automation-restored/xtask",
        ] {
            let mut docs = original.clone();
            let step = mutable_steps(&mut docs, file, name)
                .iter_mut()
                .find(|s| scalar(s, "id") == *id)
                .unwrap();
            replace(
                step,
                &["env", "MESH_LLM_AUTOMATION_BIN"],
                Node::Scalar(value.into()),
            );
            assert!(check(&docs, Some(&p)).is_err());
        }
        let mut docs = original.clone();
        let entries = mutable_steps(&mut docs, file, name);
        let audit = entries
            .iter()
            .position(|s| scalar(s, "uses") == p.consumer)
            .unwrap();
        let checkout = entries
            .iter()
            .position(|s| scalar(s, "uses").starts_with("actions/checkout@"))
            .unwrap();
        entries.swap(audit, checkout);
        assert!(check(&docs, Some(&p)).is_err());
        for field in [
            "source_sha",
            "artifact_id",
            "binary_sha256",
            "producer_os",
            "producer_arch",
            "original_event_name",
            "depot_selected",
            "allow_native_github_cache",
            "allow_depot_remote_cache",
        ] {
            let mut docs = original.clone();
            let admission = mutable_steps(&mut docs, file, name)
                .iter_mut()
                .find(|s| scalar(s, "uses") == p.consumer)
                .unwrap();
            replace(
                admission,
                &["with", field],
                Node::Scalar("${{ inputs.unrelated }}".into()),
            );
            assert!(check(&docs, Some(&p)).is_err());
        }
    }
    for (file, name, id) in CALLERS {
        let mut docs = original.clone();
        let entries = mutable_steps(&mut docs, file, name);
        let index = entries.iter().position(|s| scalar(s, "id") == *id).unwrap();
        entries.remove(index);
        assert!(check(&docs, Some(&p)).is_err());
        let mut docs = original.clone();
        let step = mutable_steps(&mut docs, file, name)
            .iter_mut()
            .find(|s| scalar(s, "id") == *id)
            .unwrap();
        set(step, "if", Node::Scalar("false".into()));
        assert!(check(&docs, Some(&p)).is_err());
        for key in ["if", "continue-on-error"] {
            let mut docs = original.clone();
            let admission = mutable_steps(&mut docs, file, name)
                .iter_mut()
                .find(|s| scalar(s, "uses") == p.consumer)
                .unwrap();
            set(admission, key, Node::Scalar("true".into()));
            assert!(check(&docs, Some(&p)).is_err());
        }
        if *file != "ci-platform-checks-slice.yml" {
            let mut docs = original.clone();
            let step = mutable_steps(&mut docs, file, name)
                .iter_mut()
                .find(|s| scalar(s, "id") == *id)
                .unwrap();
            replace(
                step,
                &["with", "batches"],
                Node::Scalar("${{ inputs.unrelated_matrix }}".into()),
            );
            assert!(check(&docs, Some(&p)).is_err());
        }
    }
    let mutable = NativePublication {
        resolver: "Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages@main",
        consumer: p.consumer,
    };
    assert!(check(&original, Some(&mutable)).is_err());
    assert!(check(&workflows(), Some(&p)).is_err()); // Actual caller cutover remains held.
}

#[test]
fn local_resolver_action_keeps_native_argv_and_output_contract() {
    let source = fs::read_to_string(
        support::root().join(".github/actions/resolve-cargo-packages/action.yml"),
    )
    .unwrap();
    for fragment in [
        "test -x \"${MESH_LLM_AUTOMATION_BIN:?prepare or restore automation before resolving packages}\"",
        "args=(--generation \"$PACKAGE_GENERATION\" --crates \"$REQUESTED_CRATES\")",
        "args+=(--batches \"$PLANNED_BATCHES\")",
        "args+=(--cargo \"$(command -v cargo)\")",
        "resolved=$(\"$MESH_LLM_AUTOMATION_BIN\" repository cargo-packages \"${args[@]}\")",
        "printf 'crates=%s\\n' \"$resolved\" >> \"$GITHUB_OUTPUT\"",
    ] {
        assert!(
            source.contains(fragment),
            "missing native action fragment: {fragment}"
        );
    }
    assert!(!source.contains("ci-cargo-packages.py"));
}
