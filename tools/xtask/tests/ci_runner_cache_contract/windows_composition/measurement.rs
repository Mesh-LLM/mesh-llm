use super::super::{
    cache_predicate, support,
    workflow_yaml::{self, Node},
};
use std::{collections::BTreeSet, fs};

const WORKFLOW: &str = ".github/workflows/windows-abi-measurement.yml";
const WARM_ONLY: &str = "${{ inputs.condition == 'exact-warm' && 'true' || 'false' }}";

fn workflow(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(support::root().join(path)).unwrap()).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn steps(node: &Node) -> &[Node] {
    match node.get("steps") {
        Some(Node::Seq(steps)) => steps,
        _ => &[],
    }
}
fn action_steps(node: &Node) -> &[Node] {
    node.get("runs").map_or(&[], steps)
}
fn calls<'a>(job: &'a Node, uses: &str) -> Vec<&'a Node> {
    steps(job)
        .iter()
        .filter(|s| text(s, "uses") == uses)
        .collect()
}
fn input<'a>(step: &'a Node, key: &str) -> &'a str {
    step.get("with").map_or("", |with| text(with, key))
}

/// Every step reachable from the diagnostic, through called workflows and local actions.
fn reachable() -> Vec<(String, Node)> {
    let mut found = Vec::new();
    let mut seen = BTreeSet::new();
    let mut pending: Vec<(String, Vec<Node>)> = Vec::new();
    let mut workflows = vec![WORKFLOW.to_owned()];
    while let Some(path) = workflows.pop() {
        for (_, job) in workflow(&path).get("jobs").unwrap().entries() {
            if let Some(callee) = text(job, "uses").strip_prefix("./") {
                workflows.push(callee.to_owned());
            }
            pending.push((path.clone(), steps(job).to_vec()));
        }
    }
    while let Some((owner, list)) = pending.pop() {
        for step in list {
            if let Some(name) = text(&step, "uses").strip_prefix("./.github/actions/")
                && seen.insert(name.to_owned())
            {
                pending.push((
                    name.to_owned(),
                    action_steps(&support::action(name)).to_vec(),
                ));
            }
            found.push((owner.clone(), step));
        }
    }
    found
}

#[test]
fn windows_abi_measurement_is_manual_read_only_and_secret_free() {
    let source = fs::read_to_string(support::root().join(WORKFLOW)).unwrap();
    let document = workflow(WORKFLOW);
    let triggers: Vec<_> = document
        .get("on")
        .unwrap()
        .entries()
        .iter()
        .map(|(key, _)| key.as_str())
        .collect();
    assert_eq!(triggers, ["workflow_dispatch"]);
    for token in [
        "secrets.",
        "secrets:",
        "id-token",
        "original_event_name",
        "write",
        "self-hosted",
        "depot-",
        "DEPOT_",
    ] {
        assert!(
            !source.contains(token),
            "diagnostic must not contain {token}"
        );
    }
    let read = "contents: read";
    let permissions = document.get("permissions").unwrap();
    assert_eq!(permissions.entries().len(), 1);
    assert_eq!(text(permissions, "contents"), "read", "{read}");
    let jobs = document.get("jobs").unwrap();
    for (name, job) in jobs.entries() {
        let granted = job.get("permissions").unwrap();
        assert_eq!(granted.entries().len(), 1, "{name}");
        assert_eq!(text(granted, "contents"), "read", "{name}");
    }
    let producer = jobs.get("automation").unwrap();
    assert_eq!(
        text(producer, "uses"),
        "./.github/workflows/product-automation-artifact.yml"
    );
    assert_eq!(input(producer, "platform"), "windows-x64");
    assert_eq!(input(producer, "product_source_sha"), "${{ github.sha }}");
    assert_eq!(
        text(jobs.get("measure").unwrap(), "runs-on"),
        "windows-2022"
    );
    let dispatch = document
        .get("on")
        .unwrap()
        .get("workflow_dispatch")
        .unwrap();
    for (name, options) in [
        ("backend", vec!["cpu", "cuda", "rocm", "vulkan"]),
        ("condition", vec!["cold", "exact-warm"]),
        ("sample", vec!["1", "2", "3", "4", "5"]),
    ] {
        let field = dispatch.get("inputs").unwrap().get(name).unwrap();
        assert_eq!(text(field, "type"), "choice", "{name}");
        assert_eq!(field.get("options").unwrap().list(), options, "{name}");
    }
}

#[test]
fn windows_abi_measurement_is_unreachable_from_other_workflows() {
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        if path.ends_with("windows-abi-measurement.yml") {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains("windows-abi-measurement"),
            "{} reaches the diagnostic",
            path.display()
        );
    }
}

#[test]
fn windows_abi_measurement_reaches_no_cache_save_prune_or_publisher() {
    let mut gated = 0;
    for (owner, step) in reachable() {
        let uses = text(&step, "uses");
        let run = text(&step, "run");
        for token in [
            "windows-cache-maintenance",
            "gh api",
            "--execute",
            "gh release",
        ] {
            assert!(!run.contains(token), "{owner}: {token}");
        }
        assert_ne!(
            uses, "./.github/actions/save-and-verify-actions-cache",
            "{owner}"
        );
        assert!(!uses.contains("release"), "{owner}: {uses}");
        if !(uses.starts_with("actions/cache/save@") || uses.starts_with("actions/cache@")) {
            continue;
        }
        let predicate = match owner.as_str() {
            "prepare-automation" => "inputs.runner-profile == 'hosted-bare'",
            "setup-windows-rocm-sdk" => "inputs.allow-native-github-cache == 'true'",
            _ => panic!("{owner}: unexpected cache writer {uses}"),
        };
        assert!(
            cache_predicate::requires(text(&step, "if"), predicate),
            "{owner}: {predicate}"
        );
        gated += 1;
    }
    assert_eq!(gated, 2);
    let prepare = support::action("prepare-automation");
    let validate = action_steps(&prepare)
        .iter()
        .find(|s| text(s, "name") == "Validate preparation profile")
        .unwrap();
    assert!(text(validate, "run").contains("hosted-bare)\n    test \"$RUNNER_OS\" = Linux"));
    for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
        let default = prepare.get("inputs").unwrap().get(flag).unwrap();
        assert_eq!(text(default, "default"), "false", "{flag}");
    }
    let upload = support::action("upload-automation");
    let call = action_steps(&upload)
        .iter()
        .find(|s| text(s, "uses") == "./.github/actions/prepare-automation")
        .unwrap();
    assert!(
        call.get("with")
            .unwrap()
            .get("allow_native_github_cache")
            .is_none()
    );
}

#[test]
fn windows_abi_measurement_binds_every_cache_authority_to_false_except_warm_restore() {
    let document = workflow(WORKFLOW);
    let job = document.get("jobs").unwrap().get("measure").unwrap();
    let restore = calls(job, "./.github/actions/restore-windows-abi-cache");
    assert_eq!(restore.len(), 1);
    assert_eq!(input(restore[0], "allow-native-github-cache"), WARM_ONLY);
    let rocm = calls(job, "./.github/actions/setup-windows-rocm-sdk");
    assert_eq!(input(rocm[0], "allow-native-github-cache"), "false");
    for action in [
        "./.github/actions/configure-sccache-gha",
        "./.github/actions/prepare-native-runtime-input",
    ] {
        let found = calls(job, action);
        assert_eq!(found.len(), 1, "{action}");
        for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
            assert_eq!(input(found[0], flag), "false", "{action} {flag}");
        }
    }
    let cuda = steps(job)
        .iter()
        .find(|s| text(s, "uses").starts_with("Jimver/cuda-toolkit@"))
        .unwrap();
    assert_eq!(input(cuda, "use-github-cache"), "false");
    let vulkan = steps(job)
        .iter()
        .find(|s| text(s, "uses").starts_with("jakoch/install-vulkan-sdk-action@"))
        .unwrap();
    assert_eq!(input(vulkan, "cache"), "false");
    assert_eq!(input(vulkan, "cache_save_if"), "false");
    assert_eq!(
        text(document.get("env").unwrap(), "SCCACHE_GHA_ENABLED"),
        "false"
    );
    let action = support::action("restore-windows-abi-cache");
    let cache_steps: Vec<_> = action_steps(&action)
        .iter()
        .map(|s| text(s, "uses"))
        .filter(|uses| uses.starts_with("actions/cache"))
        .collect();
    assert_eq!(
        cache_steps,
        ["actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25"]
    );
}

#[test]
fn windows_abi_measurement_orders_lookup_admission_preparation_and_receipt() {
    let document = workflow(WORKFLOW);
    let job = document.get("jobs").unwrap().get("measure").unwrap();
    let position = |predicate: &dyn Fn(&Node) -> bool| {
        let found: Vec<_> = steps(job)
            .iter()
            .enumerate()
            .filter(|(_, s)| predicate(s))
            .map(|(index, _)| index)
            .collect();
        assert_eq!(found.len(), 1);
        found[0]
    };
    let runs = |needle: &'static str| move |s: &Node| text(s, "run").contains(needle);
    let uses = |action: &'static str| move |s: &Node| text(s, "uses") == action;
    let order = [
        position(&runs("--phase lookup_start")),
        position(&uses("./.github/actions/restore-windows-abi-cache")),
        position(&runs("--phase lookup_end")),
        position(&runs("windows-abi-measurement admit")),
        position(&runs("--phase prepare_start")),
        position(&uses("./.github/actions/prepare-native-runtime-input")),
        position(&runs("--phase prepare_end")),
        position(&runs("windows-abi-measurement finish")),
    ];
    assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
    let admit = &steps(job)[order[3]];
    let env = admit.get("env").unwrap();
    for (key, value) in [
        (
            "CACHE_INPUT_HASH",
            "${{ steps.llama_cache.outputs.input-hash }}",
        ),
        ("CACHE_HIT", "${{ steps.llama_cache.outputs.cache-hit }}"),
        (
            "CACHE_PRIMARY_KEY",
            "${{ steps.llama_cache.outputs.cache-primary-key }}",
        ),
        ("BUILD_DIR", "${{ steps.llama_cache.outputs.cache-path }}"),
    ] {
        assert_eq!(text(env, key), value, "{key}");
    }
    let admitted = "${{ !cancelled() && steps.admit.outcome == 'success' }}";
    assert_eq!(text(admit, "id"), "admit");
    assert_eq!(text(&steps(job)[order[5]], "id"), "prepare");
    for index in [order[6], order[7]] {
        assert_eq!(text(&steps(job)[index], "if"), admitted, "step {index}");
    }
    let receipt = steps(job)[order[7]].get("env").unwrap();
    assert_eq!(
        text(receipt, "PREPARE_OUTCOME"),
        "${{ steps.prepare.outcome }}"
    );
    let upload = steps(job).last().unwrap();
    assert!(text(upload, "uses").starts_with("actions/upload-artifact@"));
    assert_eq!(input(upload, "retention-days"), "14");
}

#[test]
fn windows_abi_measurement_stays_outside_the_key_it_measures() {
    let action = support::action("restore-windows-abi-cache");
    let identity = action_steps(&action)
        .iter()
        .find(|s| text(s, "id") == "identity")
        .unwrap();
    let hash = text(identity.get("env").unwrap(), "CACHE_INPUT_HASH");
    assert!(!hash.contains("windows-abi-measurement"));
    assert!(!hash.contains(".github/workflows/**"));
}
