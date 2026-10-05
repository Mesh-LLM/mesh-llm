use super::super::{
    cache_predicate, support,
    workflow_yaml::{self, Node},
};
use std::fs;
fn doc(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(support::root().join(path)).unwrap()).unwrap()
}
fn text<'a>(node: &'a Node, field: &str) -> &'a str {
    node.get(field)
        .and_then(Node::text)
        .unwrap_or_else(|| panic!("missing scalar {field}"))
}
fn action_step<'a>(node: &'a Node, id: &str) -> &'a Node {
    let Node::Seq(steps) = node.get("runs").unwrap().get("steps").unwrap() else {
        panic!("action steps");
    };
    let selected: Vec<_> = steps
        .iter()
        .filter(|step| step.get("id").and_then(Node::text) == Some(id))
        .collect();
    assert_eq!(selected.len(), 1, "unique step {id}");
    selected[0]
}
#[test]
fn windows_abi_cache_key_binds_every_compatibility_dimension_and_exact_restore() {
    let document = support::action("restore-windows-abi-cache");
    let inputs = document.get("inputs").unwrap();
    let identity = action_step(&document, "identity");
    assert_eq!(text(identity, "shell"), "pwsh");
    let environment = identity.get("env").unwrap();
    for (input, variable) in [
        ("backend", "INPUT_BACKEND"),
        ("build_dir", "INPUT_BUILD_DIR"),
        ("toolchain_epoch", "INPUT_TOOLCHAIN_EPOCH"),
        ("architecture_set", "INPUT_ARCHITECTURE_SET"),
        ("cuda_toolchain_version", "INPUT_CUDA_TOOLCHAIN_VERSION"),
        ("vulkan_toolchain_version", "INPUT_VULKAN_TOOLCHAIN_VERSION"),
        ("rocm_toolchain_version", "INPUT_ROCM_TOOLCHAIN_VERSION"),
    ] {
        assert!(inputs.get(input).is_some());
        assert_eq!(
            text(environment, variable),
            format!("${{{{ inputs.{input} }}}}")
        );
    }
    let run = text(identity, "run");
    for required in [
        "$backend -notin @(\"cpu\", \"cuda\", \"rocm\", \"vulkan\")",
        "$backend -in @(\"cuda\", \"rocm\") -and -not $architectureSet",
        "build_dir must resolve inside GITHUB_WORKSPACE",
        "build_dir must remain outside the replaceable llama.cpp ",
        "toolchain_epoch must match MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
        "cuda-$version-Jimver-v0.2.35",
        "vulkan-$version-jakoch-v1.5.2",
        "rocm-$version",
        "mesh-llm-windows-2022-skippy-abi-$backend-$architectureSet-$toolchain-$toolchainEpoch-$inputHash",
        "Windows ABI cache input hash is missing or invalid",
    ] {
        assert!(run.contains(required), "missing {required}");
    }
    let hash = text(environment, "CACHE_INPUT_HASH");
    for source in [
        ".github/actions/restore-windows-abi-cache/action.yml",
        ".github/actions/save-and-verify-actions-cache/action.yml",
        ".github/actions/resolve-native-toolchain-epoch/action.yml",
        ".github/actions/prepare-native-runtime-input/action.yml",
        ".github/actions/setup-windows-rocm-sdk/action.yml",
        "scripts/build-llama.sh",
        "scripts/prepare-llama.sh",
        "scripts/package-native-runtime.sh",
        "third_party/llama.cpp/upstream.txt",
        "third_party/llama.cpp/patches/**",
        ".github/cache-version.txt",
    ] {
        assert!(hash.contains(&format!("'{source}'")), "unbound {source}");
    }
    assert!(hash.starts_with("${{ hashFiles("));
    let restore = action_step(&document, "restore");
    assert!(text(restore, "uses").starts_with("actions/cache/restore@"));
    assert!(cache_predicate::requires(
        text(restore, "if"),
        "inputs.allow-native-github-cache == 'true'"
    ));
    let with = restore.get("with").unwrap();
    assert!(
        with.get("restore-keys").is_none(),
        "ABI cache needs exact identity"
    );
    assert_eq!(text(with, "key"), "${{ steps.identity.outputs.cache-key }}");
    assert_eq!(
        text(with, "path"),
        "${{ steps.identity.outputs.build-dir }}"
    );
    let outputs = document.get("outputs").unwrap();
    for (output, value) in [
        ("cache-hit", "${{ steps.restore.outputs.cache-hit }}"),
        (
            "cache-primary-key",
            "${{ steps.restore.outputs.cache-primary-key }}",
        ),
        ("cache-path", "${{ steps.identity.outputs.build-dir }}"),
    ] {
        assert_eq!(text(outputs.get(output).unwrap(), "value"), value);
    }
}
#[test]
fn windows_cache_warmup_keeps_focused_cpu_dispatch_and_verified_shared_producers() {
    let document = doc(".github/workflows/windows-warm-caches.yml");
    let workload = document
        .get("on")
        .unwrap()
        .get("workflow_dispatch")
        .unwrap()
        .get("inputs")
        .unwrap()
        .get("workload")
        .unwrap();
    assert_eq!(text(workload, "type"), "choice");
    assert_eq!(text(workload, "default"), "all");
    let options = workload.get("options").unwrap().list();
    assert!(options.contains(&"all") && options.contains(&"cpu"));
    let jobs = document.get("jobs").unwrap();
    for (name, scope) in [
        (
            "warm_windows_cpu",
            "github.event_name != 'workflow_dispatch' || inputs.workload == 'all' || inputs.workload == 'cpu'",
        ),
        (
            "warm_windows_gpu",
            "github.event_name != 'workflow_dispatch' || inputs.workload == 'all'",
        ),
    ] {
        let job = jobs.get(name).unwrap();
        assert_eq!(text(job, "runs-on"), "windows-2022");
        assert!(
            cache_predicate::requires(text(job, "if"), scope),
            "{name}: workload must be required conjunct"
        );
        assert!(cache_predicate::requires(
            text(job, "if"),
            "github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'"
        ));
        let Node::Seq(steps) = job.get("steps").unwrap() else {
            panic!("warm steps");
        };
        for action in [
            "resolve-native-toolchain-epoch",
            "restore-windows-abi-cache",
            "prepare-native-runtime-input",
            "save-and-verify-actions-cache",
        ] {
            assert!(
                steps.iter().any(|s| s.get("uses").and_then(Node::text)
                    == Some(format!("./.github/actions/{action}").as_str())),
                "{name}: shared {action} owner"
            );
        }
        let verification = steps
            .iter()
            .filter_map(|s| s.get("run").and_then(Node::text))
            .find(|run| run.contains("prepared-input static-abi-stamp"))
            .unwrap();
        for token in [
            "--link-mode dynamic",
            "--stamp-version 3",
            "--toolchain-epoch",
            "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            "$LASTEXITCODE",
        ] {
            assert!(verification.contains(token), "{name}: {token}");
        }
    }
}

#[test]
fn neutral_host_action_uses_canonical_host_builder_without_runtime_or_product_composition() {
    let document = support::action("prepare-host-input");
    let prepare = action_step(&document, "prepare");
    let run = text(prepare, "run");
    for token in [
        "scripts/build-host.sh --profile \"$INPUT_PROFILE\"",
        "cargo xtool native verify-host-dependencies",
    ] {
        assert!(
            run.contains(token),
            "missing canonical host contract: {token}"
        );
    }
    for token in [
        "package-native-runtime.sh",
        "compose-product-bundle",
        "ci-compose-product-input",
    ] {
        assert!(!run.contains(token), "neutral host must not own {token}");
    }
}
