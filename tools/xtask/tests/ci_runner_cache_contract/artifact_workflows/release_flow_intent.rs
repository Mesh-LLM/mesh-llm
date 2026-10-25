//! Current release producer/consumer graph, not publication or compiler proof.
use super::{Node, document, input, job, named, source, steps, text};
fn action<'a>(node: &'a Node, uses: &str) -> &'a Node {
    let found = steps(node)
        .iter()
        .filter(|s| text(s, "uses") == Some(uses))
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1);
    found[0]
}
fn runs(node: &Node) -> String {
    steps(node)
        .iter()
        .filter_map(|s| text(s, "run"))
        .collect::<Vec<_>>()
        .join("\n")
}
fn require(run: &str, needles: &[&str]) {
    for needle in needles {
        assert!(run.contains(needle), "missing {needle}");
    }
}
#[test]
fn graph_release_ui_has_one_source_bound_producer_and_no_rebuild_consumers() {
    let doc = document("release.yml");
    let artifact = "prepared-release-ui-${{ needs.metadata.outputs.tag }}-${{ needs.metadata.outputs.source_sha }}";
    let producer = job(&doc, "release_ui");
    assert_eq!(
        text(producer, "uses"),
        Some("./.github/workflows/ci-ui-artifact-slice.yml")
    );
    assert_eq!(
        input(producer, "source_sha"),
        Some("${{ needs.metadata.outputs.source_sha }}")
    );
    assert_eq!(
        input(producer, "release_tag"),
        Some("${{ needs.metadata.outputs.tag }}")
    );
    assert_eq!(input(producer, "artifact_name"), Some(artifact));
    assert!(!artifact.starts_with("release-"));
    for (name, build) in [
        ("build", "Build and attest neutral host"),
        ("build_linux_arm64", "Build and attest neutral host"),
        (
            "windows_host_input",
            "Build and attest Windows release host",
        ),
    ] {
        let host = job(&doc, name);
        assert!(host.get("needs").unwrap().list().contains(&"release_ui"));
        let restore = action(host, "./.github/actions/restore-release-ui");
        assert_eq!(input(restore, "artifact_name"), Some(artifact));
        let restore_position = steps(host)
            .iter()
            .position(|s| std::ptr::eq(s, restore))
            .unwrap();
        let build_position = steps(host)
            .iter()
            .position(|s| text(s, "name").is_some_and(|n| n.starts_with("Build and attest")))
            .unwrap();
        assert!(restore_position < build_position, "{build}");
        let build_step = &steps(host)[build_position];
        assert_eq!(input(build_step, "skip_ui"), Some("true"));
        assert!(
            !steps(host)
                .iter()
                .any(|s| text(s, "uses").is_some_and(|u| u.starts_with("pnpm/action-setup@")))
        );
    }
    let swift = job(&doc, "build_swift_sdk_artifact");
    assert!(swift.get("needs").unwrap().list().contains(&"release_ui"));
    assert_eq!(input(swift, "ui_artifact_name"), Some(artifact));
    let publish = job(&doc, "publish");
    assert!(publish.get("needs").unwrap().list().contains(&"release_ui"));
    assert!(runs(publish).contains("--sdk all --skip-build"));
    assert!(
        !steps(publish)
            .iter()
            .any(|s| text(s, "uses").is_some_and(|u| u.starts_with("pnpm/action-setup@")))
    );
    let ui_document = document("ci-ui-artifact-slice.yml");
    let ui = job(&ui_document, "ui_artifact");
    let preparation = steps(ui)
        .iter()
        .position(|s| text(s, "name") == Some("Prepare release UI version"))
        .unwrap();
    let install = steps(ui)
        .iter()
        .position(|s| text(s, "name") == Some("Install UI dependencies"))
        .unwrap();
    assert!(preparation < install);
    let stamp = job(&ui_document, "ui_stamp");
    assert!(stamp.get("needs").unwrap().list().contains(&"ui_artifact"));
    assert_eq!(text(stamp, "if"), Some("${{ inputs.release_tag != '' }}"));
    let bind = named(stamp, "Bind release UI identity and checksums");
    assert!(
        text(bind, "run")
            .unwrap()
            .contains("prepared-input ui-distribution stamp")
    );
    let binding_env = bind.get("env").unwrap();
    assert_eq!(
        text(binding_env, "UI_SOURCE_SHA"),
        Some("${{ inputs.source_sha }}")
    );
    assert_eq!(
        text(binding_env, "UI_RELEASE_TAG"),
        Some("${{ inputs.release_tag }}")
    );
    require(
        &source("swift-sdk-artifact.yml"),
        &[
            "if: ${{ inputs.ui_artifact_name == '' }}",
            "scripts/package-sdk-console-assets.sh --sdk swift --skip-build",
        ],
    );
}
#[test]
fn graph_release_metadata_trust_and_canonical_version_precede_runner_selection() {
    let doc = document("release.yml");
    let metadata = job(&doc, "metadata");
    let items = steps(metadata);
    let guard = items
        .iter()
        .position(|s| text(s, "name") == Some("Require the trusted release ref"))
        .unwrap();
    let checkout = items
        .iter()
        .position(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@")))
        .unwrap();
    let canonical = items
        .iter()
        .position(|s| text(s, "name") == Some("Prepare canonical release source"))
        .unwrap();
    let selector = items
        .iter()
        .position(|s| text(s, "uses") == Some("./.github/actions/select-ci-runners"))
        .unwrap();
    assert!(guard < checkout && checkout < canonical && canonical < selector);
    require(
        text(&items[guard], "run").unwrap(),
        &["workflow_dispatch", "refs/heads/main"],
    );
    require(
        &runs(metadata),
        &[
            "git merge-base --is-ancestor \"$GITHUB_SHA\" refs/remotes/origin/main",
            "Manual release tag already exists and is immutable",
            "git ls-remote --tags origin",
            "Canary release: leaving main unchanged",
            "does not contain its complete version update",
        ],
    );
    let run = text(&items[canonical], "run").unwrap();
    let version = run
        .find("scripts/release-version.sh \"$RELEASE_TAG\"")
        .unwrap();
    let fmt = run[version..].find("cargo fmt --all -- --check").unwrap() + version;
    let whitespace = run[fmt..].find("git diff --check").unwrap() + fmt;
    let stage = run[whitespace..].find("git add --update").unwrap() + whitespace;
    let push = run[stage..]
        .find("git push \"$release_remote\" \"$source_sha:refs/heads/main\"")
        .unwrap()
        + stage;
    assert!(version < fmt && fmt < whitespace && whitespace < stage && stage < push);
    let outputs = metadata.get("outputs").unwrap();
    assert_eq!(
        text(outputs, "source_sha"),
        Some("${{ steps.source.outputs.sha }}")
    );
    assert_eq!(
        text(outputs, "release_notes_base"),
        Some("${{ steps.source.outputs.release_notes_base }}")
    );
    assert!(run.contains("cargo xtool release notes-base \"$RELEASE_TAG\""));
    let publish = job(&doc, "publish");
    let checkout = steps(publish)
        .iter()
        .find(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/checkout@")))
        .unwrap();
    assert_eq!(
        input(checkout, "ref"),
        Some("${{ needs.metadata.outputs.source_sha }}")
    );
    let release = named(publish, "Publish GitHub release");
    assert_eq!(input(release, "generate_release_notes"), Some("true"));
    assert_eq!(
        input(release, "previous_tag"),
        Some("${{ needs.metadata.outputs.release_notes_base }}")
    );
}
#[test]
fn graph_release_selector_and_secret_hosts_keep_approved_effective_runner_authority() {
    let doc = document("release.yml");
    let metadata = job(&doc, "metadata");
    let selector = action(metadata, "./.github/actions/select-ci-runners");
    assert_eq!(input(selector, "ref"), Some("${{ github.ref }}"));
    assert_eq!(
        input(selector, "depot_main_enabled"),
        Some("${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}")
    );
    assert_eq!(
        input(selector, "manual_use_depot"),
        Some("${{ inputs.use_depot == true }}")
    );
    assert!(
        doc.get("on")
            .unwrap()
            .get("workflow_dispatch")
            .unwrap()
            .get("inputs")
            .unwrap()
            .get("use_depot")
            .is_some()
    );
    let selectors = doc
        .get("jobs")
        .unwrap()
        .entries()
        .iter()
        .filter_map(|(_, j)| j.get("steps").map(|_| j))
        .flat_map(steps)
        .filter(|s| text(s, "uses") == Some("./.github/actions/select-ci-runners"))
        .count();
    assert_eq!(selectors, 1);
    assert_eq!(
        text(job(&doc, "compose_linux_arm64_cpu"), "runs-on"),
        Some("${{ needs.metadata.outputs.runner_arm_4 }}")
    );
    assert_eq!(
        text(job(&doc, "smoke_linux_arm64_artifact"), "runs-on"),
        Some("${{ needs.metadata.outputs.runner_arm }}")
    );
    assert_eq!(
        text(job(&doc, "build"), "runs-on"),
        Some("${{ matrix.os }}")
    );
    assert_eq!(
        text(job(&doc, "build_linux_arm64"), "runs-on"),
        Some("ubuntu-24.04-arm")
    );
    assert_eq!(text(job(&doc, "publish"), "runs-on"), Some("ubuntu-24.04"));
    for host in ["build", "build_linux_arm64"] {
        assert!(steps(job(&doc, host)).iter().any(|s| {
            s.get("env")
                .is_some_and(|env| text(env, "RELEASE_ATTESTATION_SIGNING_KEY").is_some())
        }));
        assert!(
            !text(job(&doc, host), "runs-on")
                .unwrap()
                .contains("needs.metadata.outputs.runner")
        );
    }
    for producer in [
        "build_native_runtime_linux_x86_64_rocm",
        "build_native_runtime_linux_x86_64_vulkan",
    ] {
        let j = job(&doc, producer);
        assert_eq!(
            text(j, "runs-on"),
            Some("${{ needs.metadata.outputs.runner_16 }}")
        );
        let cache = steps(j)
            .iter()
            .filter(|s| text(s, "uses") == Some("./.github/actions/configure-sccache-gha"))
            .collect::<Vec<_>>();
        assert_eq!(cache.len(), 1);
        assert_eq!(
            input(cache[0], "allow_depot_remote_cache"),
            Some("${{ needs.metadata.outputs.allow_depot_remote_cache }}")
        );
        assert!(text(cache[0], "if").is_none());
        assert_eq!(
            input(cache[0], "allow_native_github_cache"),
            Some(
                "${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') && 'false' || 'true' }}"
            )
        );
        let archive = steps(j)
            .iter()
            .filter(|s| text(s, "uses").is_some_and(|u| u.starts_with("actions/cache@")))
            .collect::<Vec<_>>();
        assert_eq!(archive.len(), 1);
        assert_eq!(
            text(archive[0], "if"),
            Some("${{ !startsWith(needs.metadata.outputs.runner_16, 'depot-') }}")
        );
    }
}
#[test]
fn graph_release_linux_composers_keep_cpu_tools_readiness_and_no_compiler_cache() {
    let doc = document("release.yml");
    let image = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";
    for name in [
        "compose_linux_aarch64_cuda",
        "compose_linux_cuda",
        "compose_linux_rocm",
        "compose_linux_vulkan",
    ] {
        let j = job(&doc, name);
        assert_eq!(text(j.get("container").unwrap(), "image"), Some(image));
        assert!(runs(j).contains("verify-runner-image public cpu"));
        let compose = action(j, "./.github/actions/compose-product-input");
        assert_eq!(input(compose, "readiness_smoke"), Some("true"));
        assert!(steps(j).iter().any(|s| {
            s.get("env")
                .is_some_and(|env| text(env, "MESH_RELEASE_HOST_PRESTAMPED") == Some("1"))
        }));
        for step in steps(j) {
            let uses = text(step, "uses").unwrap_or("");
            assert_ne!(uses, "./.github/actions/prepare-native-runtime-input");
            assert_ne!(uses, "./.github/actions/configure-sccache-gha");
            assert!(!uses.starts_with("actions/cache@"));
        }
        assert!(!runs(j).contains("cargo build"));
    }
    let cuda = job(&doc, "compose_linux_cuda");
    assert_eq!(
        text(cuda, "runs-on"),
        Some("${{ needs.metadata.outputs.runner_4 }}")
    );
}
#[test]
fn graph_release_sdk_calls_are_typed_and_publish_flat_exact_artifact_outputs() {
    let doc = document("release.yml");
    let native = job(&doc, "build_native_sdk_runtime");
    assert_eq!(
        text(native, "uses"),
        Some("./.github/workflows/native-sdk-artifact.yml")
    );
    for (key, value) in [
        ("profile", "release"),
        ("include_runtime_crate", "true"),
        (
            "produce_static_abi",
            "${{ endsWith(matrix.target, '-unknown-linux-gnu') }}",
        ),
        (
            "static_abi_artifact_name",
            "ci-release-native-sdk-static-abi-${{ matrix.artifact_suffix }}",
        ),
        (
            "artifact_name",
            "release-native-sdk-${{ matrix.artifact_suffix }}",
        ),
        ("runner_size", "8"),
    ] {
        assert_eq!(input(native, key), Some(value));
    }
    assert!(input(native, "runs_on").is_none());
    assert!(input(native, "allow_depot_remote_cache").is_none());
    let swift = job(&doc, "build_swift_sdk_artifact");
    assert_eq!(
        text(swift, "uses"),
        Some("./.github/workflows/swift-sdk-artifact.yml")
    );
    for (key, value) in [
        ("mode", "full"),
        ("artifact_name", "release-swift-sdk"),
        ("max_parallel", "4"),
        ("timeout_minutes", "180"),
        ("release_tag", "${{ needs.metadata.outputs.tag }}"),
        (
            "prepare_release_version",
            "${{ github.event_name == 'workflow_dispatch' }}",
        ),
    ] {
        assert_eq!(input(swift, key), Some(value));
    }
    assert!(input(swift, "macos_runner").is_none());
    let publish = job(&doc, "publish");
    assert!(
        steps(publish)
            .iter()
            .any(|s| input(s, "name") == Some("generated-swift-binding-release-swift-sdk"))
    );
    assert!(runs(publish).contains("install -m 0644 \"$generated_binding\" \"$tracked_binding\""));
    assert_eq!(
        input(named(publish, "Publish GitHub release"), "files"),
        Some("release-artifacts/*")
    );
}
#[test]
fn graph_release_cuda_rows_bind_compiler_version_and_preserve_pascal_target_only_for_cuda12() {
    let doc = document("release.yml");
    for name in [
        "build_native_runtime_linux_aarch64_cuda",
        "build_native_runtime_linux_x86_64_cuda",
    ] {
        let j = job(&doc, name);
        let env = j.get("env").unwrap();
        assert_eq!(
            text(env, "MESH_CUDA_VERSION"),
            Some("${{ matrix.cuda_version }}")
        );
        assert_eq!(
            text(env, "MESH_LLM_CUDA_TOOLKIT_MAJOR"),
            Some("${{ matrix.cuda_major }}")
        );
    }
    let j = job(&doc, "build_native_runtime_linux_x86_64_cuda");
    let Node::Seq(rows) = j
        .get("strategy")
        .unwrap()
        .get("matrix")
        .unwrap()
        .get("include")
        .unwrap()
    else {
        panic!("matrix")
    };
    for row in rows {
        let major = text(row, "cuda_major").unwrap();
        let architectures = text(row, "cuda_architectures").unwrap();
        let cache = text(row, "cuda_architectures_cache").unwrap();
        assert_eq!(cache, architectures.replace(';', "_"));
        if major == "12" {
            assert!(architectures.split(';').any(|a| a == "61"));
        } else {
            assert!(!architectures.split(';').any(|a| a == "61"));
        }
    }
}
#[test]
fn graph_release_smokes_consume_composed_products_and_verified_extraction() {
    let doc = document("release.yml");
    let smoke = job(&doc, "smoke_linux_arm64_artifact");
    let run = runs(smoke);
    require(
        &run,
        &[
            "cargo xtool artifact verify-checksum",
            "cargo xtool artifact extract-tar",
            "scripts/ci-hf-xet-portability-smoke.sh \"$binary\"",
        ],
    );
    assert!(!run.contains("tar -xzf"));
    assert!(!run.contains("sha256sum"));
    let inference = job(&doc, "inference_smoke_tests");
    assert!(input(inference, "runs_on").is_none());
    assert!(input(inference, "runner").is_none());
    let smoke_doc = document("smoke.yml");
    assert_eq!(
        text(
            smoke_doc
                .get("on")
                .unwrap()
                .get("workflow_call")
                .unwrap()
                .get("inputs")
                .unwrap()
                .get("runner")
                .unwrap(),
            "default"
        ),
        Some("ubuntu-24.04")
    );
    let producers = doc
        .get("jobs")
        .unwrap()
        .entries()
        .iter()
        .filter_map(|(_, j)| j.get("steps").map(|_| j))
        .flat_map(steps)
        .filter(|s| input(s, "name") == Some("ci-release-linux-inference-product"))
        .collect::<Vec<_>>();
    assert_eq!(producers.len(), 1);
    let consumer = input(inference, "artifact_name").unwrap();
    assert_eq!(consumer, "ci-release-linux-inference-product");
}

#[test]
fn graph_release_cpu_cache_authority_uses_the_selected_architecture_runner() {
    let doc = document("release.yml");
    let j = job(&doc, "build_native_runtime");
    assert_eq!(
        text(j, "runs-on"),
        Some(
            "${{ matrix.target == 'x86_64-unknown-linux-gnu' && needs.metadata.outputs.runner_8 || matrix.target == 'aarch64-unknown-linux-gnu' && needs.metadata.outputs.runner_arm_8 || matrix.os }}"
        )
    );
    let cache = action(j, "./.github/actions/configure-sccache-gha");
    assert_eq!(
        input(cache, "allow_native_github_cache"),
        Some(
            "${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_8, 'depot-')) || (matrix.target == 'aarch64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) && 'false' || 'true' }}"
        )
    );
    for name in [
        "build_native_runtime_linux_x86_64_rocm",
        "build_native_runtime_linux_x86_64_vulkan",
    ] {
        let cache = action(job(&doc, name), "./.github/actions/configure-sccache-gha");
        assert_eq!(
            input(cache, "allow_native_github_cache"),
            Some(
                "${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') && 'false' || 'true' }}"
            )
        );
    }
}
#[test]
fn graph_release_prerelease_and_notes_base_outputs_bind_actual_native_selection() {
    let doc = document("release.yml");
    let metadata = job(&doc, "metadata");
    let outputs = metadata.get("outputs").unwrap();
    assert_eq!(
        text(outputs, "prerelease"),
        Some("${{ steps.meta.outputs.prerelease }}")
    );
    let meta = steps(metadata)
        .iter()
        .find(|s| text(s, "id") == Some("meta"))
        .unwrap();
    let run = text(meta, "run").unwrap();
    require(
        run,
        &[
            "prerelease=false",
            "[[ \"$version\" == *-* ]]",
            "prerelease=true",
            "echo \"prerelease=$prerelease\"",
        ],
    );
    let source = named(metadata, "Prepare canonical release source");
    let run = text(source, "run").unwrap();
    require(
        run,
        &[
            "git tag --list 'v*'",
            "cargo xtool release notes-base \"$RELEASE_TAG\"",
            "echo \"release_notes_base=$release_notes_base\"",
        ],
    );
    assert_eq!(
        text(outputs, "release_notes_base"),
        Some("${{ steps.source.outputs.release_notes_base }}")
    );
}
#[test]
fn graph_release_shared_producers_have_one_semantic_owner_per_job() {
    let doc = document("release.yml");
    let mut hosts = 0;
    let mut runtimes = 0;
    let mut products = 0;
    for (name, j) in doc.get("jobs").unwrap().entries() {
        if j.get("steps").is_none() {
            continue;
        }
        let actions = steps(j)
            .iter()
            .filter_map(|s| text(s, "uses"))
            .collect::<Vec<_>>();
        let count = |owner: &str| actions.iter().filter(|&&u| u == owner).count();
        let host = count("./.github/actions/prepare-host-input")
            + count("./.github/actions/prepare-windows-host-input");
        let runtime = count("./.github/actions/prepare-native-runtime-input");
        let product = count("./.github/actions/compose-product-input");
        assert!(host <= 1 && runtime <= 1 && product <= 1, "{name}");
        assert!(
            host + runtime + product <= 1,
            "crossed producer responsibilities {name}"
        );
        hosts += host;
        runtimes += runtime;
        products += product;
    }
    assert_eq!((hosts, runtimes, products), (3, 7, 8));
}
