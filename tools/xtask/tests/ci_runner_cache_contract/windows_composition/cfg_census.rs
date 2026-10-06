use super::super::{
    support,
    workflow_yaml::{self, Node},
};
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    process::Command,
};

const UNVERIFIED: &[&str] = &[
    "mesh-llm-hardware-profile",
    "mesh-llm-native-runtime",
    "model-hf",
    // Native HF publication input custody and model FIFO fixtures need Windows qualification.
    "model-package",
    "mesh-llm-routing",
    "skippy-bench",
    "skippy-model",
    "skippy-model-package",
    "skippy-quantize",
    "skippy-runtime",
    "skippy-server",
    "xtask",
];

fn predicate(tokens: TokenStream) -> bool {
    let items: Vec<_> = tokens.into_iter().collect();
    items.iter().enumerate().any(|(index, token)| match token {
        TokenTree::Ident(name) if name == "windows" || name == "unix" || name == "target_os" => true,
        TokenTree::Ident(name) if name == "target_family" => {
            matches!(items.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '=')
                && matches!(items.get(index + 2), Some(TokenTree::Literal(value))
                    if matches!(syn::parse_str::<syn::Lit>(&value.to_string()),
                        Ok(syn::Lit::Str(text)) if text.value() == "windows" || text.value() == "unix"))
        }
        TokenTree::Group(group) => predicate(group.stream()),
        _ => false,
    })
}

fn divergent_tokens(tokens: TokenStream) -> bool {
    let items: Vec<_> = tokens.into_iter().collect();
    for (index, token) in items.iter().enumerate() {
        if let TokenTree::Ident(name) = token
            && (name == "cfg" || name == "cfg_attr")
        {
            let next = if matches!(items.get(index+1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
            {
                index + 2
            } else {
                index + 1
            };
            if let Some(TokenTree::Group(group)) = items.get(next)
                && group.delimiter() == Delimiter::Parenthesis
            {
                let tokens = if name == "cfg_attr" {
                    group
                        .stream()
                        .into_iter()
                        .take_while(|t| !matches!(t, TokenTree::Punct(p) if p.as_char() == ','))
                        .collect()
                } else {
                    group.stream()
                };
                if predicate(tokens) {
                    return true;
                }
            }
        }
        if let TokenTree::Group(group) = token
            && divergent_tokens(group.stream())
        {
            return true;
        }
    }
    false
}
fn divergent(source: &str) -> bool {
    divergent_tokens(source.parse().expect("Rust lexical source must parse"))
}
fn packages(root: &Path) -> BTreeMap<String, String> {
    let manifest: toml::Value =
        toml::from_str(&fs::read_to_string(root.join("Cargo.toml")).unwrap()).unwrap();
    let mut owners = BTreeMap::new();
    for member in manifest["workspace"]["members"].as_array().unwrap() {
        let member = member.as_str().unwrap();
        assert!(
            !member.contains(['*', '?', '[']),
            "review expanded workspace member grammar: {member}"
        );
        let package: toml::Value =
            toml::from_str(&fs::read_to_string(root.join(member).join("Cargo.toml")).unwrap())
                .unwrap();
        assert!(
            owners
                .insert(
                    package["package"]["name"].as_str().unwrap().to_owned(),
                    member.to_owned()
                )
                .is_none()
        );
    }
    owners
}
fn source_divergence(path: &Path) -> bool {
    match fs::symlink_metadata(path) {
        Ok(metadata) => assert!(
            !metadata.file_type().is_symlink(),
            "source census refuses symlink root"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return false,
        Err(error) => panic!("source root metadata failed: {error}"),
    }
    if !path.is_dir() {
        return false;
    }
    let mut found = false;
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        assert!(
            !entry.file_type().unwrap().is_symlink(),
            "source census refuses symlink"
        );
        let path = entry.path();
        if path.is_dir() {
            found |= source_divergence(&path);
        } else if path.extension().is_some_and(|e| e == "rs") {
            found |= divergent(&fs::read_to_string(&path).unwrap());
        }
    }
    found
}
fn workflow() -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(support::root().join(".github/workflows/ci-platform-checks-slice.yml"))
            .unwrap(),
    )
    .unwrap()
}
fn steps(document: &Node) -> &[Node] {
    let Node::Seq(steps) = document
        .get("jobs")
        .unwrap()
        .get("platform_checks")
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("steps");
    };
    steps
}
fn unit_owners(platform: &str) -> BTreeSet<String> {
    let document = workflow();
    let mut owners = BTreeSet::new();
    for step in steps(&document) {
        let uses = step.get("uses").and_then(Node::text).unwrap_or_default();
        if !uses.starts_with("Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages@") {
            continue;
        }
        let condition = step.get("if").and_then(Node::text).unwrap_or_default();
        if !condition.contains("kind == 'unit'") {
            continue;
        }
        if condition.contains("platform == '")
            && !condition.contains(&format!("platform == '{platform}'"))
        {
            continue;
        }
        let crates: Vec<String> = serde_json::from_str(
            step.get("with")
                .unwrap()
                .get("crates")
                .and_then(Node::text)
                .unwrap(),
        )
        .unwrap();
        owners.extend(crates);
    }
    assert!(
        !owners.is_empty(),
        "{platform} unit row must resolve owners"
    );
    owners
}
fn catalog(domain: &str) -> BTreeSet<String> {
    let doc: serde_json::Value =
        serde_json::from_slice(&fs::read(support::root().join("ci/ownership.yml")).unwrap())
            .unwrap();
    doc["crate_rules"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|rule| rule["domain"].as_str().unwrap().starts_with(domain))
        .flat_map(|rule| {
            rule["crates"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| name.as_str().unwrap().to_owned())
        })
        .collect()
}
#[test]
fn platform_cfg_detector_uses_rust_tokens_and_only_cfg_predicates() {
    for source in [
        "#[cfg(windows)] fn p() {}",
        "#[cfg(unix)] fn p() {}",
        "#[cfg(not(windows))] fn p() {}",
        "#[cfg(target_os = \"linux\")] fn p() {}",
        "#[cfg(target_os = \"macos\")] fn p() {}",
        "const P: bool = cfg!(windows);",
        "#[cfg_attr(unix, derive(Debug))] struct P;",
        "#[cfg(any(windows, unix))] fn p() {}",
        "#[cfg(all(not(test), windows))] fn p() {}",
        "const P: bool = cfg ! (all(not(test), unix));",
        "#[cfg(target_family = \"unix\")] fn p() {}",
        "#[cfg(target_family = \"windows\")] fn p() {}",
        "#[cfg(target_os = \"windows\")] fn p() {}",
    ] {
        assert!(divergent(source), "{source}");
    }
    for source in [
        "#[cfg(feature = \"windows\")] fn p() {}",
        "#[cfg(test)] fn windows_path() {}",
        "const P: &str = r#\"#[cfg(windows)]\"#;",
        "// #[cfg(windows)]\n/* nested /* cfg!(unix) */ comment */ fn p() {}",
        "#![cfg_attr(not(debug_assertions), windows_subsystem = \"windows\")]",
        "#[cfg(target_family = \"wasm\")] fn p() {}",
    ] {
        assert!(!divergent(source), "{source}");
    }
}
#[test]
fn live_divergent_workspace_is_routed_or_explicitly_unverified_without_stale_exceptions() {
    let root = support::root();
    let packages = packages(&root);
    let divergent: BTreeSet<_> = packages
        .iter()
        .filter(|(_, path)| source_divergence(&root.join(path).join("src")))
        .map(|(name, _)| name.clone())
        .collect();
    let windows = unit_owners("windows");
    let routed: BTreeSet<_> = catalog("platform-windows")
        .union(&windows)
        .cloned()
        .collect();
    let unverified: BTreeSet<_> = UNVERIFIED.iter().map(|s| (*s).to_owned()).collect();
    assert!(
        unverified.is_subset(&packages.keys().cloned().collect()),
        "unknown unverified owner"
    );
    assert!(
        unverified.is_subset(&divergent),
        "nondivergent unverified owner"
    );
    assert!(
        unverified.is_disjoint(&routed),
        "already routed unverified owner"
    );
    assert!(
        divergent
            .difference(&routed)
            .all(|name| unverified.contains(name)),
        "unaccounted cfg owner: {:?}",
        divergent
            .difference(&routed)
            .filter(|n| !unverified.contains(*n))
            .collect::<Vec<_>>()
    );
    assert!(catalog("platform-windows-cfg").is_subset(&windows));
    assert!(
        windows.is_subset(&packages.keys().cloned().collect()),
        "unknown Windows workflow unit owner"
    );
    // Catalog domains control per-PR selection; complete workflow owners run on
    // exhaustive main/manual profiles. The actual planner tests below bind both.
}
#[test]
fn windows_unit_resolver_outputs_keep_shared_macos_owners_and_windows_only_plugin() {
    let windows = unit_owners("windows");
    let macos = unit_owners("macos");
    for owner in ["model-artifact", "mesh-llm-host-runtime", "mesh-llm"] {
        assert!(windows.contains(owner) && macos.contains(owner));
    }
    assert!(windows.contains("mesh-llm-plugin") && !macos.contains("mesh-llm-plugin"));
    let document = workflow();
    let step = steps(&document)
        .iter()
        .find(|s| s.get("name").and_then(Node::text) == Some("Run Windows unit tests"))
        .unwrap();
    let environment = step.get("env").unwrap();
    assert_eq!(
        environment.get("WINDOWS_TEST_CRATES").and_then(Node::text),
        Some("${{ steps.windows_packages.outputs.crates }}")
    );
    assert_eq!(
        environment.get("TEST_CRATES").and_then(Node::text),
        Some("${{ steps.packages.outputs.crates }}")
    );
    let run = step.get("run").and_then(Node::text).unwrap();
    for token in [
        "$env:TEST_CRATES | ConvertFrom-Json",
        "$env:WINDOWS_TEST_CRATES | ConvertFrom-Json",
        "foreach ($crate in $crates)",
        "cargo test --locked -p $crate --lib",
        "if ($LASTEXITCODE -ne 0)",
    ] {
        assert!(run.contains(token));
    }
    assert!(!run.contains("foreach ($crate in '"));
}
fn plan_for_workflow_owner(owner: &str, profile: &str) -> serde_json::Value {
    let root = support::root();
    let packages = packages(&root);
    let mut input: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("tools/xtask/tests/fixtures/ci_plan/cases/windows-log-store.json"))
            .unwrap(),
    )
    .unwrap();
    input = input["input"].take();
    input["profile"] = serde_json::json!(profile);
    input["event_name"] = serde_json::json!(match profile {
        "main" => "push",
        "manual-full" => "workflow_dispatch",
        _ => "pull_request",
    });
    input["workspace_packages"] = serde_json::json!(
        packages
            .iter()
            .map(|(name, path)| serde_json::json!({"name":name,"path":path}))
            .collect::<Vec<_>>()
    );
    input["changed_files"] = serde_json::json!([format!("{}/src/lib.rs", packages[owner])]);
    // Empty explicit scope delegates to the real native reverse-dependency owner
    // for PR; exhaustive profiles admit the complete declared workspace themselves.
    input["affected_crates"] = serde_json::json!([]);
    let f = support::Fixture::new();
    let payload = f.path().join("input.json");
    fs::write(&payload, serde_json::to_vec(&input).unwrap()).unwrap();
    let mut command = Command::new("bash");
    command
        .current_dir(&root)
        .args([
            "-euo",
            "pipefail",
            "-c",
            "exec \"$XTASK\" ci plan < \"$INPUT\"",
        ])
        .env("XTASK", env!("CARGO_BIN_EXE_xtask"))
        .env("INPUT", &payload);
    let output = f.run(command);
    assert!(
        output.status.success(),
        "{owner}/{profile}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
#[test]
fn every_current_windows_catalog_unit_owner_selects_the_native_planner_row() {
    let unit_only = catalog("platform-windows-cfg");
    let catalog_owners = catalog("platform-windows");
    let owners = unit_owners("windows");
    assert!(owners.contains("mesh-llm-host-runtime"));
    for profile in ["pr-ready", "main", "manual-full"] {
        for owner in &owners {
            if profile == "pr-ready" && !catalog_owners.contains(owner) {
                continue;
            }
            let plan = plan_for_workflow_owner(owner, profile);
            let rows = plan["matrices"]["platform_checks"].as_array().unwrap();
            assert!(
                rows.iter().any(|row| row["id"] == "windows-unit"),
                "{owner}/{profile}"
            );
            if profile == "pr-ready" && unit_only.contains(owner) {
                assert!(
                    rows.iter().all(|row| row["id"] == "windows-unit"),
                    "{owner}"
                );
                for matrix in ["hosts", "runtime_products"] {
                    assert!(
                        plan["matrices"][matrix]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|row| row["platform"] != "windows"),
                        "{owner}"
                    );
                }
            }
        }
    }
}
#[test]
fn wallet_dependency_closure_preserves_direct_domain_pr_routing_and_full_cadence() {
    let plan = plan_for_workflow_owner("mesh-wallet-lexe", "pr-ready");
    assert_eq!(
        plan["direct_crates"],
        serde_json::json!(["mesh-wallet-lexe"])
    );
    assert!(
        plan["affected_crates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|name| name == "mesh-llm-host-runtime")
    );
    assert!(
        !plan["domains"]
            .as_array()
            .unwrap()
            .iter()
            .any(|domain| domain.as_str().unwrap().starts_with("platform-windows"))
    );
    assert!(
        plan["matrices"]["platform_checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["id"] != "windows-unit")
    );
    assert!(
        plan["matrices"]["rust_tests"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row["crates"].as_array().unwrap())
            .any(|name| name == "mesh-wallet-lexe")
    );
    assert!(unit_owners("windows").contains("mesh-wallet-lexe"));
    for profile in ["main", "manual-full"] {
        assert!(
            plan_for_workflow_owner("mesh-wallet-lexe", profile)["matrices"]["platform_checks"]
                .as_array()
                .unwrap()
                .iter()
                .any(|row| row["id"] == "windows-unit")
        );
    }
}

#[test]
fn source_census_rejects_symlinked_source_root_and_child_entry() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("src");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("lib.rs"), "#[cfg(windows)] fn platform() {}").unwrap();
    let linked = root.path().join("linked");
    std::os::unix::fs::symlink(&source, &linked).unwrap();
    assert!(std::panic::catch_unwind(|| source_divergence(&linked)).is_err());
    fs::remove_file(&linked).unwrap();
    std::os::unix::fs::symlink(root.path().join("absent"), &linked).unwrap();
    assert!(std::panic::catch_unwind(|| source_divergence(&linked)).is_err());
    std::os::unix::fs::symlink(source.join("lib.rs"), source.join("alias.rs")).unwrap();
    assert!(std::panic::catch_unwind(|| source_divergence(&source)).is_err());
    root.close().expect("source census fixture deletion failed");
}
