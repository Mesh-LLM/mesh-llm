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
    "skippy-hardware-profile",
    "skippy-native-runtime",
    "skippy-model-hf",
    "skippy-hf-hub",
    // Native HF publication input custody and model FIFO fixtures need Windows qualification.
    "skippy-model-package",
    "mesh-llm-routing",
    "skippy-bench",
    "skippy-model",
    "skippy-package-builder",
    "skippy-quantize",
    "skippy-runtime",
    // Current divergent owners split from the already-unverified skippy-server.
    "skippy-serving",
    "skippy-api",
    "skippy-cli",
    "xtask",
    // Extracted native reader has Unix regular-file admission branches; Windows runtime proof is pending.
    "trajectory-reader",
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
#[path = "cfg_source.rs"]
mod cfg_source;
use cfg_source::{Sites, classify_sites};
fn source_divergence(path: &Path) -> bool {
    source_sites(path).any()
}
fn source_sites(path: &Path) -> Sites {
    match fs::symlink_metadata(path) {
        Ok(metadata) => assert!(
            !metadata.file_type().is_symlink(),
            "source census refuses symlink root"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Sites::default(),
        Err(error) => panic!("source root metadata failed: {error}"),
    }
    if !path.is_dir() {
        return Sites::default();
    }
    let mut found = Sites::default();
    for entry in fs::read_dir(path).unwrap() {
        let entry = entry.unwrap();
        assert!(
            !entry.file_type().unwrap().is_symlink(),
            "source census refuses symlink"
        );
        let path = entry.path();
        if path.is_dir() {
            found.merge(source_sites(&path));
        } else if path.extension().is_some_and(|e| e == "rs") {
            found.merge(classify_sites(&fs::read_to_string(&path).unwrap()));
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
    unit_owner_names(platform, true)
}
fn unit_owner_names(platform: &str, resolve: bool) -> BTreeSet<String> {
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
        owners.extend(if resolve {
            resolve_owners(
                &crates,
                step.get("with")
                    .unwrap()
                    .get("generation")
                    .and_then(Node::text)
                    .unwrap_or("legacy"),
            )
        } else {
            crates
        });
    }
    assert!(
        !owners.is_empty(),
        "{platform} unit row must resolve owners"
    );
    owners
}
fn catalog(domain: &str) -> BTreeSet<String> {
    resolve_owners(
        &catalog_requests(domain).into_iter().collect::<Vec<_>>(),
        "legacy",
    )
    .into_iter()
    .collect()
}
fn catalog_requests(domain: &str) -> BTreeSet<String> {
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
fn admit_catalog_visibility(
    divergent: &BTreeSet<String>,
    windows: &BTreeSet<String>,
    catalog: &BTreeSet<String>,
) -> Result<(), String> {
    let invisible: Vec<_> = divergent
        .intersection(windows)
        .filter(|owner| !catalog.contains(*owner))
        .collect();
    if invisible.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Windows cfg owners lack direct catalog routing: {invisible:?}"
        ))
    }
}

fn admit_runtime_coverage(
    sites: &BTreeMap<String, Sites>,
    routed: &BTreeSet<String>,
    unverified: &BTreeSet<String>,
) -> Result<(), String> {
    let uncovered: Vec<_> = sites
        .iter()
        .filter(|(name, site)| {
            site.runtime && !routed.contains(*name) && !unverified.contains(*name)
        })
        .map(|(name, _)| name)
        .collect();
    if uncovered.is_empty() {
        Ok(())
    } else {
        Err(format!("unaccounted runtime cfg owner: {uncovered:?}"))
    }
}

#[test]
fn windows_row_membership_does_not_replace_direct_cfg_catalog_routing() {
    let names = |values: &[&str]| values.iter().map(|name| (*name).to_owned()).collect();
    let divergent = names(&["shared", "windows-only", "unverified"]);
    let windows = names(&["shared", "windows-only", "platform-neutral"]);
    let mut catalog = names(&["shared", "windows-only"]);
    assert!(admit_catalog_visibility(&divergent, &windows, &catalog).is_ok());
    for owner in ["shared", "windows-only"] {
        catalog.remove(owner);
        let error = admit_catalog_visibility(&divergent, &windows, &catalog).unwrap_err();
        assert!(error.contains(owner));
        catalog.insert(owner.to_owned());
    }
    // Nondivergent row members and divergent owners absent from this row do
    // not create a false direct-routing requirement.
    assert!(admit_catalog_visibility(&divergent, &windows, &catalog).is_ok());
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
    let sites: BTreeMap<_, _> = packages
        .iter()
        .map(|(name, path)| (name.clone(), source_sites(&root.join(path).join("src"))))
        .collect();
    let divergent: BTreeSet<_> = packages
        .iter()
        .filter(|(_, path)| source_divergence(&root.join(path).join("src")))
        .map(|(name, _)| name.clone())
        .collect();
    let windows = unit_owners("windows");
    admit_catalog_visibility(&divergent, &windows, &catalog("platform-windows")).unwrap();
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
    admit_runtime_coverage(&sites, &routed, &unverified).unwrap();
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
    for owner in ["skippy-model-artifact", "mesh-llm-host-runtime", "mesh-llm"] {
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
    let current = if owner == "model-artifact" {
        "skippy-model-artifact"
    } else {
        owner
    };
    input["changed_files"] = serde_json::json!([format!("{}/src/lib.rs", packages[current])]);
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

fn resolve_owners(requested: &[String], generation: &str) -> Vec<String> {
    let root = support::root();
    let owners = packages(&root);
    let fixture = support::Fixture::new();
    let metadata = fixture.path().join("workspace-metadata.json");
    let rows: Vec<_> = owners.iter().map(|(name, path)| serde_json::json!({
        "id":name,"name":name,"version":"0.0.0","manifest_path":root.join(path).join("Cargo.toml")
    })).collect();
    fs::write(
        &metadata,
        serde_json::to_vec(&serde_json::json!({
            "packages":rows,"workspace_members":owners.keys().collect::<Vec<_>>()
        }))
        .unwrap(),
    )
    .unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.current_dir(&root).args([
        "repository",
        "cargo-packages",
        "--generation",
        generation,
        "--crates",
        &serde_json::to_string(requested).unwrap(),
        "--metadata",
        metadata.to_str().unwrap(),
    ]);
    let output = fixture.run(command);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn dynamic_request(document: &Node) -> Result<(&str, Vec<String>), String> {
    let unique = |field: &str, value: &str| {
        let found: Vec<_> = steps(document)
            .iter()
            .filter(|step| step.get(field).and_then(Node::text) == Some(value))
            .collect();
        match found.as_slice() {
            [step] => Ok(*step),
            _ => Err(format!("expected one dynamic Windows step: {value}")),
        }
    };
    let resolver = unique("id", "windows_dynamic_packages")?;
    if !resolver
        .get("uses")
        .and_then(Node::text)
        .is_some_and(|uses| {
            uses.starts_with("Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages@")
        })
    {
        return Err("dynamic Windows owners must use package resolver".into());
    }
    let inputs = resolver
        .get("with")
        .ok_or("missing dynamic resolver inputs")?;
    let generation = inputs
        .get("generation")
        .and_then(Node::text)
        .ok_or("missing package generation")?;
    if !["legacy", "current"].contains(&generation) {
        return Err("unsupported package generation".into());
    }
    let requested: Vec<String> = serde_json::from_str(
        inputs
            .get("crates")
            .and_then(Node::text)
            .ok_or("missing dynamic crate request")?,
    )
    .map_err(|error| error.to_string())?;
    if requested.is_empty() {
        return Err("dynamic Windows owner request must not be empty".into());
    }
    let run = unique("name", "Run Windows unit tests")?;
    if run
        .get("env")
        .and_then(|env| env.get("WINDOWS_DYNAMIC_TEST_CRATES"))
        .and_then(Node::text)
        != Some("${{ steps.windows_dynamic_packages.outputs.crates }}")
    {
        return Err("dynamic Windows loop must consume its resolver output".into());
    }
    let lines: Vec<_> = run
        .get("run")
        .and_then(Node::text)
        .ok_or("missing Windows run")?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let expected = [
        "foreach ($crate in @($env:WINDOWS_DYNAMIC_TEST_CRATES | ConvertFrom-Json)) {",
        "cargo test --locked -p $crate --lib --features mesh-llm-system/dynamic-native-runtime",
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
        "}",
    ];
    let starts: Vec<_> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.contains("WINDOWS_DYNAMIC_TEST_CRATES"))
        .map(|(index, _)| index)
        .collect();
    if starts.len() != 1
        || lines.get(starts[0]..starts[0] + expected.len()) != Some(expected.as_slice())
    {
        return Err("dynamic Windows loop must test every resolved owner with the system feature and exit on failure".into());
    }
    Ok((generation, requested))
}

fn admit_dynamic_dependencies(
    resolved: &[String],
    manifests: &BTreeMap<String, toml::Value>,
) -> Result<(), String> {
    if resolved.is_empty() {
        return Err("dynamic resolver produced no owners".into());
    }
    let feature = manifests
        .get("mesh-llm-system")
        .and_then(|manifest| manifest.get("features"))
        .and_then(|features| features.get("dynamic-native-runtime"))
        .and_then(toml::Value::as_array)
        .ok_or("system dynamic-native-runtime feature missing")?;
    if !feature
        .iter()
        .any(|value| value.as_str() == Some("skippy-runtime/dynamic-native-runtime"))
    {
        return Err("system dynamic-native-runtime feature does not activate the runtime".into());
    }
    for owner in resolved {
        let manifest = manifests
            .get(owner)
            .ok_or_else(|| format!("unknown resolved dynamic owner: {owner}"))?;
        if owner != "mesh-llm-system"
            && manifest
                .get("dependencies")
                .and_then(|dependencies| dependencies.get("mesh-llm-system"))
                .is_none()
        {
            return Err(format!(
                "{owner} lacks direct mesh-llm-system dependency for the dynamic feature"
            ));
        }
    }
    Ok(())
}

#[test]
fn windows_dynamic_resolver_output_admits_each_actual_package_feature_dependency() {
    let document = workflow();
    let (generation, requested) = dynamic_request(&document).unwrap();
    let root = support::root();
    let owners = packages(&root);
    let resolved = resolve_owners(&requested, generation);
    let manifests = owners
        .iter()
        .map(|(name, path)| {
            (
                name.clone(),
                toml::from_str(&fs::read_to_string(root.join(path).join("Cargo.toml")).unwrap())
                    .unwrap(),
            )
        })
        .collect();
    admit_dynamic_dependencies(&resolved, &manifests).unwrap();
}

#[test]
fn windows_dynamic_owner_mutations_refuse_missing_indirect_or_featureless_admission() {
    let mut manifests: BTreeMap<String, toml::Value> = [
        (
            "mesh-llm-system".into(),
            toml::from_str(
                "[features]\ndynamic-native-runtime=['skippy-runtime/dynamic-native-runtime']\n",
            )
            .unwrap(),
        ),
        (
            "direct".into(),
            toml::from_str("[dependencies]\nmesh-llm-system='1'\n").unwrap(),
        ),
        (
            "indirect".into(),
            toml::from_str("[dependencies]\ndirect='1'\n[dev-dependencies]\nmesh-llm-system='1'\n")
                .unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    let resolved = vec!["mesh-llm-system".into(), "direct".into()];
    assert!(admit_dynamic_dependencies(&resolved, &manifests).is_ok());
    for invalid in [vec![], vec!["missing".into()], vec!["indirect".into()]] {
        assert!(admit_dynamic_dependencies(&invalid, &manifests).is_err());
    }
    manifests.get_mut("direct").unwrap()["dependencies"]
        .as_table_mut()
        .unwrap()
        .remove("mesh-llm-system");
    assert!(admit_dynamic_dependencies(&resolved, &manifests).is_err());
    manifests.get_mut("direct").unwrap()["dependencies"]
        .as_table_mut()
        .unwrap()
        .insert("mesh-llm-system".into(), toml::Value::String("1".into()));
    for activation in [
        vec![],
        vec![toml::Value::String("unrelated-feature".into())],
    ] {
        manifests.get_mut("mesh-llm-system").unwrap()["features"]["dynamic-native-runtime"] =
            toml::Value::Array(activation);
        assert!(admit_dynamic_dependencies(&resolved, &manifests).is_err());
    }
    manifests.get_mut("mesh-llm-system").unwrap()["features"]
        .as_table_mut()
        .unwrap()
        .remove("dynamic-native-runtime");
    assert!(admit_dynamic_dependencies(&resolved, &manifests).is_err());
}

#[test]
fn windows_dynamic_loop_mutations_refuse_detached_output_and_unfeatured_execution() {
    let source =
        fs::read_to_string(support::root().join(".github/workflows/ci-platform-checks-slice.yml"))
            .unwrap();
    assert!(dynamic_request(&workflow_yaml::parse(&source).unwrap()).is_ok());
    for (before, after) in [
        (
            "WINDOWS_DYNAMIC_TEST_CRATES: ${{ steps.windows_dynamic_packages.outputs.crates }}",
            "WINDOWS_DYNAMIC_TEST_CRATES: ${{ steps.packages.outputs.crates }}",
        ),
        (
            "cargo test --locked -p $crate --lib --features mesh-llm-system/dynamic-native-runtime",
            "cargo test --locked -p $crate --lib",
        ),
        (
            "cargo test --locked -p $crate --lib --features mesh-llm-system/dynamic-native-runtime",
            "# cargo test --locked -p $crate --lib --features mesh-llm-system/dynamic-native-runtime",
        ),
        (
            "foreach ($crate in @($env:WINDOWS_DYNAMIC_TEST_CRATES | ConvertFrom-Json)) {",
            "foreach ($crate in @('mesh-llm-system')) {",
        ),
        (
            "id: windows_dynamic_packages",
            "id: detached_dynamic_packages",
        ),
    ] {
        assert!(source.contains(before));
        let changed = workflow_yaml::parse(&source.replace(before, after)).unwrap();
        assert!(
            dynamic_request(&changed).is_err(),
            "mutation must refuse: {before}"
        );
    }
}
#[test]
fn every_current_windows_catalog_unit_owner_selects_the_native_planner_row() {
    let unit_only = catalog_requests("platform-windows-cfg");
    let catalog_owners = catalog_requests("platform-windows");
    let owners = unit_owner_names("windows", false);
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
    let plan = plan_for_workflow_owner("mesh-llm-wallet", "pr-ready");
    assert_eq!(
        plan["direct_crates"],
        serde_json::json!(["mesh-llm-wallet"])
    );
    assert!(
        plan["affected_crates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|name| name == "mesh-llm-host-runtime")
    );
    assert!(
        plan["domains"]
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
            .any(|row| row["id"] == "windows-unit")
    );
    assert!(
        plan["matrices"]["rust_tests"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row["crates"].as_array().unwrap())
            .any(|name| name == "mesh-llm-wallet")
    );
    assert!(unit_owners("windows").contains("mesh-llm-wallet"));
    for profile in ["main", "manual-full"] {
        assert!(
            plan_for_workflow_owner("mesh-llm-wallet", profile)["matrices"]["platform_checks"]
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

#[test]
fn resolved_legacy_host_keeps_current_membership_visible_in_both_catalog_and_windows_row() {
    let raw = vec!["mesh-llm-host-runtime".to_owned()];
    let resolved: BTreeSet<_> = resolve_owners(&raw, "legacy").into_iter().collect();
    assert!(resolved.contains("mesh-llm-membership"));
    assert!(unit_owners("windows").contains("mesh-llm-membership"));
    assert!(catalog("platform-windows").contains("mesh-llm-membership"));
    let divergent = BTreeSet::from(["mesh-llm-membership".to_owned()]);
    assert!(admit_catalog_visibility(&divergent, &resolved, &resolved).is_ok());
    // Remove the actual legacy catalog owner; the observed row remains unchanged.
    let removed: BTreeSet<_> = resolve_owners(&["mesh-llm-system".to_owned()], "legacy")
        .into_iter()
        .collect();
    assert!(admit_catalog_visibility(&divergent, &resolved, &removed).is_err());
}

#[test]
fn current_runtime_installer_platform_site_is_unix_test_evidence_only() {
    let source = fs::read_to_string(
        support::root().join("skippy/crates/skippy-runtime-install/src/import.rs"),
    )
    .unwrap();
    let sites = classify_sites(&source);
    assert!(sites.test && !sites.runtime);
    let mutated = format!("{source}\n#[cfg(windows)] fn product_import_branch() {{}}");
    let mutated_sites = classify_sites(&mutated);
    assert!(mutated_sites.runtime);
    let name = "skippy-runtime-install".to_owned();
    let empty = BTreeSet::new();
    assert!(
        admit_runtime_coverage(&BTreeMap::from([(name.clone(), sites)]), &empty, &empty).is_ok()
    );
    assert!(
        admit_runtime_coverage(&BTreeMap::from([(name, mutated_sites)]), &empty, &empty).is_err()
    );
    // The owner has no claimed Windows row/catalog/exception qualification.
    assert!(!unit_owners("windows").contains("skippy-runtime-install"));
    assert!(!catalog("platform-windows").contains("skippy-runtime-install"));
    assert!(!UNVERIFIED.contains(&"skippy-runtime-install"));
}

#[test]
fn native_trajectory_reader_platform_admission_is_explicitly_unverified() {
    let root = support::root();
    let owners = packages(&root);
    assert_eq!(
        owners.get("trajectory-reader").map(String::as_str),
        Some("tools/trajectory-reader")
    );
    let sites = source_sites(&root.join("tools/trajectory-reader/src"));
    assert!(
        sites.runtime,
        "reader regular-file admission must stay in runtime census"
    );
    assert!(!unit_owners("windows").contains("trajectory-reader"));
    assert!(!catalog("platform-windows").contains("trajectory-reader"));
    assert!(UNVERIFIED.contains(&"trajectory-reader"));
    let reader = BTreeMap::from([("trajectory-reader".to_owned(), sites)]);
    let empty = BTreeSet::new();
    assert!(admit_runtime_coverage(&reader, &empty, &empty).is_err());
    let explicit = BTreeSet::from(["trajectory-reader".to_owned()]);
    assert!(admit_runtime_coverage(&reader, &empty, &explicit).is_ok());
    // This accounting is not a Windows routing or runtime qualification claim.
    let unrelated = BTreeSet::from(["xtask".to_owned()]);
    assert!(admit_runtime_coverage(&reader, &empty, &unrelated).is_err());
}
