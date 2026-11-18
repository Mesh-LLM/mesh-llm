use super::*;
use crate::ci_validation::lane_results::workflow_yaml;

const PACKAGE_NATIVE_SDK: &str = "mesh/scripts/package-native-sdk.sh";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn workflows(root: &Path) -> BTreeMap<String, Node> {
    let mut workflows = BTreeMap::new();
    for entry in std::fs::read_dir(root.join(".github/workflows")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.ends_with(".yml") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let document =
            if name == "llama-upstream-canary.yml" || name == "llama-canary-family-pass.yml" {
                workflow_yaml::parse_resolved_aliases(&source)
            } else {
                workflow_yaml::parse(&source)
            };
        workflows.insert(name, document.unwrap());
    }
    workflows
}

#[test]
fn restored_consumers_reach_no_direct_cargo_xtool() {
    let root = root();
    let workflows = workflows(&root);
    let reached = reachable_scripts(&root, &workflows).unwrap();
    assert!(reached.contains(Path::new(PACKAGE_NATIVE_SDK)));
    check(&root, &workflows).unwrap();
}

#[test]
fn native_sdk_packaging_without_restored_selection_is_rejected() {
    let source = std::fs::read_to_string(root().join(PACKAGE_NATIVE_SDK)).unwrap();
    assert!(direct_calls(&source).is_empty());
    let selection = "    automation=(\"$MESH_LLM_AUTOMATION_BIN\")\n";
    assert!(source.contains(selection));
    let reverted = source
        .replace(selection, "")
        .replace("\"${automation[@]}\" ", "cargo xtool ");
    assert_eq!(direct_calls(&reverted).len(), 3);
}

#[test]
fn restored_fallback_forms_are_admitted_and_direct_calls_rejected() {
    for admitted in [
        "if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then\n  \"$MESH_LLM_AUTOMATION_BIN\" a\nelse\n  cargo xtool a\nfi\n",
        "if command -v shasum; then\n  shasum\nelse\n  if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then\n    \"$MESH_LLM_AUTOMATION_BIN\" a\n  else\n    if true; then\n      cargo xtool a\n    fi\n  fi\nfi\n",
        "x() { if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then \"$MESH_LLM_AUTOMATION_BIN\" \"$@\"; else cargo xtool \"$@\"; fi; }\n",
        "automation=(cargo xtool)\nif [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then automation=(\"$MESH_LLM_AUTOMATION_BIN\"); fi\n",
        "# cargo xtool a\n",
    ] {
        assert!(direct_calls(admitted).is_empty(), "{admitted}");
    }
    for (rejected, lines) in [
        ("cargo xtool a\n", vec![1]),
        ("automation=(cargo xtool)\n", vec![1]),
        (
            "if [[ -n \"${MESH_LLM_AUTOMATION_BIN:-}\" ]]; then\n  cargo xtool a\nfi\n",
            vec![2],
        ),
        ("if true; then\n  :\nelse\n  cargo xtool a\nfi\n", vec![4]),
    ] {
        assert_eq!(direct_calls(rejected), lines, "{rejected}");
    }
}
