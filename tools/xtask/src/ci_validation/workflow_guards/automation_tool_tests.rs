use super::*;

fn root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn action(name: &str) -> String {
    std::fs::read_to_string(root().join(format!(".github/actions/{name}/action.yml"))).unwrap()
}

fn parse(source: &str) -> Node {
    super::super::workflow_yaml::parse(source).unwrap()
}

#[test]
fn automation_tool_actual_actions_bind_the_narrow_protected_producer() {
    check(&root()).unwrap();
}

#[test]
fn automation_tool_validation_rejects_an_unknown_tool_value() {
    let document = parse(&action("prepare-automation"));
    let run = field(
        step(&document, "name", "Validate preparation profile").unwrap(),
        "run",
    )
    .unwrap();
    for (tool, accepted) in [
        ("xtask", true),
        ("authority-automation", true),
        ("ci-controller", false),
        ("", false),
        ("xtask --features automation-contracts", false),
    ] {
        let output = std::process::Command::new("bash")
            .args(["-c", run])
            .env("AUTOMATION_TOOL", tool)
            .env("AUTOMATION_PROFILE", "image")
            .output()
            .unwrap();
        assert_eq!(output.status.success(), accepted, "{tool:?}");
    }
}

#[test]
fn automation_tool_rejects_an_open_or_unvalidated_tool_input() {
    let source = action("prepare-automation");
    for changed in [
        source.replace("    default: xtask\n", "    default: authority-automation\n"),
        source.replace(
            "          xtask|authority-automation) ;;\n          *) echo \"unsupported automation tool\" >&2; exit 1 ;;\n",
            "          *) ;;\n",
        ),
        source.replace(
            "&& inputs.reuse-hosted-binary == 'true' && inputs.tool == 'xtask'\n      uses: actions/cache/restore",
            "&& inputs.reuse-hosted-binary == 'true'\n      uses: actions/cache/restore",
        ),
    ] {
        assert_ne!(source, changed);
        assert!(prepare(&parse(&changed)).is_err());
    }
}

#[test]
fn automation_tool_rejects_a_protected_clean_producer_that_builds_full_xtask() {
    let source = action("upload-automation");
    for changed in [
        source.replacen(TOOL_BINDING, "xtask", 1),
        source.replacen(
            TOOL_BINDING,
            "${{ inputs.source-profile == 'protected-clean' && 'xtask' || 'xtask' }}",
            1,
        ),
        source.replace(
            &format!("        AUTOMATION_TOOL: {TOOL_BINDING}\n"),
            "        AUTOMATION_TOOL: xtask\n",
        ),
        source.replace(
            "            authority-automation) recipe=authority-automation-bootstrap ;;\n            *) exit 1 ;;\n",
            "            authority-automation) recipe=automation-bootstrap ;;\n            *) exit 1 ;;\n",
        ),
    ] {
        assert_ne!(source, changed);
        assert!(upload(&parse(&changed)).is_err());
    }
}
