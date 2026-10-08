use super::{
    SYSTEM_ONE_OWNER, check_selected_processes, indirect_launch, typed_owner, typed_source_argv,
};
use crate::command::DynResult;
use std::fs;

const CALLER: &str = "skippy/scripts/skippy-system-one-smoke.sh";
const SOURCE: &str = include_str!("../../../../skippy/scripts/skippy-system-one-smoke.sh");

fn records(source: &str) -> DynResult<Vec<super::SelectedProcessCall>> {
    let lines = source.lines().collect::<Vec<_>>();
    lines.iter().enumerate().filter(|(_, line)| indirect_launch(CALLER, line)).map(|(index, line)| {
        let mixed = line.trim().starts_with("\"${case_command[@]}\"");
        let mut argv = typed_source_argv(&lines, index)?;
        if mixed {
            argv = format!("{}\n{argv}", lines[index - 4..index].iter().map(|line| line.trim()).collect::<Vec<_>>().join("\n"));
        }
        Ok(super::SelectedProcessCall {
            caller: CALLER.to_owned(), line: index + 1, source_block: line.trim().to_owned(),
            child: if mixed { "tools/xtask default; bounded explicit native $CASES_DRIVER override" } else { "tools/xtask configured absolute regular executable or trusted Just automation facade" }.to_owned(),
            child_source_known: false,
            replacement_owner: if mixed { SYSTEM_ONE_OWNER } else { typed_owner(&argv).ok_or("missing typed owner")? }.to_owned(),
            argv, status_streams_effects: "Source fixture only; no native or model qualification".to_owned(),
        })
    }).collect()
}

#[test]
fn system_one_serializers_require_each_actual_typed_launch_and_owner() -> DynResult<()> {
    let root = crate::command::unique_temp_dir("system-one-serializer-binding");
    fs::create_dir_all(root.join("skippy/scripts"))?;
    fs::write(root.join(CALLER), SOURCE)?;
    let bindings = records(SOURCE)?;
    assert_eq!(bindings.len(), 6); // port, two resolvers, stage, mixed driver, outcome.
    check_selected_processes(&root, &bindings)?;
    for index in 0..bindings.len() {
        let mut omitted = bindings.clone();
        omitted.remove(index);
        assert!(check_selected_processes(&root, &omitted).is_err());
        let mut rebound = bindings.clone();
        rebound[index].replacement_owner = "tools/xtask/src/unrelated.rs".to_owned();
        assert!(check_selected_processes(&root, &rebound).is_err());
        let mut argv = bindings.clone();
        argv[index].argv.push_str(" --extra");
        assert!(check_selected_processes(&root, &argv).is_err());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn system_one_serializers_reject_rebound_helpers_and_conditional_native_driver_drift()
-> DynResult<()> {
    let root = crate::command::unique_temp_dir("system-one-serializer-shape");
    fs::create_dir_all(root.join("skippy/scripts"))?;
    let bindings = records(SOURCE)?;
    for (old, new) in [
        ("automation local-ports 1", "automation local-ports 2"),
        (
            "automation system-one-smoke stage \"$@\"",
            "automation system-one-smoke stage \"$1\"",
        ),
        ("port=\"$(pick_port)\" || return 2", "port=12345"),
        ("\"$gpu_layers\" || return 2", "\"$gpu_layers\" || true"),
        (
            "\"$REPORT_PATH\" \"$status\"",
            "\"$OTHER_REPORT\" \"$status\"",
        ),
        ("! -x \"$CASES_DRIVER\"", "! -f \"$CASES_DRIVER\""),
    ] {
        assert!(SOURCE.contains(old), "missing mutation anchor: {old}");
        fs::write(root.join(CALLER), SOURCE.replace(old, new))?;
        assert!(
            check_selected_processes(&root, &bindings).is_err(),
            "admitted mutation: {old}"
        );
    }
    for tail in [
        "\npick_port() {\necho 12345\n}\n",
        "\nwrite_stage_config() {\necho unsafe\n}\n",
        "\nrequire_cmd python3 || exit 2\n",
        "\n\"${automation[@]}\" automation system-one-smoke stage \"$@\"\n",
    ] {
        fs::write(root.join(CALLER), format!("{SOURCE}{tail}"))?;
        assert!(check_selected_processes(&root, &bindings).is_err());
    }
    fs::remove_dir_all(root)?;
    Ok(())
}
