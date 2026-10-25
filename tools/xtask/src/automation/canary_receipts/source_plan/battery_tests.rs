use super::{super::contract::Resolved, preflight_with_cancellation};
use crate::process::Cancellation;
use std::{fs, path::PathBuf};

#[test]
fn selected_historical_battery_receives_only_current_controller_and_keeps_its_checks() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("historical source");
    fs::create_dir_all(source.join("scripts")).unwrap();
    fs::create_dir_all(source.join("target/debug")).unwrap();
    let source = source.canonicalize().unwrap();
    // A candidate executable exists but is never selected as controller.
    let candidate = source.join("target/debug/xtask");
    fs::write(&candidate, "untrusted selected-source candidate").unwrap();
    let battery = source.join("scripts/skippy-family-battery.sh");
    let plan = directory.path().join("immutable plan.json");
    fs::write(&plan, "plan bytes").unwrap();
    let resolved = Resolved {
        controller: directory.path().join("controller source"),
        source: source.clone(),
        manifest: source.join("selected manifest.json"),
        output: directory.path().join("output"),
    };
    for (status, success) in [(0, true), (19, false)] {
        fs::write(&battery, format!(
            "#!/bin/sh\nprintf '%s' \"$MESH_LLM_AUTOMATION_BIN\" > controller-path\nprintf '%s\\n' \"$@\" > battery-args\n# Historical component checks remain authoritative.\nexit {status}\n"
        )).unwrap();
        let result = preflight_with_cancellation(&resolved, &plan, &Cancellation::default());
        assert_eq!(result.is_ok(), success, "{result:?}");
        assert_eq!(
            PathBuf::from(fs::read_to_string(source.join("controller-path")).unwrap()),
            std::env::current_exe().unwrap()
        );
        assert_ne!(std::env::current_exe().unwrap(), candidate);
        assert_eq!(
            fs::read_to_string(source.join("battery-args")).unwrap(),
            format!("--skip-build\n--dry-run\n--plan\n{}\n", plan.display())
        );
        assert_eq!(fs::read(&plan).unwrap(), b"plan bytes");
        assert_eq!(
            fs::read(&candidate).unwrap(),
            b"untrusted selected-source candidate"
        );
    }
}
