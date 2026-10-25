//! Exercise the heartbeat program shipped in the protected repair wrapper.
use crate::{process::*, support::*};
use std::{collections::BTreeMap, fs, time::Duration};

fn heartbeat(root: &std::path::Path, normal_stop: bool) -> ProcessSpec {
    let wrapper = fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let program = wrapper
        .split_once("env -i PATH=\"$PATH\" bash -c '")
        .unwrap()
        .1
        .split_once("' heartbeat \"$ROOT\" \"$started\" &")
        .unwrap()
        .0
        .replace(
            "sleeper=$!",
            "sleeper=$!; printf '%s' \"$sleeper\" > \"$root/sleeper.pid\"; printf 'READY\\n'",
        );
    fs::write(root.join("heartbeat.sh"), program).unwrap();
    let script = if normal_stop {
        "bash heartbeat.sh \"$PWD\" 0 & worker=$!; while [[ ! -f sleeper.pid ]]; do sleep 0.01; done; kill \"$worker\"; wait \"$worker\""
    } else {
        "exec bash heartbeat.sh \"$PWD\" 0"
    };
    ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
        cwd: root.into(),
        environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
    }
}

#[test]
fn protected_canary_heartbeat_reaps_sleep_on_normal_stop_and_owned_group_stop() {
    for normal_stop in [true, false] {
        let root = tempfile::tempdir().unwrap();
        let mut sentinel = Sentinel::new(root.path());
        let mut bounds = limits();
        bounds.graceful_shutdown = Duration::from_secs(2);
        if !normal_stop {
            ready(&mut bounds, b"READY");
            bounds.completion = Completion::StopAfterReady;
        }
        let report = supervise(
            &heartbeat(root.path(), normal_stop),
            &bounds,
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(report.success() && report.cleanup.complete, "{report:?}");
        assert_stopped(root.path(), &["sleeper"]);
        sentinel.assert_alive();
    }
}
