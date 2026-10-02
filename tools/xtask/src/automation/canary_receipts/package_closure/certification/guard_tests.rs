use super::*;
use crate::process::{Outcome, Value};
use std::collections::BTreeMap;

#[cfg(unix)]
#[test]
fn low_memory_observation_stops_actual_owned_fixture_tree_and_records_minimum() {
    let directory = tempfile::tempdir().unwrap();
    let ready = directory.path().join("ready");
    let spec = ProcessSpec {
        executable: "/bin/sh".into(),
        arguments: [
            "-c".into(),
            "printf ready > \"$1\"; sleep 30".into(),
            "fixture".into(),
            ready.clone().into(),
        ]
        .into_iter()
        .map(Value::Public)
        .collect(),
        cwd: directory.path().to_owned(),
        environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
    };
    let observed = monitored(
        &spec,
        Instant::now() + Duration::from_secs(10),
        &Cancellation::default(),
        Host {
            total: 1000,
            available: 900,
        },
        100,
        OutputFiles::default(),
        |_| {
            Ok(Host {
                total: 1000,
                available: if ready.exists() { 99 } else { 900 },
            })
        },
    )
    .unwrap();
    assert!(ready.exists());
    assert_eq!(observed.minimum, 99);
    assert!(observed.guard_error.unwrap().contains("10% reserve"));
    assert_eq!(observed.process.outcome, Outcome::Cancelled);
    assert!(observed.process.cleanup.complete);
    assert!(!observed.process.success());
}

#[cfg(unix)]
#[test]
fn successful_fixture_keeps_host_observation_and_owned_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let spec = ProcessSpec {
        executable: "/bin/sh".into(),
        arguments: ["-c", "printf fixture"]
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        cwd: directory.path().to_owned(),
        environment: BTreeMap::new(),
    };
    let observed = monitored(
        &spec,
        Instant::now() + Duration::from_secs(5),
        &Cancellation::default(),
        Host {
            total: 1000,
            available: 900,
        },
        100,
        OutputFiles::default(),
        |_| {
            Ok(Host {
                total: 1000,
                available: 800,
            })
        },
    )
    .unwrap();
    assert!(observed.process.success());
    assert!(observed.guard_error.is_none());
    assert!(observed.minimum <= 900);
    assert!(observed.process.cleanup.complete);
}
