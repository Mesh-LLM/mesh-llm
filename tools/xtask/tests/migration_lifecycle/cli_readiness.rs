use crate::protocol::{Behavior, Destination};
use crate::support::{Case, Sentinel, record};

#[test]
fn migration_lifecycle_cli_nested_message() {
    let case = Case::new(
        Behavior::Clean,
        vec![record(
            Destination::Stdout,
            b"{\"event\":7,\"message\":{\"Client ready\":false}}\n",
        )],
    );

    let output = case.run();

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_cli_approved_structured_with_object_message() {
    let case = Case::new(
        Behavior::Clean,
        vec![record(
            Destination::Stderr,
            b"{\"event\":\"passive_mode\",\"status\":\"ready\",\"role\":\"client\",\"message\":{\"detail\":\"starting\"}}\n",
        )],
    );
    let mut sentinel = Sentinel::new(&case);

    let output = case.run();

    assert!(output.status.success());
    assert!(case.native.join("handler").is_file());
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_cli_approved_nonstring_messages_cannot_supply_readiness() {
    let case = Case::new(
        Behavior::Clean,
        vec![record(
            Destination::Stdout,
            concat!(
                "{\"message\":[\"Client ready\"]}\n",
                "{\"message\":{\"Client ready\":true}}\n",
                "{\"message\":{\"detail\":\"Client ready\"}}\n",
                "{\"message\":[\"\\fLIENT ready\"]}\n",
                "{\"message\":{\"\\u001cLIENT ready\":false}}\n",
                "{\"message\":[\"\\u0c5cLIENT ready\"]}\n",
                "{\"message\":42}\n{\"message\":true}\n{\"message\":null}\n",
            )
            .as_bytes(),
        )],
    );
    let mut sentinel = Sentinel::new(&case);

    let output = case.run();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("readiness deadline expired"));
    assert!(case.native.join("handler").is_file());
    assert!(sentinel.0.try_wait().unwrap().is_none());
    case.assert_removed();
}

#[test]
fn migration_lifecycle_cli_approved_stale_ready_file_is_not_observed_or_deleted() {
    let case = Case::new(Behavior::Clean, vec![]);
    let stale = case.state.join("mlc-state.stale");
    std::fs::create_dir(&stale).unwrap();
    let previous = stale.join("stdout.log");
    let bytes = b"{\"message\":\"Client ready\"}\n";
    std::fs::write(&previous, bytes).unwrap();
    let mut sentinel = Sentinel::new(&case);

    let output = case.run();

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(std::fs::read(previous).unwrap(), bytes);
    assert_eq!(std::fs::read_dir(&case.state).unwrap().count(), 1);
    crate::support::assert_absent(case.audit().pid);
    assert!(sentinel.0.try_wait().unwrap().is_none());
}
