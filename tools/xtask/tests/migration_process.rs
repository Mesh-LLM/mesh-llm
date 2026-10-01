#[path = "migration_process/driver_expectations.rs"]
mod driver_expectations;
#[path = "migration_process/driver_tests.rs"]
mod driver_tests;
#[path = "migration_process/fixture.rs"]
mod fixture;
#[path = "migration_process/lifecycle.rs"]
mod lifecycle;
#[cfg(unix)]
#[path = "migration_process/observed.rs"]
mod observed;
#[path = "migration_process/observed_fixture.rs"]
mod observed_fixture;
#[path = "migration_process/output.rs"]
mod output;
#[path = "migration_process/pipe_capture.rs"]
mod pipe_capture;
#[path = "migration_process/pipe_fixture.rs"]
mod pipe_fixture;
#[cfg(unix)]
#[path = "migration_process/probe.rs"]
mod probe;
#[path = "../src/process/mod.rs"]
pub mod process;
#[cfg(unix)]
#[path = "migration_process/readiness.rs"]
mod readiness;
#[path = "migration_process/readiness_fixture.rs"]
mod readiness_fixture;
#[path = "migration_process/retained.rs"]
mod retained;
#[path = "migration_process/retained_fixture.rs"]
mod retained_fixture;
#[path = "migration_process/support.rs"]
mod support;
#[cfg(windows)]
#[path = "migration_process/windows.rs"]
mod windows;

#[test]
fn migration_process_fixture() {
    if std::env::var_os("MIGRATION_PROCESS_MODE").is_some() {
        fixture::run().expect("fixture execution");
    }
}
