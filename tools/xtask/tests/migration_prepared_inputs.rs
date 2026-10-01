//! Prepared-input consumers ported from `scripts/`: release UI stamping and
//! verification (`ui-distribution.py`), the static ABI build stamp
//! (`verify-static-abi-build-stamp.py`) and input manifest (the inline
//! programs in `prepare-static-abi-input` / `restore-static-abi-input.sh`),
//! the native SDK manifest consumers, and the SDK runtime-list JSON reader
//! that `test_ci_sdk_json_consumer.py` exercises.
//!
//! Every fixture is built at runtime in a scratch directory. Expected bytes
//! include authored contract-derived literals, not necessarily captured output.

#[path = "migration_prepared_inputs/support.rs"]
mod support;

#[path = "migration_prepared_inputs/native_sdk.rs"]
mod native_sdk;
#[path = "migration_prepared_inputs/sdk_runtime_report.rs"]
mod sdk_runtime_report;
#[path = "migration_prepared_inputs/static_abi_input.rs"]
mod static_abi_input;
#[path = "migration_prepared_inputs/static_abi_producer.rs"]
mod static_abi_producer;
#[path = "migration_prepared_inputs/static_abi_stamp.rs"]
mod static_abi_stamp;
#[path = "migration_prepared_inputs/ui_distribution.rs"]
mod ui_distribution;

#[path = "migration_prepared_inputs/sdk_console.rs"]
mod sdk_console;
