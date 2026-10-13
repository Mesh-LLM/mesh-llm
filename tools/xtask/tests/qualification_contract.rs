//! Contract tests for the standalone Skippy qualification commands, the
//! host-receipts provenance record, and the Cargo batch census. These port the
//! intent of the legacy Python suites
//! (`test_ci_skippy_qualification_contract.py`,
//! `test_ci_host_qualification_receipts.py`,
//! `test_ci_skippy_qualification_availability.py`, `test_ci_cargo_census.py`)
//! as fixture-driven CLI tests against the compiled xtask binary.
#![cfg(feature = "automation-contracts")]

#[path = "qualification_contract/availability.rs"]
mod availability;
#[path = "qualification_contract/census.rs"]
mod census;
#[path = "qualification_contract/compose_record.rs"]
mod compose_record;
#[path = "qualification_contract/host_receipts.rs"]
mod host_receipts;
#[path = "qualification_contract/support.rs"]
mod support;
#[path = "qualification_contract/validate.rs"]
mod validate;
