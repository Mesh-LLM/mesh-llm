//! `product compose` parity with `scripts/compose-product-bundle.py`.
//!
//! Each case in `tests/fixtures/product/compose_cases.json` names the input
//! tree to build in a scratch directory and the command line to run there.
//! `compose_goldens.json` holds the legacy script's observed status, streams
//! and resulting `product-manifest.json` for every case.
#![cfg(unix)]

#[path = "migration_product/support.rs"]
mod support;

#[path = "migration_product/compose.rs"]
mod compose;

#[path = "migration_product/packaging_cases.rs"]
mod packaging_cases;

#[path = "migration_product/packaging.rs"]
mod packaging;

#[path = "migration_product/release_manifest.rs"]
mod release_manifest;

#[path = "migration_product/compose_input.rs"]
mod compose_input;

#[path = "migration_product/archive.rs"]
mod archive;
