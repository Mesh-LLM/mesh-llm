//! `native {select-runtime,verify-host-dependencies,linux-runtime-deps,
//! windows-runtime-deps}` parity with `scripts/select-native-runtime.py`,
//! `scripts/verify-host-dependencies.py`,
//! `scripts/linux-native-runtime-deps.py` and
//! `scripts/windows-native-runtime-deps.py`.
//!
//! Cases live in `tests/fixtures/native_policy/*.json`; their expected
//! streams, statuses and report files are the legacy scripts' observed
//! output. Inspection tools are shell stubs placed alone on `PATH`, so no
//! real `readelf`/`otool`/`objdump` runs.
#![cfg(unix)]

#[path = "migration_native_policy/support.rs"]
mod support;

#[path = "migration_native_policy/select_runtime.rs"]
mod select_runtime;

#[path = "migration_native_policy/host_dependencies.rs"]
mod host_dependencies;

#[path = "migration_native_policy/elf_stub.rs"]
mod elf_stub;

#[path = "migration_native_policy/linux_deps.rs"]
mod linux_deps;

#[path = "migration_native_policy/pe_stub.rs"]
mod pe_stub;

#[path = "migration_native_policy/windows_deps.rs"]
mod windows_deps;

#[path = "migration_native_policy/release_matrix.rs"]
mod release_matrix;

#[path = "migration_native_policy/runtime_package.rs"]
mod runtime_package;
