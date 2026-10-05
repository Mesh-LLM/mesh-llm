//! Cross-crate test-home isolation regression.
//!
//! This integration test compiles `mesh-llm-membership` as a *dependency* (its
//! `#[cfg(test)]` is therefore `false`), which is exactly the configuration
//! the host's own unit tests run in: they call
//! `mesh_llm_membership::identity_home_dir()` / `default_node_key_path()`
//! through a normal dependency edge, not through the membership crate's own
//! test build.
//!
//! A bare `#[cfg(test)]` guard on the `MESH_LLM_TEST_HOME` override would be
//! compiled out in that configuration, silently pointing the host's
//! `public_identity_tests` and admission tests at the real user home. The
//! explicit `test-support` feature — enabled here through the host crate's
//! dev-dependency, and never in `[dependencies]` — restores the override for
//! that cross-crate case while leaving production builds (no `test-support`)
//! to resolve the home from `dirs::home_dir()` exactly as before.
//!
//! This test is the executable proof of that contract; the host's
//! `public_identity_tests` and `mesh/tests/admission/requirements.rs` then
//! rely on it rather than on the implicit `cfg(test)` coincidence.

use mesh_llm_membership::{default_node_key_path, identity_home_dir};
use std::ffi::OsString;
use std::path::PathBuf;

struct EnvGuard {
    test_home: Option<OsString>,
    node_key_path: Option<OsString>,
}

impl EnvGuard {
    fn install(home: &std::path::Path) -> Self {
        let test_home = std::env::var_os("MESH_LLM_TEST_HOME");
        let node_key_path = std::env::var_os("MESH_LLM_NODE_KEY_PATH");
        // SAFETY: this serial test's guard restores the value; no concurrent readers.
        unsafe { std::env::set_var("MESH_LLM_TEST_HOME", home.as_os_str()) };
        // SAFETY: this serial test's guard restores the value; no concurrent readers.
        unsafe { std::env::remove_var("MESH_LLM_NODE_KEY_PATH") };
        Self {
            test_home,
            node_key_path,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match self.test_home.take() {
            Some(value) => {
                // SAFETY: restore the prior value under the owning serial test's guard.
                unsafe { std::env::set_var("MESH_LLM_TEST_HOME", value) };
            }
            None => {
                // SAFETY: restore the prior value under the owning serial test's guard.
                unsafe { std::env::remove_var("MESH_LLM_TEST_HOME") };
            }
        }
        match self.node_key_path.take() {
            Some(value) => {
                // SAFETY: restore the prior value under the owning serial test's guard.
                unsafe { std::env::set_var("MESH_LLM_NODE_KEY_PATH", value) };
            }
            None => {
                // SAFETY: restore the prior value under the owning serial test's guard.
                unsafe { std::env::remove_var("MESH_LLM_NODE_KEY_PATH") };
            }
        }
    }
}

fn temp_home() -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "membership-test-home-{}-{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).expect("create isolated test home");
    dir
}

#[test]
#[serial_test::serial]
fn test_support_restores_test_home_override_across_the_dependency_edge() {
    let home = temp_home();
    let _guard = EnvGuard::install(&home);

    // Guard 1: identity_home_dir() honors MESH_LLM_TEST_HOME when the
    // membership crate is compiled as a dependency with `test-support` on.
    assert_eq!(identity_home_dir(), home);

    // Guard 2: default_node_key_path() resolves under the test home when
    // MESH_LLM_NODE_KEY_PATH is unset (the second `cfg(test)` branch).
    assert_eq!(
        default_node_key_path().expect("resolve default node key path"),
        home.join(".mesh-llm").join("key")
    );

    let _ = std::fs::remove_dir_all(&home);
}
