use super::*;
use serial_test::serial;
use std::ffi::OsString;
use std::fs;

struct IdentityEnvGuard {
    home: Option<OsString>,
    test_home: Option<OsString>,
    node_key_path: Option<OsString>,
}

impl IdentityEnvGuard {
    fn set_home(home: &std::path::Path) -> Self {
        let previous = Self {
            home: std::env::var_os("HOME"),
            test_home: std::env::var_os("MESH_LLM_TEST_HOME"),
            node_key_path: std::env::var_os("MESH_LLM_NODE_KEY_PATH"),
        };
        unsafe {
            // SAFETY: this guard is used only by the #[serial] test and restores both values.
            std::env::set_var("HOME", home);
            // SAFETY: this guard is used only by the #[serial] test and restores all values.
            std::env::set_var("MESH_LLM_TEST_HOME", home);
            // SAFETY: this guard is used only by the #[serial] test and restores both values.
            std::env::remove_var("MESH_LLM_NODE_KEY_PATH");
        }
        previous
    }
}

impl Drop for IdentityEnvGuard {
    fn drop(&mut self) {
        match self.home.take() {
            Some(value) => {
                // SAFETY: this guard is used only by the #[serial] test and restores both values.
                unsafe { std::env::set_var("HOME", value) }
            }
            None => {
                // SAFETY: this guard is used only by the #[serial] test and restores both values.
                unsafe { std::env::remove_var("HOME") }
            }
        }
        match self.test_home.take() {
            Some(value) => {
                // SAFETY: this guard is used only by the #[serial] test and restores all values.
                unsafe { std::env::set_var("MESH_LLM_TEST_HOME", value) }
            }
            None => {
                // SAFETY: this guard is used only by the #[serial] test and restores all values.
                unsafe { std::env::remove_var("MESH_LLM_TEST_HOME") }
            }
        }
        match self.node_key_path.take() {
            Some(value) => {
                // SAFETY: this guard is used only by the #[serial] test and restores both values.
                unsafe { std::env::set_var("MESH_LLM_NODE_KEY_PATH", value) }
            }
            None => {
                // SAFETY: this guard is used only by the #[serial] test and restores both values.
                unsafe { std::env::remove_var("MESH_LLM_NODE_KEY_PATH") }
            }
        }
    }
}

/// Test that mark_was_public / was_previously_public / clear_public_identity
/// work correctly in an isolated temporary home and key namespace.
#[test]
#[serial]
pub(crate) fn public_to_private_transition_clears_identity() {
    let temp = tempfile::tempdir().expect("temp home");
    let _env = IdentityEnvGuard::set_home(temp.path());
    let dir = identity_home_dir().join(".mesh-llm");
    fs::create_dir_all(&dir).ok();

    // --- Scenario 1: no marker → was_previously_public is false ---
    let _ = fs::remove_file(dir.join("was-public"));
    assert!(
        !was_previously_public().expect("resolve marker"),
        "should be false when no marker"
    );

    // --- Scenario 2: mark as public → marker exists ---
    mark_was_public().expect("mark public identity");
    assert!(
        was_previously_public().expect("resolve marker"),
        "should be true after marking"
    );

    // Plant some identity files to verify clear removes them.
    fs::write(dir.join("key"), b"test-key").unwrap();
    fs::write(dir.join("nostr.nsec"), b"test-nsec").unwrap();
    fs::write(dir.join("mesh-id"), b"test-mesh-id").unwrap();
    fs::write(dir.join("last-mesh"), b"test-last-mesh").unwrap();

    // --- Scenario 3: clear_public_identity removes everything ---
    clear_public_identity().expect("clear public identity");
    for name in &["key", "nostr.nsec", "mesh-id", "last-mesh", "was-public"] {
        assert!(
            !dir.join(name).exists(),
            "{name} should be deleted after clear"
        );
    }
    assert!(
        !was_previously_public().expect("resolve marker"),
        "marker should be gone after clear"
    );

    // --- Scenario 4: clear on already-clean directory is fine ---
    clear_public_identity().expect("clear already-clean public identity");
}
