//! Execute the portable gate with inert tools to verify ordering and failures.
use super::*;

/// The portable gate serializes Cargo owners and retains every stage failure.
#[test]
fn portable_gate_serializes_cargo_owners_and_retains_failed_checks() {
    for failed_stage in ["none", "sdk", "ui", "website"] {
        let state = tempfile::tempdir().unwrap();
        let root = state.path();
        for directory in ["mesh/crates/mesh-llm-ui", "mesh/website"] {
            fs::create_dir_all(root.join(directory)).unwrap();
        }
        executable(
            &root.join("scripts/test-portable.sh"),
            &fs::read_to_string(repo().join("scripts/test-portable.sh")).unwrap(),
        );
        fs::write(root.join("failed-stage"), failed_stage).unwrap();
        executable(
            &root.join("bin/pnpm"),
            "#!/bin/sh\n[ \"$(cat \"$HOME/failed-stage\")\" != ui ]\n",
        );
        for command in ["just", "node"] {
            executable(&root.join(format!("bin/{command}")), "#!/bin/sh\nexit 0\n");
        }
        // The last SDK command marks completion. A concurrently started website
        // build must fail rather than silently contend for Cargo's locks.
        executable(
            &root.join("scripts/check-sdk-contract.sh"),
            "#!/bin/sh\ntouch \"$HOME/sdk-finished\"\n[ \"$(cat \"$HOME/failed-stage\")\" != sdk ]\n",
        );
        executable(&root.join("bin/java"), "#!/bin/sh\nexit 1\n");
        executable(
            &root.join("bin/npm"),
            "#!/bin/sh\nif [ \"$*\" = 'run build' ]; then\n touch \"$HOME/website-ran\"\n test -f \"$HOME/sdk-finished\" || exit 91\n [ \"$(cat \"$HOME/failed-stage\")\" != website ]\nfi\n",
        );
        let (success, output) = execute(root, &installed("bash"), &["scripts/test-portable.sh"]);
        assert_eq!(
            success,
            failed_stage == "none",
            "{failed_stage}: {output:?}"
        );
        assert!(root.join("website-ran").exists());
        let output = String::from_utf8(output).unwrap();
        for check in [
            "ui-lint",
            "ui-typecheck-build",
            "ui-unit",
            "script-sdk",
            "website-build",
        ] {
            assert!(
                output.contains(&format!("PASS {check}"))
                    || output.contains(&format!("FAIL {check}")),
                "{output}"
            );
        }
    }
}
