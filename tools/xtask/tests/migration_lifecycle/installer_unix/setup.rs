//! The actual installer hands post-install setup to the installed host.
use super::fixture::{Fixture, PREFERRED, stderr, stdout};
use flate2::{Compression, write::GzEncoder};
use std::fs;

fn release(fixture: &Fixture, asset: &str) {
    let host =
        b"#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$*\" >> \"$FIXTURE_ROOT/mesh.calls\"\n";
    let mut archive = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
    for (name, mode, body) in [
        ("mesh-bundle/mesh-llm", 0o755, host.as_slice()),
        (
            "mesh-bundle/product-manifest.json",
            0o644,
            b"{}\n".as_slice(),
        ),
        (
            "mesh-bundle/native-runtimes/fixture/manifest.json",
            0o644,
            b"{}\n".as_slice(),
        ),
        (
            "mesh-bundle/native-runtimes/fixture/lib/libllama.so",
            0o644,
            b"fixture runtime".as_slice(),
        ),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(mode);
        header.set_mtime(0);
        header.set_cksum();
        archive.append_data(&mut header, name, body).unwrap();
    }
    fixture.asset(asset, &archive.into_inner().unwrap().finish().unwrap());
    for tool in ["systemctl", "launchctl"] {
        fixture.stub(
            tool,
            "#!/bin/bash\nprintf called >> \"$FIXTURE_ROOT/service.calls\"\nexit 0\n",
        );
    }
}

fn main_body(interactive: bool, verbose: bool, arguments: &str) -> String {
    format!(
        "export MESH_LLM_TEST_UNAME_S=Darwin MESH_LLM_TEST_UNAME_M=arm64 MESH_LLM_TEST_INTERACTIVE={}\nINSTALL_VERBOSE={}\nmain --install-dir \"$INSTALL_DIR\" {arguments}",
        u8::from(interactive),
        u8::from(verbose)
    )
}

fn no_shell_service(fixture: &Fixture) {
    assert!(!fixture.root.join("service.calls").exists());
}

#[test]
fn installer_setup_runs_only_interactively_unless_explicitly_disabled() {
    for (interactive, arguments, runs) in [
        (true, "", true),
        (false, "", false),
        (true, "--no-setup", false),
    ] {
        let fixture = Fixture::new();
        release(&fixture, PREFERRED);
        let report = fixture.run(&main_body(interactive, false, arguments));
        assert!(report.success(), "{report:?}");
        let calls = fixture.root.join("mesh.calls");
        if runs {
            assert_eq!(fs::read_to_string(calls).unwrap(), "setup\n");
            assert!(stdout(&report).contains("↓ Fetching mesh-llm release..."));
        } else {
            assert!(!calls.exists());
            assert!(stdout(&report).contains("Run this next:"));
            assert!(stdout(&report).contains("/mesh-llm setup"));
        }
        assert!(stdout(&report).contains("Installed mesh-llm to"));
        for detail in [
            "Release channel:",
            "Verified checksum:",
            "Running post-install setup:",
        ] {
            assert!(!stdout(&report).contains(detail));
        }
        no_shell_service(&fixture);
    }
}

#[test]
fn installer_verbose_flag_and_environment_preserve_download_details_and_setup_arguments() {
    for (verbose, arguments) in [(false, "--verbose"), (true, "")] {
        let fixture = Fixture::new();
        release(&fixture, PREFERRED);
        let report = fixture.run(&main_body(true, verbose, arguments));
        assert!(report.success(), "{report:?}");
        assert_eq!(
            fs::read_to_string(fixture.root.join("mesh.calls")).unwrap(),
            "setup --verbose\n"
        );
        for detail in [
            "Release channel: stable",
            "Verified checksum:",
            "Running post-install setup:",
        ] {
            assert!(stdout(&report).contains(detail));
        }
        assert!(stdout(&report).contains(&format!("Installed {PREFERRED}")));
        assert!(!stdout(&report).contains("↓ Fetching mesh-llm release..."));
        no_shell_service(&fixture);
    }
}

#[test]
fn installer_jetson_download_and_legacy_service_flags_keep_setup_ownership_in_the_host() {
    let fixture = Fixture::new();
    release(
        &fixture,
        "mesh-llm-aarch64-unknown-linux-gnu-cuda-13.tar.gz",
    );
    let report = fixture.run(
        "export MESH_LLM_TEST_UNAME_S=Linux MESH_LLM_TEST_UNAME_M=aarch64 MESH_LLM_TEST_INTERACTIVE=0 MESH_LLM_TEST_TEGRA_MODEL='NVIDIA Jetson AGX Orin' MESH_LLM_TEST_CUDA_MAJOR=13\nINSTALL_VERBOSE=0\nmain --install-dir \"$INSTALL_DIR\" --no-setup",
    );
    assert!(report.success(), "{report:?}");
    assert!(stdout(&report).contains("Installed mesh-llm to"));
    assert!(!stdout(&report).contains("mesh-llm-aarch64-unknown-linux-gnu-cuda-13.tar.gz"));
    assert!(!fixture.root.join("mesh.calls").exists());
    no_shell_service(&fixture);

    let fixture = Fixture::new();
    release(&fixture, PREFERRED);
    let report = fixture.run(&main_body(true, false, "--service --no-start-service"));
    assert!(report.success(), "{report:?}");
    assert_eq!(
        fs::read_to_string(fixture.root.join("mesh.calls")).unwrap(),
        "setup --service\n"
    );
    assert!(stderr(&report).contains("forwarding it to `mesh-llm setup --service`"));
    no_shell_service(&fixture);
}
