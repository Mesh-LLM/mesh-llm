//! Actual copied target wrapper and no-target Darwin probe behavior.
use super::{Fixture, executable, program};

#[test]
fn accelerator_driver_linux_wrapper_falls_back_to_lld_and_removes_injected_linker() {
    let fixture = Fixture::new();
    let result = fixture.run(
        program("bash"),
        &[
            "scripts/cargo-linker-linux-x86_64",
            "-m64",
            "-fuse-ld=rustc-injected",
            "input with spaces.o",
        ],
        &[],
    );
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert!(result.stderr.contains("mold is unavailable"));
    assert_eq!(fixture.text("compiler.events"), "probe\nlink\n");
    let arguments = fixture.arguments();
    assert!(
        !arguments
            .iter()
            .any(|value| value.contains("rustc-injected"))
    );
    let final_start = arguments
        .iter()
        .rposition(|value| value.ends_with("/cc"))
        .unwrap();
    let final_arguments = &arguments[final_start + 1..];
    assert_eq!(
        final_arguments,
        [
            "-m64".to_owned(),
            "input with spaces.o".to_owned(),
            "-fuse-ld=lld".to_owned(),
        ]
    );
}

#[test]
fn accelerator_driver_macos_bare_probe_requires_no_explicit_target_arguments() {
    let fixture = Fixture::new();
    executable(
        &fixture.root.join("bin/xcrun"),
        "#!/bin/bash\nprintf '%s\\n' 'finite SDK with spaces'\n",
    );
    let result = fixture.run(
        "/bin/bash".into(),
        &["scripts/cargo-linker", "--mesh-probe"],
        &[("FAKE_OS", "Darwin"), ("FAKE_ARCH", "arm64")],
    );
    assert_eq!(result.code, 0, "{}", result.stderr);
    assert_eq!(
        result.stdout.trim(),
        fixture.root.join("bin/ld64.lld").to_str().unwrap()
    );
    assert_eq!(fixture.text("compiler.events"), "probe\n");
    let arguments = fixture.arguments();
    for flag in ["-target", "-arch", "-isysroot", "--mesh-probe"] {
        assert!(!arguments.iter().any(|value| value == flag), "{flag}");
    }
    assert!(arguments.iter().any(|value| {
        value == &format!("-fuse-ld={}", fixture.root.join("bin/ld64.lld").display())
    }));
}
