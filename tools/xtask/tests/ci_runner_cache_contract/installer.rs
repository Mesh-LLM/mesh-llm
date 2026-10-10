use super::support::{Fixture, action, step};
use crate::workflow_yaml::Node;
use std::{fs, process::Command};

fn install(platform: &str, verify_failure: bool) -> (Fixture, std::process::Output) {
    let f = Fixture::new();
    let document = action("install-actionlint");
    let Node::Seq(steps) = document.get("runs").unwrap().get("steps").unwrap() else {
        panic!("installer steps");
    };
    let version = steps[0]
        .get("env")
        .unwrap()
        .get("ACTIONLINT_VERSION")
        .and_then(Node::text)
        .unwrap();
    assert_eq!(version, "1.7.12");
    f.executable(
        "uname",
        "case $1 in -s) printf '%s\\n' Linux;; -m) printf '%s\\n' \"$ARCH\";; *) exit 91;; esac",
    );
    f.executable("curl", r#"printf '%s\0' "$@" > "$FIXTURE_ROOT/download-args"; while (( $# )); do if [[ "$1" == --output ]]; then printf 'finite archive' > "$2"; exit 0; fi; shift; done; exit 92"#);
    f.executable(
        "cargo",
        r#"
[[ "$1" == xtool && "$2" == artifact ]] || exit 94
printf '%s\n' "$3" >> "$FIXTURE_ROOT/events"
if [[ "$3" == verify-checksum ]]; then
 [[ -f "$4" && -f "$4.sha256" ]] || exit 95
 [[ "$VERIFY_FAILURE" == 0 ]] || exit 41
elif [[ "$3" == extract-tar ]]; then
 [[ -s "$4.sha256" && -d "$5" ]] || exit 96
 printf '#!/bin/bash\nprintf "fixture actionlint version\\n"\n' > "$5/actionlint"
 chmod +x "$5/actionlint"
else exit 93; fi
"#,
    );
    let mut c = Command::new("bash");
    c.args(["-c", &step(&action("install-actionlint"), "run")])
        .env_clear()
        .env(
            "PATH",
            format!("{}:/usr/bin:/bin", f.path().join("bin").display()),
        )
        .env("FIXTURE_ROOT", f.path())
        .env("RUNNER_TEMP", f.path())
        .env("GITHUB_PATH", f.path().join("paths"))
        .env("ACTIONLINT_VERSION", version)
        .env("ARCH", platform)
        .env("VERIFY_FAILURE", if verify_failure { "1" } else { "0" });
    let out = f.run(c);
    (f, out)
}

#[test]
fn pinned_installer_verifies_each_platform_checksum_before_extracting_or_exporting() {
    for (arch, label, digest) in [
        (
            "x86_64",
            "amd64",
            "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8",
        ),
        (
            "aarch64",
            "arm64",
            "325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6",
        ),
        (
            "arm64",
            "arm64",
            "325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6",
        ),
    ] {
        let (f, out) = install(arch, false);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let download = fs::read(f.path().join("download-args")).unwrap();
        let url = format!(
            "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_{label}.tar.gz"
        );
        assert!(
            download
                .split(|byte| *byte == 0)
                .any(|argument| argument == url.as_bytes())
        );
        assert_eq!(
            fs::read_to_string(f.path().join("events")).unwrap(),
            "verify-checksum\nextract-tar\n"
        );
        assert_eq!(
            fs::read_to_string(
                f.path()
                    .join(format!("actionlint_1.7.12_linux_{label}.tar.gz.sha256"))
            )
            .unwrap(),
            format!("{digest}  actionlint_1.7.12_linux_{label}.tar.gz\n")
        );
        assert_eq!(
            fs::read_to_string(f.path().join("paths")).unwrap(),
            format!("{}/actionlint-1.7.12-linux_{label}\n", f.path().display())
        );
    }
}

#[test]
fn installer_verification_failure_or_unsupported_arch_never_extracts_or_exports() {
    for (arch, verify_failure) in [("x86_64", true), ("riscv64", false)] {
        let (f, out) = install(arch, verify_failure);
        assert!(!out.status.success());
        assert!(!f.path().join("paths").exists());
        assert_eq!(
            fs::read_to_string(f.path().join("events")).unwrap_or_default(),
            if verify_failure {
                "verify-checksum\n"
            } else {
                ""
            }
        );
    }
}
