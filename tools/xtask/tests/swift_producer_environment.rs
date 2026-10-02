//! Behavioral Swift producer fixtures. All native/build/automation boundaries are local fakes.
#![cfg(unix)]

use std::{
    fs,
    os::unix::{fs::PermissionsExt as _, process::CommandExt as _},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

const TARGETS: [&str; 4] = [
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "aarch64-apple-ios-macabi",
    "aarch64-apple-darwin",
];
const RECORD: &str = r#"
record() {
  printf 'mac=%s\nios=%s\n' "${MACOSX_DEPLOYMENT_TARGET-UNSET}" "${IPHONEOS_DEPLOYMENT_TARGET-UNSET}" > "$FIXTURE_ROOT/logs/$1.env"
  label="$1"; shift
  printf '%s\n' "$@" > "$FIXTURE_ROOT/logs/$label.argv"
}
"#;

struct Fixture {
    directory: tempfile::TempDir,
}

struct Receipt {
    status: ExitStatus,
    stderr: String,
}

impl Fixture {
    fn new() -> Self {
        let fixture = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        for path in ["bin", "logs", "sdk/swift/scripts", "scripts"] {
            fs::create_dir_all(fixture.path().join(path)).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        for name in ["build-xcframework.sh", "build-host-macos-xcframework.sh"] {
            fs::copy(
                source.join("sdk/swift/scripts").join(name),
                fixture.path().join("sdk/swift/scripts").join(name),
            )
            .unwrap();
        }
        fixture.install_tools();
        fixture
    }

    fn path(&self) -> &Path {
        self.directory.path()
    }

    fn executable(&self, relative: &str, body: &str) {
        let path = self.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            format!("#!/bin/bash\nset -euo pipefail\n{RECORD}\n{body}\n"),
        )
        .unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
    }

    fn install_tools(&self) {
        self.executable(
            "bin/cargo",
            r#"
case "$1" in
metadata) record metadata "$@"; printf '{"packages":[{"name":"mesh-llm-ffi"}]}\n' ;;
build)
  target=''
  for ((i=1; i<=$#; i++)); do
    if [[ "${!i}" == --target ]]; then j=$((i+1)); target="${!j}"; fi
  done
  [[ -n "$target" ]] || exit 31
  record "cargo-$target" "$@"
  mkdir -p "$CARGO_TARGET_DIR/$target/release"
  printf 'native:%s\n' "$target" > "$CARGO_TARGET_DIR/$target/release/libmeshllm_ffi.a"
  chmod 0600 "$CARGO_TARGET_DIR/$target/release/libmeshllm_ffi.a"
  ;;
*) exit 32 ;;
esac
"#,
        );
        self.executable("bin/rustup", "record rustup \"$@\"");
        self.executable(
            "home/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc",
            "exit 43",
        );
        self.executable("scripts/prepare-llama.sh", "record prepare \"$@\"");
        self.executable(
            "scripts/build-llama.sh",
            r#"
target="${LLAMA_STAGE_BUILD_DIR##*/build-stage-abi-}"; target="${target%-metal}"
record "native-$target" "$@"
[[ "$LLAMA_STAGE_BACKEND" == metal ]] || exit 44
"#,
        );
        self.executable(
            "sdk/swift/scripts/generate-swift-bindings.sh",
            r#"
record bindings "$@"
mkdir -p "$FIXTURE_ROOT/sdk/swift/Generated/FFI" "$FIXTURE_ROOT/sdk/swift/Sources/MeshLLM/Generated"
printf 'fixture header\n' > "$FIXTURE_ROOT/sdk/swift/Generated/FFI/MeshLLMFFI.h"
printf 'fixture module\n' > "$FIXTURE_ROOT/sdk/swift/Generated/FFI/MeshLLMFFI.modulemap"
printf 'fixture guards\n' > "$FIXTURE_ROOT/sdk/swift/Sources/MeshLLM/Generated/mesh_ffi.swift"
"#,
        );
        self.executable(
            "bin/automation",
            r#"
record automation "$@"
[[ "$1 $2" == 'prepared-input swift-api-checksum' ]] || exit 45
[[ -f "$3" && -f "$4" ]] || exit 46
"#,
        );
        self.executable(
            "bin/xcodebuild",
            r#"
record xcode "$@"
[[ "$1" == -create-xcframework ]] || exit 47
shift
count=0; output=''
while (( $# )); do
  case "$1" in
  -framework)
    [[ -f "$2/MeshLLMFFI" ]] || exit 48
    count=$((count+1)); shift 2 ;;
  -output) output="$2"; shift 2 ;;
  *) exit 49 ;;
  esac
done
[[ "$count" == 4 && -n "$output" ]] || exit 50
mkdir -p "$output/fixture"
cp "$FIXTURE_ROOT/sdk/swift/PrivacyInfo.xcprivacy" "$output/fixture/PrivacyInfo.xcprivacy"
"#,
        );
        // Linux's ln lacks macOS -h; preserve the symlink operation at this OS boundary.
        self.executable(
            "bin/ln",
            r#"if [[ "$1" == -sfh ]]; then shift; exec /bin/ln -sfn "$@"; fi
exec /bin/ln "$@""#,
        );
        fs::write(
            self.path().join("sdk/swift/PrivacyInfo.xcprivacy"),
            "fixture privacy\n",
        )
        .unwrap();
    }

    fn execute(&self, args: &[&str], mac: &str, ios: &str) -> Receipt {
        let stdout = fs::File::create(self.path().join("stdout")).unwrap();
        let stderr = fs::File::create(self.path().join("stderr")).unwrap();
        let mut child = Command::new("/bin/bash")
            .arg(self.path().join("sdk/swift/scripts/build-xcframework.sh"))
            .args(args)
            .current_dir(self.path())
            .env_clear()
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.path().join("bin").display()),
            )
            .env("HOME", self.path().join("home"))
            .env("FIXTURE_ROOT", self.path())
            .env("CARGO_TARGET_DIR", self.path().join("target"))
            .env("SWIFT_TARGET_OUTPUT_DIR", self.path().join("staged"))
            .env(
                "MESH_LLM_AUTOMATION_BIN",
                self.path().join("bin/automation"),
            )
            .env("MACOSX_DEPLOYMENT_TARGET", mac)
            .env("IPHONEOS_DEPLOYMENT_TARGET", ios)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(stderr)
            .process_group(0)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                let _cleanup = Command::new("/bin/kill")
                    .args(["-KILL", "--", &format!("-{}", child.id())])
                    .status();
                let _reap = child.wait();
                panic!(
                    "Swift fixture exceeded deadline: {}",
                    fs::read_to_string(self.path().join("stderr")).unwrap()
                );
            }
            thread::sleep(Duration::from_millis(10));
        };
        Receipt {
            status,
            stderr: fs::read_to_string(self.path().join("stderr")).unwrap(),
        }
    }

    fn log(&self, label: &str, kind: &str) -> String {
        fs::read_to_string(self.path().join(format!("logs/{label}.{kind}"))).unwrap()
    }

    fn staged_input(&self, target: &str) -> PathBuf {
        let directory = self.path().join("assembly-input").join(target);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("libmeshllm_ffi.a");
        fs::write(&path, format!("assembled:{target}\n")).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        path
    }
}

#[test]
fn deployment_overrides_are_not_exported_to_prepare_or_native_children() {
    let fixture = Fixture::new();
    let result = fixture.execute(&["--target", TARGETS[3]], "14.6", "17.2");
    assert!(result.status.success(), "{}", result.stderr);
    for label in ["prepare", "rustup", "native-aarch64-apple-darwin"] {
        assert_eq!(fixture.log(label, "env"), "mac=UNSET\nios=UNSET\n");
    }
}

#[test]
fn cargo_children_receive_only_their_platform_deployment_override() {
    for target in TARGETS {
        let fixture = Fixture::new();
        let result = fixture.execute(&["--target", target], "14.6", "17.2");
        assert!(result.status.success(), "{}", result.stderr);
        let expected = if target.ends_with("darwin") {
            "mac=14.6\nios=UNSET\n"
        } else {
            "mac=UNSET\nios=17.2\n"
        };
        assert_eq!(fixture.log(&format!("cargo-{target}"), "env"), expected);
        let args = fixture.log(&format!("cargo-{target}"), "argv");
        assert!(args.contains(&format!("--target\n{target}\n")));
        assert!(args.contains("--no-default-features\n--features\nembedded-runtime\n"));
    }
}

#[test]
fn macos_native_effective_deployment_target_matches_cargo_override() {
    let fixture = Fixture::new();
    let result = fixture.execute(&["--target", TARGETS[3]], "14.6", "17.2");
    assert!(result.status.success(), "{}", result.stderr);
    let args = fixture.log("native-aarch64-apple-darwin", "argv");
    let values: Vec<_> = args
        .lines()
        .filter_map(|line| line.strip_prefix("-DCMAKE_OSX_DEPLOYMENT_TARGET="))
        .collect();
    assert_eq!(values.last(), Some(&"14.6"));
    assert!(
        args.lines()
            .any(|line| line == "-DCMAKE_OSX_SYSROOT=macosx")
    );
    assert!(
        args.lines()
            .any(|line| line == "-DCMAKE_OSX_ARCHITECTURES=arm64")
    );
    assert_eq!(
        fixture.log("cargo-aarch64-apple-darwin", "env"),
        "mac=14.6\nios=UNSET\n"
    );
}

#[test]
fn target_mode_stages_one_selected_library_with_exact_bytes_and_mode() {
    let fixture = Fixture::new();
    let target = TARGETS[1];
    let result = fixture.execute(&["--target", target], "14.6", "17.2");
    assert!(result.status.success(), "{}", result.stderr);
    let staged = fixture
        .path()
        .join("staged")
        .join(target)
        .join("libmeshllm_ffi.a");
    assert_eq!(
        fs::read(&staged).unwrap(),
        format!("native:{target}\n").as_bytes()
    );
    assert_eq!(
        fs::metadata(staged).unwrap().permissions().mode() & 0o777,
        0o644
    );
    assert_eq!(
        fs::read_dir(fixture.path().join("staged")).unwrap().count(),
        1
    );
    assert!(!fixture.path().join("logs/bindings.env").exists());
    assert!(!fixture.path().join("logs/xcode.env").exists());
    assert_eq!(
        fixture.log("rustup", "argv"),
        format!("target\nadd\n{target}\n")
    );
}

#[test]
fn assembly_requires_all_target_inputs_and_preserves_them_without_rebuilding() {
    for missing in TARGETS {
        let fixture = Fixture::new();
        for target in TARGETS.into_iter().filter(|target| *target != missing) {
            fixture.staged_input(target);
        }
        let inputs = fixture.path().join("assembly-input");
        let result = fixture.execute(
            &["--assemble-from", inputs.to_str().unwrap()],
            "14.6",
            "17.2",
        );
        assert!(!result.status.success());
        assert!(
            result
                .stderr
                .contains("staged Swift target library is missing"),
            "{}",
            result.stderr
        );
        assert!(result.stderr.contains(missing));
        assert!(!fixture.path().join("logs/bindings.env").exists());
        assert!(!fixture.path().join("logs/xcode.env").exists());
    }
    let fixture = Fixture::new();
    for target in TARGETS {
        fixture.staged_input(target);
    }
    let inputs = fixture.path().join("assembly-input");
    let result = fixture.execute(
        &["--assemble-from", inputs.to_str().unwrap()],
        "14.6",
        "17.2",
    );
    assert!(result.status.success(), "{}", result.stderr);
    for target in TARGETS {
        let restored = fixture
            .path()
            .join("target")
            .join(target)
            .join("release/libmeshllm_ffi.a");
        assert_eq!(
            fs::read(&restored).unwrap(),
            format!("assembled:{target}\n").as_bytes()
        );
        assert_eq!(
            fs::metadata(restored).unwrap().permissions().mode() & 0o777,
            0o644
        );
        assert!(
            !fixture
                .path()
                .join(format!("logs/cargo-{target}.env"))
                .exists()
        );
        assert!(
            !fixture
                .path()
                .join(format!("logs/native-{target}.env"))
                .exists()
        );
    }
    assert!(!fixture.path().join("logs/prepare.env").exists());
    assert!(fixture.path().join("logs/xcode.env").exists());
}
