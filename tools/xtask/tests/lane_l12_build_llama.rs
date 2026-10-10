#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Fixture {
    directory: tempfile::TempDir,
    build: PathBuf,
    work: PathBuf,
    tools: PathBuf,
}

impl Fixture {
    fn new(ninja: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let work = directory.path().join("llama");
        let build = directory.path().join("build");
        let tools = directory.path().join("tools");
        fs::create_dir_all(work.join(".git")).unwrap();
        fs::create_dir(&build).unwrap();
        fs::create_dir(&tools).unwrap();
        fs::write(
            work.join(".mesh-llm-patched-sha"),
            format!("{}\n", "0".repeat(40)),
        )
        .unwrap();
        fs::write(build.join("marker"), "warm").unwrap();
        write_tool(
            &tools.join("cmake"),
            include_str!("lane_l12_build_llama/cmake.sh"),
        );
        write_tool(
            &tools.join("uname"),
            "#!/bin/sh\nprintf '%s\\n' \"$FIXTURE_OS\"\n",
        );
        write_tool(
            &tools.join("nvcc"),
            "#!/bin/sh\necho 'Cuda compilation tools, release 12.9, V12.9.0'\n",
        );
        if ninja {
            write_tool(&tools.join("ninja"), "#!/bin/sh\nexit 0\n");
        }
        Self {
            directory,
            build,
            work,
            tools,
        }
    }
    fn command(&self, os: &str) -> Command {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut command = Command::new("/bin/bash");
        command
            .arg(root.join("scripts/build-llama.sh"))
            .current_dir(root);
        for (key, _) in std::env::vars_os() {
            let name = key.to_string_lossy();
            if name.starts_with("LLAMA_")
                || name.starts_with("SKIPPY_")
                || name.starts_with("CMAKE_")
                || matches!(
                    name.as_ref(),
                    "MACOSX_DEPLOYMENT_TARGET" | "CUDACXX" | "NVCC" | "CUDAToolkit_ROOT"
                )
            {
                command.env_remove(key);
            }
        }
        command
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin:/usr/sbin:/sbin", self.tools.display()),
            )
            .env("FIXTURE_OS", os)
            .env("LLAMA_WORKDIR", &self.work)
            .env("LLAMA_STAGE_BUILD_DIR", &self.build)
            .env("LLAMA_STAGE_BACKEND", "cpu")
            .env("LLAMA_STAGE_LINK_MODE", "dynamic")
            .env("LLAMA_STAGE_USE_SCCACHE", "0")
            .env("CMAKE_BUILD_PARALLEL_LEVEL", "1")
            .env("CMAKE_STUB_LOG", self.directory.path().join("calls"));
        command
    }
    fn log(&self) -> String {
        fs::read_to_string(self.directory.path().join("calls")).unwrap()
    }
    fn stamp(&self) -> String {
        fs::read_to_string(self.build.join(".mesh-llm-build-stamp")).unwrap()
    }
    fn cache(&self, generator: &str) {
        fs::write(
            self.build.join("CMakeCache.txt"),
            format!("CMAKE_GENERATOR:INTERNAL={generator}\n"),
        )
        .unwrap();
    }
}

fn write_tool(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn default_macos_target_reaches_configure_and_stamp() {
    let fixture = Fixture::new(true);
    success(fixture.command("Darwin").output().unwrap());
    assert!(fixture.log().contains("-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3"));
    assert!(
        fixture
            .stamp()
            .contains("-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3")
    );
}
#[test]
fn explicit_macos_target_reaches_configure_and_stamp() {
    let fixture = Fixture::new(true);
    success(
        fixture
            .command("Darwin")
            .env("MACOSX_DEPLOYMENT_TARGET", "14.0")
            .output()
            .unwrap(),
    );
    assert!(fixture.log().contains("-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0"));
    assert!(
        fixture
            .stamp()
            .contains("-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0")
    );
}
#[test]
fn mobile_target_follows_native_default() {
    let fixture = Fixture::new(true);
    success(
        fixture
            .command("Darwin")
            .args([
                "-DCMAKE_SYSTEM_NAME=iOS",
                "-DCMAKE_OSX_SYSROOT=iphoneos",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=16.0",
            ])
            .output()
            .unwrap(),
    );
    let log = fixture.log();
    assert!(
        log.find("DEPLOYMENT_TARGET=13.3").unwrap() < log.find("DEPLOYMENT_TARGET=16.0").unwrap()
    );
}
#[test]
fn linux_omits_macos_target() {
    let fixture = Fixture::new(true);
    success(
        fixture
            .command("Linux")
            .env("MACOSX_DEPLOYMENT_TARGET", "14.0")
            .output()
            .unwrap(),
    );
    assert!(!fixture.log().contains("CMAKE_OSX_DEPLOYMENT_TARGET"));
}
#[test]
fn cuda_disables_graph_capture() {
    let fixture = Fixture::new(true);
    success(
        fixture
            .command("Linux")
            .env("LLAMA_STAGE_BACKEND", "cuda")
            .output()
            .unwrap(),
    );
    assert!(fixture.log().contains("-DGGML_CUDA=ON"));
    assert!(fixture.stamp().contains("cmake-arg=-DGGML_CUDA_GRAPHS=OFF"));
}
#[test]
fn mismatched_generator_clears_cache() {
    let fixture = Fixture::new(true);
    fixture.cache("Unix Makefiles");
    success(fixture.command("Linux").output().unwrap());
    assert!(!fixture.build.join("marker").exists());
    assert_eq!(
        fs::read_to_string(fixture.build.join("CMakeCache.txt")).unwrap(),
        "CMAKE_GENERATOR:INTERNAL=Ninja\n"
    );
}
#[test]
fn absent_ninja_selects_makefiles() {
    let fixture = Fixture::new(false);
    success(fixture.command("Linux").output().unwrap());
    assert!(fixture.log().contains("-G Unix Makefiles"));
}
#[test]
fn inherited_generator_does_not_override_selection() {
    let fixture = Fixture::new(false);
    success(
        fixture
            .command("Linux")
            .env("CMAKE_GENERATOR", "Ninja")
            .output()
            .unwrap(),
    );
    assert_eq!(
        fs::read_to_string(fixture.build.join("CMakeCache.txt")).unwrap(),
        "CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n"
    );
}
#[test]
fn matching_generator_preserves_cache() {
    let fixture = Fixture::new(true);
    fixture.cache("Ninja");
    success(fixture.command("Linux").output().unwrap());
    assert!(fixture.build.join("marker").exists());
}
#[test]
fn require_existing_preserves_stale_cache() {
    let fixture = Fixture::new(true);
    fixture.cache("Unix Makefiles");
    let output = fixture
        .command("Linux")
        .arg("--require-existing")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(fixture.build.join("marker").exists());
}
#[test]
fn deployment_change_rejects_warm_build_without_configuring() {
    let fixture = Fixture::new(true);
    success(fixture.command("Darwin").output().unwrap());
    success(
        Command::new("/usr/bin/git")
            .args(["init", "-q"])
            .arg(&fixture.work)
            .output()
            .unwrap(),
    );
    success(
        Command::new("/usr/bin/git")
            .arg("-C")
            .arg(&fixture.work)
            .args(["add", "."])
            .output()
            .unwrap(),
    );
    success(
        Command::new("/usr/bin/git")
            .arg("-C")
            .arg(&fixture.work)
            .args([
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "fixture",
            ])
            .output()
            .unwrap(),
    );
    success(
        fixture
            .command("Darwin")
            .arg("--require-existing")
            .output()
            .unwrap(),
    );
    let before = fixture.log();
    let output = fixture
        .command("Darwin")
        .env("MACOSX_DEPLOYMENT_TARGET", "14.0")
        .arg("--require-existing")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(fixture.log(), before);
    success(
        fixture
            .command("Darwin")
            .env("MACOSX_DEPLOYMENT_TARGET", "14.0")
            .output()
            .unwrap(),
    );
    assert!(fixture.stamp().contains("DEPLOYMENT_TARGET=14.0"));
}
