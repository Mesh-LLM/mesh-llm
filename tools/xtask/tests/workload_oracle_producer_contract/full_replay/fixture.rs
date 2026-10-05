use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) struct Fixture {
    pub(super) directory: tempfile::TempDir,
    pub(super) build: PathBuf,
    pub(super) work: PathBuf,
    pub(super) tools: PathBuf,
}

impl Fixture {
    pub(super) fn new(ninja: bool) -> Self {
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
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/lane_l12_build_llama/cmake.sh"
            )),
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
    pub(super) fn command(&self, os: &str) -> Command {
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
}

pub(super) fn write_tool(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
