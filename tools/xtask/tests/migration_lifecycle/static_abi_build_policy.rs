//! Actual static build caller, fake finite CMake/archive/cache boundaries.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

const ARCHIVES: [&str; 8] = [
    "src/libllama.a",
    "common/libllama-common.a",
    "common/libllama-common-base.a",
    "ggml/src/libggml.a",
    "ggml/src/libggml-base.a",
    "ggml/src/ggml-cpu/libggml-cpu.a",
    "tools/mtmd/libmtmd.a",
    "vendor/hash/libvendor-hash.a",
];

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    work: PathBuf,
    build: PathBuf,
    cache: PathBuf,
}
fn executable(path: &Path, body: &str) {
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("static policy with spaces");
        let work = root.join("llama source");
        let build = root.join("ABI output");
        let cache = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("sccache tool");
        for path in [
            root.join("scripts/lib"),
            root.join("bin"),
            work.join(".git"),
            build.clone(),
        ] {
            fs::create_dir_all(path).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for relative in [
            "scripts/build-llama.sh",
            "scripts/lib/cuda-toolkit.sh",
            "scripts/lib/macos-deployment-target.sh",
        ] {
            fs::copy(source.join(relative), root.join(relative)).unwrap();
        }
        fs::write(
            work.join(".mesh-llm-patched-sha"),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n",
        )
        .unwrap();
        fs::write(root.join("outside-sentinel"), b"preserve").unwrap();
        executable(&root.join("bin/uname"), "#!/bin/sh\nprintf 'Linux\\n'\n");
        executable(&root.join("bin/ninja"), "#!/bin/sh\nexit 0\n");
        executable(&root.join("bin/cmake"), CMAKE);
        executable(&cache, SCCACHE);
        let deny =
            "#!/bin/sh\nprintf forbidden > \"$STATIC_FIXTURE_ROOT/native-called\"\nexit 98\n";
        for name in [
            "cargo", "rustc", "git", "cc", "c++", "gcc", "g++", "clang", "clang++", "make", "nvcc",
        ] {
            executable(&root.join("bin").join(name), deny);
        }
        Self {
            _temporary: temporary,
            root,
            work,
            build,
            cache,
        }
    }

    fn invoke(&self, mode: &str) -> (bool, String) {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("STATIC_FIXTURE_ROOT", self.root.display().to_string()),
            ("STATIC_CACHE_MODE", mode.to_owned()),
            ("SCCACHE", self.cache.display().to_string()),
            ("LLAMA_WORKDIR", self.work.display().to_string()),
            ("LLAMA_STAGE_BUILD_DIR", self.build.display().to_string()),
            ("LLAMA_STAGE_BACKEND", "cpu".to_owned()),
            ("LLAMA_STAGE_LINK_MODE", "static".to_owned()),
            ("LLAMA_STAGE_USE_SCCACHE", "1".to_owned()),
            ("MESH_LLM_REQUIRE_SCCACHE", "1".to_owned()),
            (
                "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
                "finite-runner-epoch".to_owned(),
            ),
            ("CMAKE_BUILD_PARALLEL_LEVEL", "1".to_owned()),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Public(v.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: vec![Value::Public(
                    self.root.join("scripts/build-llama.sh").into_os_string(),
                )],
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 16384,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(16384),
                stderr: NonZeroUsize::new(16384),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert!(!self.root.join("native-called").exists());
        assert_eq!(
            fs::read(self.root.join("outside-sentinel")).unwrap(),
            b"preserve"
        );
        assert_eq!(
            fs::read(self.work.join(".mesh-llm-patched-sha")).unwrap(),
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        );
        (
            report.process.status.unwrap().success(),
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes()).into_owned(),
        )
    }

    fn events(&self) -> Vec<String> {
        fs::read_to_string(self.root.join("events"))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn configure_args(&self) -> Vec<String> {
        fs::read(self.root.join("configure.argv"))
            .unwrap()
            .split(|b| *b == 0)
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| String::from_utf8(bytes.to_vec()).unwrap())
            .collect()
    }
    fn stamp(&self) -> String {
        fs::read_to_string(self.build.join(".mesh-llm-build-stamp")).unwrap()
    }
    fn assert_archives(&self) {
        for relative in ARCHIVES {
            assert_eq!(
                fs::read(self.build.join(relative)).unwrap(),
                b"inert static output"
            );
        }
    }
}

const CMAKE: &str = r#"#!/bin/bash
set -euo pipefail
target=''
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in -B|--build) target="${args[i+1]}" ;; esac
done
[[ -n "$target" ]] || exit 93
mkdir -p "$target"
if [[ "$1" == --build ]]; then
  printf 'cmake:build\n' >> "$STATIC_FIXTURE_ROOT/events"
  for relative in src/libllama.a common/libllama-common.a common/libllama-common-base.a ggml/src/libggml.a ggml/src/libggml-base.a ggml/src/ggml-cpu/libggml-cpu.a tools/mtmd/libmtmd.a vendor/hash/libvendor-hash.a; do
    mkdir -p "$(dirname "$target/$relative")"
    printf 'inert static output' > "$target/$relative"
  done
else
  printf 'cmake:configure\n' >> "$STATIC_FIXTURE_ROOT/events"
  printf '%s\0' "$@" > "$STATIC_FIXTURE_ROOT/configure.argv"
  printf 'CMAKE_GENERATOR:INTERNAL=Ninja\n' > "$target/CMakeCache.txt"
fi
"#;

const SCCACHE: &str = r#"#!/bin/bash
set -euo pipefail
[[ "$#" == 1 ]] || exit 94
printf 'sccache:%s\n' "$1" >> "$STATIC_FIXTURE_ROOT/events"
case "$1" in
--show-stats)
  if [[ "$STATIC_CACHE_MODE" == ready || ( "$STATIC_CACHE_MODE" == recovered && -f "$STATIC_FIXTURE_ROOT/daemon-ready" ) ]]; then exit 0; fi
  exit 1 ;;
--start-server)
  [[ "$STATIC_CACHE_MODE" != start-fails ]] || exit 1
  touch "$STATIC_FIXTURE_ROOT/daemon-ready"
  exit 0 ;;
*) exit 95 ;;
esac
"#;

#[test]
fn static_abi_build_policy_emits_static_compiler_flags_and_portable_stamp() {
    let mut stamps = Vec::new();
    for _ in 0..2 {
        let fixture = Fixture::new();
        let (success, stderr) = fixture.invoke("ready");
        assert!(success, "{stderr}");
        let args = fixture.configure_args();
        assert!(args.iter().any(|arg| arg == "-DBUILD_SHARED_LIBS=OFF"));
        for language in ["C", "CXX"] {
            let flags = args
                .iter()
                .find(|arg| arg.starts_with(&format!("-DCMAKE_{language}_FLAGS=")))
                .unwrap();
            for flag in [
                "-ffile-prefix-map",
                "-fdebug-prefix-map",
                "-fmacro-prefix-map",
            ] {
                assert!(
                    flags.contains(&format!("{flag}={}=/mesh-llm", fixture.root.display())),
                    "{flags}"
                );
            }
            assert!(args.iter().any(|arg| arg
                == &format!(
                    "-DCMAKE_{language}_COMPILER_LAUNCHER={}",
                    fixture.cache.display()
                )));
        }
        let stamp = fixture.stamp();
        assert!(stamp.lines().any(|line| line == "stamp-version=3"));
        assert!(
            stamp
                .lines()
                .any(|line| line == "toolchain-epoch=finite-runner-epoch")
        );
        for normalized in [
            "@LLAMA_BUILD_DIR@",
            "@LLAMA_WORKDIR@",
            "@MESH_LLM_ROOT@",
            "@SCCACHE@",
        ] {
            assert!(stamp.contains(normalized), "missing {normalized}: {stamp}");
        }
        assert!(
            !stamp.contains(fixture.root.to_str().unwrap()),
            "runner-local path in stamp: {stamp}"
        );
        fixture.assert_archives();
        stamps.push(stamp);
    }
    assert_eq!(
        stamps[0], stamps[1],
        "different private roots must emit identical portable identities"
    );
}

#[test]
fn static_abi_build_policy_probes_ready_daemon_or_recovers_before_configure() {
    for (mode, probes) in [
        ("ready", vec!["sccache:--show-stats"]),
        (
            "recovered",
            vec![
                "sccache:--show-stats",
                "sccache:--start-server",
                "sccache:--show-stats",
            ],
        ),
    ] {
        let fixture = Fixture::new();
        let (success, stderr) = fixture.invoke(mode);
        assert!(success, "{stderr}");
        let events = fixture.events();
        let configure = events
            .iter()
            .position(|event| event == "cmake:configure")
            .unwrap();
        assert_eq!(&events[..configure], probes.as_slice());
        assert_eq!(&events[configure..], &["cmake:configure", "cmake:build"]);
        fixture.assert_archives();
    }
}

#[test]
fn static_abi_build_policy_required_cache_refuses_before_configure() {
    for (mode, probes) in [
        (
            "start-fails",
            vec!["sccache:--show-stats", "sccache:--start-server"],
        ),
        (
            "reprobe-fails",
            vec![
                "sccache:--show-stats",
                "sccache:--start-server",
                "sccache:--show-stats",
            ],
        ),
    ] {
        let fixture = Fixture::new();
        let (success, stderr) = fixture.invoke(mode);
        assert!(!success);
        assert!(
            stderr.contains("sccache is unavailable and MESH_LLM_REQUIRE_SCCACHE=1"),
            "{stderr}"
        );
        assert_eq!(fixture.events(), probes);
        assert!(!fixture.root.join("configure.argv").exists());
        assert!(!fixture.build.join(".mesh-llm-build-stamp").exists());
        for relative in ARCHIVES {
            assert!(!fixture.build.join(relative).exists());
        }
    }
}
