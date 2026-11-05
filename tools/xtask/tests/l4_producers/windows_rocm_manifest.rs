//! Exercise the consumed manifest, with no HIP SDK or native compiler required.
use process_supervision as process;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

const KERNELS: &str = "lib/rocblas/library";
const ABC_SHA: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

struct Fixture {
    directory: tempfile::TempDir,
    arguments: Vec<String>,
}

impl Fixture {
    fn new(os: &str, backend: &str, rocblas: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("lib")).unwrap();
        fs::write(directory.path().join("lib/llama.dll"), b"abc").unwrap();
        let mut arguments: Vec<String> = [
            "native",
            "runtime-manifest-write",
            "--schema-v2",
            directory.path().join("manifest.json").to_str().unwrap(),
            "runtime-fixture",
            "0.80.0",
            "1.2.3",
            os,
            "x86_64",
            if os == "windows" {
                "x86_64-pc-windows-msvc"
            } else {
                "x86_64-unknown-linux-gnu"
            },
            os,
            backend,
            "13",
            "lib/llama.dll",
            "upstream",
            "patched",
            "queue",
            "lib/llama.dll",
        ]
        .map(str::to_owned)
        .into();
        if rocblas {
            fs::write(directory.path().join("lib/rocblas.dll"), b"abc").unwrap();
            arguments.push("lib/rocblas.dll".into());
        }
        arguments.extend(["--", "--", "--"].map(str::to_owned));
        Self {
            directory,
            arguments,
        }
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }

    fn file(&self, relative: &str) {
        let path = self.root().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"abc").unwrap();
    }

    fn object(&self, name: &str) {
        self.file(&format!("{KERNELS}/{name}"));
    }

    fn run(&self, extra: &[(&str, &str)]) -> (bool, String) {
        // ProcessSpec clears the inherited environment. Keep only OS loader
        // inputs and this fixture's explicit configuration, never caller targets.
        let mut environment: BTreeMap<_, _> = ["PATH", "SystemRoot", "WINDIR"]
            .into_iter()
            .filter_map(|key| {
                std::env::var_os(key).map(|value| (key.into(), process::Value::Public(value)))
            })
            .collect();
        for (key, value) in extra {
            environment.insert((*key).into(), process::Value::Public((*value).into()));
        }
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                cwd: self.root().into(),
                environment,
                arguments: self
                    .arguments
                    .iter()
                    .map(|value| process::Value::Public(value.into()))
                    .collect(),
            },
            &process::Limits {
                execution: Duration::from_secs(60),
                graceful_shutdown: Duration::from_secs(2),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        let report = raw.process;
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none() && report.cleanup.complete,
            "{report:?}"
        );
        let stdout = raw.stdout.unwrap();
        let stderr = raw.stderr.unwrap();
        assert_eq!(stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
        assert_eq!(stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
        assert!(stdout.as_bytes().is_empty());
        (
            report.status.unwrap().success(),
            String::from_utf8(stderr.as_bytes().to_vec()).unwrap(),
        )
    }

    fn manifest(&self) -> Value {
        serde_json::from_slice(&fs::read(self.root().join("manifest.json")).unwrap()).unwrap()
    }

    fn write(&self, extra: &[(&str, &str)]) -> (Value, String) {
        let (ok, stderr) = self.run(extra);
        assert!(ok, "{stderr}");
        (self.manifest(), stderr)
    }

    fn refuses_without_replacing(&self, expected_error: &str) {
        let path = self.root().join("manifest.json");
        for prior in [None, Some(b"prior manifest".as_slice())] {
            if let Some(bytes) = prior {
                fs::write(&path, bytes).unwrap();
            }
            let (ok, stderr) = self.run(&[("LLAMA_STAGE_AMDGPU_TARGETS", "gfx1100")]);
            assert!(!ok);
            assert!(stderr.contains(expected_error), "{stderr}");
            assert_eq!(fs::read(&path).ok().as_deref(), prior);
        }
    }
}

#[test]
fn windows_rocm_manifest_binds_code_objects_and_drops_unsupported_targets() {
    let fixture = Fixture::new("windows", "rocm", true);
    fixture.object("kernel-gfx1100.co");
    fixture.object("kernel-gfx1201.hsaco");
    fixture.object("TensileLibrary_lazy_gfx1102.dat");
    fixture.file("lib/hipblaslt/library/kernel-gfx1103.hsaco");
    let (manifest, stderr) = fixture.write(&[
        (
            "LLAMA_STAGE_AMDGPU_TARGETS",
            "gfx90a; gfx1201, gfx1102;gfx1103;gfx1100",
        ),
        ("MESH_LLM_ROCM_VERSION", "6.4"),
    ]);
    assert_eq!(
        manifest["runtime"]["backend"],
        json!({"kind":"rocm", "rocm":{"gpu_arches":["gfx1201","gfx1100"],"version":"6.4"}})
    );
    assert!(stderr.contains("not advertising gfx90a gfx1102 gfx1103"));
    assert_eq!(manifest["runtime"]["release_version"], "0.80.0");
    assert_eq!(manifest["runtime"]["skippy_abi"], "1.2.3");
    assert_eq!(manifest["build"]["backend"], "rocm");
    assert_eq!(
        manifest["runtime"]["libraries"],
        json!(["lib/llama.dll", "lib/rocblas.dll"])
    );
    assert_eq!(
        manifest["runtime"]["files"],
        json!({
            "lib/llama.dll":ABC_SHA, "lib/rocblas.dll":ABC_SHA,
            "lib/rocblas/library/kernel-gfx1100.co":ABC_SHA,
            "lib/rocblas/library/kernel-gfx1201.hsaco":ABC_SHA
        })
    );
}

#[test]
fn windows_rocm_manifest_preserves_parser_precedence_and_hip_alias() {
    for (primary, fallback, expected) in [
        (
            Some("gfx1100,gfx1100"),
            Some("gfx1201"),
            json!(["gfx1100", "gfx1100"]),
        ),
        (None, Some("gfx1201;gfx1100"), json!(["gfx1201", "gfx1100"])),
        (Some(""), Some("gfx1201"), json!(["gfx1201"])),
        (Some(" ; , "), Some("gfx1201"), json!([])),
        (None, None, json!([])),
    ] {
        let fixture = Fixture::new("windows", "hip", true);
        fixture.object("kernel-gfx1100.co");
        fixture.object("kernel-gfx1201.hsaco");
        let extra: Vec<_> = primary
            .map(|value| ("LLAMA_STAGE_AMDGPU_TARGETS", value))
            .into_iter()
            .chain(fallback.map(|value| ("SKIPPY_AMDGPU_TARGETS", value)))
            .collect();
        let (manifest, stderr) = fixture.write(&extra);
        assert!(stderr.is_empty(), "{stderr}");
        assert_eq!(
            manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
            expected
        );
        assert_eq!(manifest["build"]["backend"], "hip");
    }
}

#[test]
fn windows_rocm_manifest_keeps_unconfigured_and_non_rocblas_packages_unchanged() {
    let fixture = Fixture::new("windows", "rocm", true);
    let (manifest, stderr) = fixture.write(&[]);
    assert!(stderr.is_empty());
    assert_eq!(
        manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
        json!([])
    );
    assert_eq!(manifest["runtime"]["files"].as_object().unwrap().len(), 2);

    let fixture = Fixture::new("windows", "rocm", false);
    fixture.object("kernel-gfx1201.co");
    let (manifest, stderr) = fixture.write(&[("LLAMA_STAGE_AMDGPU_TARGETS", "gfx1100")]);
    assert!(stderr.is_empty());
    assert_eq!(
        manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
        json!(["gfx1100"])
    );
    assert_eq!(manifest["runtime"]["files"].as_object().unwrap().len(), 1);
}

#[test]
fn windows_rocm_manifest_leaves_other_backends_and_linux_rocm_unchanged() {
    for (os, backend, expected) in [
        ("windows", "cpu", json!({"kind":"cpu"})),
        ("windows", "vulkan", json!({"kind":"vulkan","vulkan":{}})),
        (
            "windows",
            "cuda",
            json!({"kind":"cuda","cuda":{"toolkit_major":13,"gpu_arches":[]}}),
        ),
        (
            "linux",
            "rocm",
            json!({"kind":"rocm","rocm":{"gpu_arches":["gfx1100"]}}),
        ),
    ] {
        let fixture = Fixture::new(os, backend, true);
        // This deliberately cannot supply gfx1100. Other paths must not inspect it.
        fixture.object("kernel-gfx1201.co");
        let (manifest, stderr) = fixture.write(&[("LLAMA_STAGE_AMDGPU_TARGETS", "gfx1100")]);
        assert!(stderr.is_empty());
        assert_eq!(manifest["runtime"]["backend"], expected);
        assert_eq!(manifest["runtime"]["files"].as_object().unwrap().len(), 2);
    }
}

#[test]
fn windows_rocm_manifest_requires_literal_architecture_boundaries_and_code_object_suffixes() {
    for (name, admitted) in [
        ("gfx1100.co", true),
        ("kernel-gfx1100_xnack-.hsaco", true),
        ("kernel-gfx1100.CO", true),
        ("kernel-gfx1100.HsAcO", true),
        ("kernel-gfx11000.co", false),
        ("kernel-agfx1100.co", false),
        ("kernel-gfx1100a.co", false),
        ("kernel-gfx1100A.co", false),
        ("kernel-GFX1100.co", false),
        ("kernel-gfx1100.dat", false),
    ] {
        let fixture = Fixture::new("windows", "rocm", true);
        fixture.object("kernel-gfx1201.co");
        fixture.object(name);
        let (manifest, _) = fixture.write(&[("LLAMA_STAGE_AMDGPU_TARGETS", "gfx1100;gfx1201")]);
        let expected = if admitted {
            json!(["gfx1100", "gfx1201"])
        } else {
            json!(["gfx1201"])
        };
        assert_eq!(
            manifest["runtime"]["backend"]["rocm"]["gpu_arches"], expected,
            "{name}"
        );
    }
    let fixture = Fixture::new("windows", "rocm", true);
    fixture.object("kernel-gfx1100.co");
    let (ok, stderr) = fixture.run(&[("LLAMA_STAGE_AMDGPU_TARGETS", "gfx11.*")]);
    assert!(!ok && stderr.contains("no rocBLAS kernels"), "{stderr}");
}

#[test]
fn windows_rocm_manifest_refuses_missing_empty_catalog_only_or_other_library_kernels() {
    for contents in ["missing", "empty", "catalog", "hipblaslt"] {
        let fixture = Fixture::new("windows", "rocm", true);
        if contents != "missing" {
            fs::create_dir_all(fixture.root().join(KERNELS)).unwrap();
        }
        match contents {
            "catalog" => fixture.object("TensileLibrary_lazy_gfx1100.dat"),
            "hipblaslt" => fixture.file("lib/hipblaslt/library/kernel-gfx1100.hsaco"),
            _ => (),
        }
        fixture.refuses_without_replacing(if contents == "missing" {
            "cannot inspect packaged rocBLAS kernels"
        } else {
            "no rocBLAS kernels for any ROCm target (gfx1100)"
        });
    }
}

#[test]
fn windows_rocm_manifest_refuses_directory_named_as_code_object() {
    let fixture = Fixture::new("windows", "rocm", true);
    fs::create_dir_all(fixture.root().join(KERNELS).join("kernel-gfx1100.co")).unwrap();
    fixture.refuses_without_replacing("not a file");
}

#[cfg(unix)]
#[test]
fn windows_rocm_manifest_refuses_escaping_objects_and_unsafe_names() {
    use std::os::unix::fs::symlink;
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("kernel-gfx1100.co"), b"abc").unwrap();
    for directory_link in [false, true] {
        let fixture = Fixture::new("windows", "rocm", true);
        if directory_link {
            fs::create_dir(fixture.root().join("lib/rocblas")).unwrap();
            symlink(outside.path(), fixture.root().join(KERNELS)).unwrap();
        } else {
            fs::create_dir_all(fixture.root().join(KERNELS)).unwrap();
            symlink(
                outside.path().join("kernel-gfx1100.co"),
                fixture.root().join(KERNELS).join("kernel-gfx1100.co"),
            )
            .unwrap();
        }
        fixture.refuses_without_replacing("escapes root");
    }
    let fixture = Fixture::new("windows", "rocm", true);
    fixture.object("unsafe:gfx1100.co");
    fixture.refuses_without_replacing("unsafe artifact path");
}
