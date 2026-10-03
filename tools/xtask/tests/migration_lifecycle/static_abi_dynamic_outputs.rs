//! Actual dynamic output owner, inert filesystem libraries and simulated uname.
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

const PREFIXED: [&str; 3] = ["libllama.dll", "libllama-common.dll", "libmtmd.dll"];
const UNPREFIXED: [&str; 3] = ["llama.dll", "llama-common.dll", "mtmd.dll"];
const MIXED: [&str; 3] = ["libllama.dll", "llama-common.dll", "libmtmd.dll"];
const LINUX: [&str; 3] = ["libllama.so", "libllama-common.so", "libmtmd.so"];
const DARWIN: [&str; 3] = ["libllama.dylib", "libllama-common.dylib", "libmtmd.dylib"];

struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    build: PathBuf,
}

impl Fixture {
    fn new(names: &[&str; 3]) -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary
            .path()
            .canonicalize()
            .unwrap()
            .join("private dynamic probe");
        let build = root.join("build with spaces");
        for path in [&build, &root.join("bin")] {
            fs::create_dir_all(path).unwrap();
        }
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/build-llama.sh");
        let source = fs::read_to_string(source).unwrap();
        let start = source.find("dynamic_library_name_groups() {\n").unwrap();
        let end = source[start..]
            .find("\nif [[ -z \"${LLAMA_BUILD_DIR:-}\" ]]")
            .unwrap()
            + start;
        let owned = &source[start..end];
        assert_eq!(
            owned
                .matches("required_dynamic_libraries_exist() {")
                .count(),
            1
        );
        assert_eq!(owned.matches("required_outputs_exist() {").count(), 1);
        let driver = format!(
            "set -euo pipefail\nLLAMA_LINK_MODE=dynamic\nLLAMA_STAGE_WORKLOAD_ORACLE=OFF\n{owned}\nrequired_outputs_exist\nprintf 'admitted\\n'\n"
        );
        fs::write(root.join("driver.sh"), driver).unwrap();
        let uname = root.join("bin/uname");
        fs::write(&uname, "#!/bin/sh\n[ \"$#\" -eq 1 ] && [ \"$1\" = -s ] || exit 99\nprintf '%s\\n' \"$PROBE_OS\"\n").unwrap();
        fs::set_permissions(uname, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(root.join("outside-sentinel"), b"preserve").unwrap();
        for name in names {
            fs::write(build.join(name), b"inert library output").unwrap();
        }
        Self {
            _temporary: temporary,
            root,
            build,
        }
    }

    fn large_matching_tree(&self, names: &[&str; 3]) {
        for name in names {
            fs::remove_file(self.build.join(name)).unwrap();
        }
        let mut candidate_listing_bytes = 0;
        for index in 0..1024 {
            let directory = self.build.join(format!(
                "{index:04}-{}",
                "finite-output-directory-".repeat(4)
            ));
            fs::create_dir(&directory).unwrap();
            for name in names {
                let path = directory.join(name);
                fs::write(&path, b"inert library output").unwrap();
                if *name == names[0] {
                    candidate_listing_bytes += path.as_os_str().len() + 1;
                }
            }
        }
        // A naive full matching listing exceeds common pipe buffers; the real
        // probe must stop find at the first match rather than closing a consumer early.
        assert!(candidate_listing_bytes > 128 * 1024);
    }

    fn check(&self, os: &str, expected: bool) {
        let environment: BTreeMap<_, _> = [
            (
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.join("bin").display()),
            ),
            ("PROBE_OS", os.to_owned()),
            ("LLAMA_BUILD_DIR", self.build.display().to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                arguments: vec![Value::Public(self.root.join("driver.sh").into_os_string())],
                environment,
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 8192,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(8192),
                stderr: NonZeroUsize::new(8192),
            },
        )
        .unwrap();
        assert_eq!(report.process.outcome, Outcome::Exited);
        assert!(report.process.failure.is_none());
        assert!(report.process.cleanup.complete);
        assert_eq!(
            report.process.status.unwrap().success(),
            expected,
            "{os}: {}",
            String::from_utf8_lossy(report.stderr.as_ref().unwrap().as_bytes())
        );
        assert_eq!(
            report.stdout.as_ref().unwrap().as_bytes(),
            if expected {
                b"admitted\n".as_slice()
            } else {
                b"".as_slice()
            }
        );
        assert_eq!(
            fs::read(self.root.join("outside-sentinel")).unwrap(),
            b"preserve"
        );
    }
}

#[test]
fn static_abi_dynamic_output_probe_accepts_windows_naming_alternatives() {
    for os in ["MINGW64_NT-10.0", "MSYS_NT-10.0", "CYGWIN_NT-10.0"] {
        for names in [&PREFIXED, &UNPREFIXED, &MIXED] {
            let fixture = Fixture::new(names);
            fixture.check(os, true);
            for name in names {
                assert_eq!(
                    fs::read(fixture.build.join(name)).unwrap(),
                    b"inert library output"
                );
            }
        }
    }
}

#[test]
fn static_abi_dynamic_output_probe_refuses_each_missing_library() {
    for (os, names) in [
        ("Linux", &LINUX),
        ("Darwin", &DARWIN),
        ("MINGW64_NT-10.0", &PREFIXED),
        ("MSYS_NT-10.0", &UNPREFIXED),
        ("CYGWIN_NT-10.0", &MIXED),
    ] {
        for omitted in names {
            let fixture = Fixture::new(names);
            fs::remove_file(fixture.build.join(omitted)).unwrap();
            fixture.check(os, false);
            for name in names.iter().filter(|name| *name != omitted) {
                assert_eq!(
                    fs::read(fixture.build.join(name)).unwrap(),
                    b"inert library output"
                );
            }
        }
    }
}

#[test]
fn static_abi_dynamic_output_probe_large_tree_survives_pipefail() {
    for (os, names) in [("Linux", &LINUX), ("MINGW64_NT-10.0", &PREFIXED)] {
        let fixture = Fixture::new(names);
        fixture.large_matching_tree(names);
        fixture.check(os, true);
        assert_eq!(fs::read_dir(&fixture.build).unwrap().count(), 1024);
    }
}
