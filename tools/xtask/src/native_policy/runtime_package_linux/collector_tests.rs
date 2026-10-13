//! Finite collector contracts. ELF bytes are inert; no native inspector runs.
use crate::native_policy::runtime_package;
use crate::native_policy::toolchain::{Captured, Exit, Toolchain};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};

const LIBRARY: &str = "lib/libllama.so";
const TOOL: &str = "tools/mesh-llm-gpu-bench";
const NORMAL: &str = "Version needs section '.gnu.version_r'\nName: GLIBC_2.35\n";
const OLDER: &str = "Version needs section '.gnu.version_r'\nName: GLIBC_2.17\n";
const RELR: &str =
    "Version needs section '.gnu.version_r'\nName: GLIBC_2.35\nName: GLIBC_ABI_DT_RELR\n";

struct Inspector {
    root: PathBuf,
    library: &'static str,
    tool: &'static str,
    calls: RefCell<Vec<(String, String)>>,
}

impl Toolchain for Inspector {
    fn which(&self, program: &str) -> bool {
        program == "readelf"
    }

    fn capture(&self, argv: &[String]) -> std::io::Result<Captured> {
        if argv.len() != 3 || argv[0] != "readelf" {
            return Err(std::io::ErrorKind::InvalidInput.into());
        }
        let relative = Path::new(&argv[2])
            .strip_prefix(&self.root)
            .map_err(|_| std::io::ErrorKind::InvalidInput)?
            .to_str()
            .ok_or(std::io::ErrorKind::InvalidInput)?
            .replace('\\', "/");
        let version = match relative.as_str() {
            LIBRARY => self.library,
            TOOL => self.tool,
            _ => return Err(std::io::ErrorKind::NotFound.into()),
        };
        let output = match argv[1].as_str() {
            "-V" => version,
            "-d" => "Dynamic section contains no packaged dependencies\n",
            _ => return Err(std::io::ErrorKind::InvalidInput.into()),
        };
        self.calls
            .borrow_mut()
            .push((argv[1].clone(), relative.to_owned()));
        Ok(Captured {
            output: output.as_bytes().to_vec(),
            exit: Exit::Code(0),
        })
    }
}

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    manifest: Vec<u8>,
    bytes: Vec<u8>,
}

impl Fixture {
    fn new(declared: Option<&str>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let id = "meshllm-native-runtime-linux-x86_64-cpu";
        let root = directory.path().join(id);
        let bytes = b"\x7fELFinert collector input".to_vec();
        let checksum = hex::encode(Sha256::digest(&bytes));
        for relative in [LIBRARY, TOOL] {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, &bytes).unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root.join(TOOL), fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut document = json!({
            "schema_version": 2,
            "runtime": {
                "id": id, "release_version": "fixture", "skippy_abi": "fixture",
                "platform": {"os": "linux", "arch": "x86_64", "target": "x86_64-unknown-linux-gnu"},
                "backend": {"kind": "cpu"}, "libraries": [LIBRARY],
                "files": {(LIBRARY): checksum.clone()}, "tools": {(TOOL): checksum.clone()}
            },
            "build": {"primary_library": LIBRARY, "library_sha256": checksum}
        });
        if let Some(declared) = declared {
            document["runtime"]["platform"]["min_glibc"] = json!(declared);
        }
        let manifest = serde_json::to_vec(&document).unwrap();
        fs::write(root.join("manifest.json"), &manifest).unwrap();
        Self {
            _directory: directory,
            root,
            manifest,
            bytes,
        }
    }

    fn check(&self, library: &'static str, tool: &'static str, observed: Option<&str>) {
        let inspector = Inspector {
            root: self.root.clone(),
            library,
            tool,
            calls: RefCell::new(Vec::new()),
        };
        let report = runtime_package::run(&[self.root.to_string_lossy().into_owned()], &inspector);
        if let Some(observed) = observed {
            assert_eq!(report.code, 1, "{}", report.stderr);
            assert!(report.stdout.is_empty());
            assert!(
                report
                    .stderr
                    .contains("does not match packaged ELF requirement"),
                "{}",
                report.stderr
            );
            assert!(
                report.stderr.contains(&format!("requirement '{observed}'")),
                "{}",
                report.stderr
            );
            // Both declared roles participate in the observed maximum before refusal.
        } else {
            assert_eq!(report.code, 0, "{}", report.stderr);
            assert!(report.stderr.is_empty());
            assert_eq!(
                report.stdout,
                format!(
                    "verified native runtime artifact: {}\n",
                    self.root.display()
                )
            );
        }
        let calls = inspector.calls.borrow();
        for relative in [LIBRARY, TOOL] {
            assert_eq!(
                calls
                    .iter()
                    .filter(|(flag, path)| flag == "-V" && path == relative)
                    .count(),
                2,
                "host policy and package collector must inspect {relative}"
            );
            assert!(
                calls
                    .iter()
                    .any(|(flag, path)| flag == "-d" && path == relative)
            );
        }
        assert_eq!(
            fs::read(self.root.join("manifest.json")).unwrap(),
            self.manifest
        );
        for relative in [LIBRARY, TOOL] {
            assert_eq!(fs::read(self.root.join(relative)).unwrap(), self.bytes);
        }
    }
}

#[test]
fn migration_native_collector_accepts_optional_and_matching_observed_floor() {
    Fixture::new(None).check(NORMAL, NORMAL, None);
    // A higher tool requirement and a higher library requirement each determine the floor.
    for (library, tool) in [(NORMAL, NORMAL), (OLDER, NORMAL), (NORMAL, OLDER)] {
        Fixture::new(Some("2.35")).check(library, tool, None);
    }
}

#[test]
fn migration_native_collector_refuses_misleading_declared_floor() {
    for declared in ["2.34", "2.36"] {
        for (library, tool) in [(NORMAL, NORMAL), (OLDER, NORMAL), (NORMAL, OLDER)] {
            Fixture::new(Some(declared)).check(library, tool, Some("2.35"));
        }
    }
}

#[test]
fn migration_native_collector_relr_sets_floor_for_library_and_tool() {
    for (library, tool) in [(RELR, NORMAL), (NORMAL, RELR)] {
        Fixture::new(Some("2.36")).check(library, tool, None);
        Fixture::new(Some("2.35")).check(library, tool, Some("2.36"));
    }
}
