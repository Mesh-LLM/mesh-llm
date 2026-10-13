//! The host policy against a fake [`Toolchain`]: no inspection tool runs.

use super::*;
use crate::native_policy::toolchain::{Captured, Exit};
use std::cell::RefCell;

/// Canned tool output keyed by program, recording every invocation.
struct FakeToolchain {
    outputs: Vec<(&'static str, &'static str, i32)>,
    calls: RefCell<Vec<Vec<String>>>,
}

impl FakeToolchain {
    fn new(outputs: Vec<(&'static str, &'static str, i32)>) -> Self {
        Self {
            outputs,
            calls: RefCell::new(Vec::new()),
        }
    }

    fn entry(&self, program: &str) -> Option<&(&'static str, &'static str, i32)> {
        self.outputs.iter().find(|(name, ..)| *name == program)
    }
}

impl Toolchain for FakeToolchain {
    fn which(&self, program: &str) -> bool {
        self.entry(program).is_some()
    }

    fn capture(&self, argv: &[String]) -> std::io::Result<Captured> {
        self.calls.borrow_mut().push(argv.to_vec());
        let (_, output, code) = self.entry(&argv[0]).ok_or(std::io::ErrorKind::NotFound)?;
        Ok(Captured {
            output: output.as_bytes().to_vec(),
            exit: Exit::Code(*code),
        })
    }
}

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_owned()).collect()
}

fn no_floor() -> Result<PathBuf, String> {
    Err("unused".to_owned())
}

const READELF: &str = " 0x1 (NEEDED) Shared library: [libcuda.so.1]\n 0x1 (NEEDED) Shared library: [libc.so.6]\nVersion needs section '.gnu.version_r'\n  Name: GLIBC_2.40\n";

#[test]
fn migration_native_policy_rejects_backend_imports_through_the_adapter() {
    let tools = FakeToolchain::new(vec![("readelf", READELF, 0)]);
    let report = run(
        &args(&["--format", "elf", "bin/mesh-llm"]),
        &tools,
        &mut no_floor,
    );
    assert_eq!(
        report.stdout,
        "{\"binary\": \"mesh-llm\", \"format\": \"elf\", \"glibc_floor\": \"2.40\", \
         \"imports\": [\"libc.so.6\", \"libcuda.so.1\"], \"policy\": \"mesh-llm-dynamic-host-v2\", \
         \"rejected_imports\": [\"libcuda.so.1\"]}\n"
    );
    assert_eq!(
        report.stderr,
        "host dependency policy rejected: libcuda.so.1\n"
    );
    assert_eq!(report.code, 1);
    assert_eq!(
        *tools.calls.borrow(),
        [
            args(&["readelf", "-d", "bin/mesh-llm"]),
            args(&["readelf", "-V", "bin/mesh-llm"])
        ]
    );
}

#[test]
fn migration_native_policy_glibc_floor_above_maximum_fails() {
    let tools = FakeToolchain::new(vec![("readelf", READELF, 0)]);
    let words = [
        "--format",
        "elf",
        "--no-import-policy",
        "--max-glibc",
        "2.39",
        "lib.so",
    ];
    let report = run(&args(&words), &tools, &mut no_floor);
    assert_eq!(report.code, 1);
    assert!(report.stdout.contains("\"policy\": \"none\""));
    assert_eq!(
        report.stderr,
        "lib.so needs GLIBC_2.40 but the declared floor is 2.39. Raising the floor drops \
         Linux distributions that were supported before; see mesh-llm#1522.\n"
    );
}

#[test]
fn migration_native_policy_pe_prefers_llvm_readobj() {
    let tools = FakeToolchain::new(vec![
        (
            "llvm-readobj",
            "  Name: KERNEL32.dll\n  Name: nvcuda.dll\n",
            0,
        ),
        ("objdump", "", 1),
    ]);
    let report = run(&args(&["--format", "pe", "m.exe"]), &tools, &mut no_floor);
    assert_eq!(report.code, 1);
    assert_eq!(
        report.stderr,
        "host dependency policy rejected: nvcuda.dll\n"
    );
    assert_eq!(
        *tools.calls.borrow(),
        [args(&["llvm-readobj", "--coff-imports", "m.exe"])]
    );
}

#[test]
fn migration_native_policy_tool_failures_exit_two() {
    let tools = FakeToolchain::new(vec![("objdump", "", 3)]);
    let report = run(&args(&["--format", "pe", "m.exe"]), &tools, &mut no_floor);
    assert_eq!(
        report.stderr,
        "native inspection command [\"objdump\", \"-p\", \"m.exe\"] exited with status 3\n"
    );
    assert_eq!(report.code, 2);
    let missing = FakeToolchain::new(Vec::new());
    let report = run(&args(&["--format", "macho", "m"]), &missing, &mut no_floor);
    assert_eq!(
        report.stderr,
        "otool is required to inspect host dependencies\n"
    );
    assert_eq!(report.code, 2);
}

#[test]
fn bound_host_report_matches_actual_binary_bytes_and_preserves_failed_policy_report() {
    use sha2::{Digest, Sha256};
    let state = tempfile::tempdir().unwrap();
    let binary = state.path().join("skippy");
    let report_path = state.path().join("host-imports.json");
    for (bytes, imports, expected_code) in [
        (b"first actual binary bytes".as_slice(), "", 0),
        (b"second actual binary bytes".as_slice(), READELF, 1),
    ] {
        std::fs::write(&binary, bytes).unwrap();
        let tools = FakeToolchain::new(vec![("readelf", imports, 0)]);
        let result = run(
            &args(&[
                binary.to_str().unwrap(),
                "--format",
                "elf",
                "--bind-sha256",
                "--report",
                report_path.to_str().unwrap(),
            ]),
            &tools,
            &mut no_floor,
        );
        assert_eq!(result.code, expected_code, "{}", result.stderr);
        let stdout: serde_json::Value = serde_json::from_str(&result.stdout).unwrap();
        let written: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(stdout, written);
        assert_eq!(stdout["binary_sha256"], hex::encode(Sha256::digest(bytes)));
        assert_eq!(stdout["binary"], "skippy");
        if expected_code == 1 {
            assert_eq!(
                stdout["rejected_imports"],
                serde_json::json!(["libcuda.so.1"])
            );
            assert!(result.stderr.contains("host dependency policy rejected"));
        }
    }
}

#[test]
fn bound_host_refuses_missing_nonregular_empty_and_oversized_input_before_inspection() {
    let state = tempfile::tempdir().unwrap();
    let empty = state.path().join("empty");
    std::fs::write(&empty, []).unwrap();
    let oversized = state.path().join("oversized");
    std::fs::File::create(&oversized)
        .unwrap()
        .set_len(2 * 1024 * 1024 * 1024 + 1)
        .unwrap();
    for binary in [
        state.path().join("missing"),
        state.path().to_owned(),
        empty,
        oversized,
    ] {
        let tools = FakeToolchain::new(vec![("readelf", "", 0)]);
        let result = run(
            &args(&[binary.to_str().unwrap(), "--format", "elf", "--bind-sha256"]),
            &tools,
            &mut no_floor,
        );
        assert_eq!(result.code, 2);
        assert!(result.stdout.is_empty());
        assert!(tools.calls.borrow().is_empty());
    }
}

#[cfg(unix)]
#[test]
fn bound_host_refuses_leaf_symlink_and_fifo_without_blocking_or_inspection() {
    use std::{ffi::CString, os::unix::fs::symlink};
    let state = tempfile::tempdir().unwrap();
    let target = state.path().join("target");
    std::fs::write(&target, b"actual binary").unwrap();
    let link = state.path().join("link");
    symlink(&target, &link).unwrap();
    let fifo = state.path().join("fifo");
    let name = CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // This finite FIFO is owned by the temporary fixture; no writer is created.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    for binary in [link, fifo] {
        let tools = FakeToolchain::new(vec![("readelf", "", 0)]);
        let result = run(
            &args(&[binary.to_str().unwrap(), "--format", "elf", "--bind-sha256"]),
            &tools,
            &mut no_floor,
        );
        assert_eq!(result.code, 2);
        assert!(tools.calls.borrow().is_empty());
    }
}

#[test]
fn bound_host_refuses_binary_changed_by_inspection_before_publishing_report() {
    struct MutatingToolchain(PathBuf);
    impl Toolchain for MutatingToolchain {
        fn which(&self, _: &str) -> bool {
            true
        }
        fn capture(&self, _: &[String]) -> std::io::Result<Captured> {
            std::fs::write(&self.0, b"changed binary bytes")?;
            Ok(Captured {
                output: Vec::new(),
                exit: Exit::Code(0),
            })
        }
    }
    let state = tempfile::tempdir().unwrap();
    let binary = state.path().join("skippy");
    std::fs::write(&binary, b"initial binary bytes").unwrap();
    let report_path = state.path().join("report.json");
    let result = run(
        &args(&[
            binary.to_str().unwrap(),
            "--format",
            "elf",
            "--bind-sha256",
            "--report",
            report_path.to_str().unwrap(),
        ]),
        &MutatingToolchain(binary),
        &mut no_floor,
    );
    assert_eq!(result.code, 2);
    assert!(
        result
            .stderr
            .contains("changed during dependency inspection")
    );
    assert!(!report_path.exists());
}
