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
        "Command '('objdump', '-p', 'm.exe')' returned non-zero exit status 3.\n"
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
