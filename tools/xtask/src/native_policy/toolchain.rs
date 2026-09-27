//! The native inspection tools `verify-host-dependencies.py` shells out to
//! (`readelf`, `otool`, `llvm-readobj`, `objdump`), behind [`Toolchain`] so
//! the policy runs against a fake in unit tests. [`run_tool`] reproduces
//! `run_tool`: a `shutil.which` probe, then `subprocess.check_output` with
//! stderr merged into stdout, text decoding, and `LC_ALL=C`.

use crate::ci_operations::python_json_decode::{DecodeError, Hooks, loads};
use crate::ci_plan::catalog::os_error_text;
use crate::ci_plan::document::Json;
use crate::repository::python_text::repr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// How a finished tool ended.
pub(super) enum Exit {
    Code(i32),
    Signal(i32),
}

/// The merged output and exit of one tool run.
pub(super) struct Captured {
    pub(super) output: Vec<u8>,
    pub(super) exit: Exit,
}

/// The separate streams and exit of one tool run.
pub(super) struct Split {
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
    pub(super) exit: Exit,
}

/// The inspection-tool adapter.
pub(super) trait Toolchain {
    /// `shutil.which(program) is not None`.
    fn which(&self, program: &str) -> bool;
    /// Runs `argv` with stderr merged into stdout.
    fn capture(&self, argv: &[String]) -> std::io::Result<Captured>;
    /// Runs `argv` keeping stdout and stderr apart, as
    /// `subprocess.run(capture_output=True)`. Adapters without separate
    /// streams report everything as stdout.
    fn output(&self, argv: &[String]) -> std::io::Result<Split> {
        let captured = self.capture(argv)?;
        Ok(Split {
            stdout: captured.output,
            stderr: Vec::new(),
            exit: captured.exit,
        })
    }
    fn capture_without_ld_library_path(&self, argv: &[String]) -> std::io::Result<Captured> {
        self.capture(argv)
    }
    /// `shutil.which(program)`: where the program would be found, if
    /// anywhere. Adapters without a filesystem locate nothing.
    fn locate(&self, _program: &str) -> Option<PathBuf> {
        None
    }
}

/// The tools on the process `PATH`.
pub(super) struct HostToolchain;

impl Toolchain for HostToolchain {
    fn which(&self, program: &str) -> bool {
        self.locate(program).is_some()
    }

    fn locate(&self, program: &str) -> Option<PathBuf> {
        which(program, std::env::var_os("PATH").as_deref())
    }

    fn capture(&self, argv: &[String]) -> std::io::Result<Captured> {
        let (mut reader, writer) = std::io::pipe()?;
        let mut command = tool_command(argv);
        command.stdout(writer.try_clone()?).stderr(writer);
        let mut child = command.spawn()?;
        drop(command);
        let mut output = Vec::new();
        reader.read_to_end(&mut output)?;
        let status = child.wait()?;
        Ok(Captured {
            output,
            exit: exit_of(status),
        })
    }

    fn output(&self, argv: &[String]) -> std::io::Result<Split> {
        let output = tool_command(argv).output()?;
        Ok(Split {
            stdout: output.stdout,
            stderr: output.stderr,
            exit: exit_of(output.status),
        })
    }

    fn capture_without_ld_library_path(&self, argv: &[String]) -> std::io::Result<Captured> {
        let (mut reader, writer) = std::io::pipe()?;
        let mut command = tool_command(argv);
        command.env_remove("LD_LIBRARY_PATH");
        command.stdout(writer.try_clone()?).stderr(writer);
        let mut child = command.spawn()?;
        drop(command);
        let mut output = Vec::new();
        reader.read_to_end(&mut output)?;
        let status = child.wait()?;
        Ok(Captured {
            output,
            exit: exit_of(status),
        })
    }
}

/// `argv` in the C locale with the caller's stdin.
fn tool_command(argv: &[String]) -> Command {
    let program = HostToolchain
        .locate(&argv[0])
        .unwrap_or_else(|| PathBuf::from(&argv[0]));
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .env("LC_ALL", "C")
        .stdin(Stdio::inherit());
    command
}

#[cfg(unix)]
fn exit_of(status: std::process::ExitStatus) -> Exit {
    use std::os::unix::process::ExitStatusExt;
    match (status.code(), status.signal()) {
        (Some(code), _) => Exit::Code(code),
        (None, Some(signal)) => Exit::Signal(signal),
        (None, None) => Exit::Code(1),
    }
}

#[cfg(not(unix))]
fn exit_of(status: std::process::ExitStatus) -> Exit {
    Exit::Code(status.code().unwrap_or(1))
}

#[cfg(unix)]
fn which(program: &str, path: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    if program.contains('/') {
        return executable(Path::new(program)).then(|| PathBuf::from(program));
    }
    let search = path.map_or_else(|| "/bin:/usr/bin".into(), |value| value.to_owned());
    std::env::split_paths(&search)
        .map(|directory| directory.join(program))
        .find(|candidate| executable(candidate))
}

#[cfg(windows)]
fn which(program: &str, path: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let path_value = path.unwrap_or_default();
    let suffixes = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    if program.contains(['/', '\\']) {
        let requested = Path::new(program);
        return windows_which(
            program,
            requested.parent().unwrap_or(Path::new(".")),
            &suffixes,
        );
    }
    std::env::split_paths(path_value)
        .find_map(|directory| windows_which(program, &directory, &suffixes))
}

#[cfg(any(test, windows))]
fn windows_which(program: &str, directory: &Path, pathext: &str) -> Option<PathBuf> {
    let name = Path::new(program).file_name()?.to_str()?;
    let direct = directory.join(name);
    if direct.is_file() {
        return Some(direct);
    }
    if Path::new(name).extension().is_some() {
        return None;
    }
    let extensions = pathext.split(';').filter(|suffix| suffix.starts_with('.'));
    for extension in extensions {
        let candidate = directory.join(format!("{name}{}", extension.to_ascii_lowercase()));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(unix)]
fn executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

/// `run_tool(command)`: the decoded output, or `str(error)` of the
/// exception the legacy script caught.
pub(super) fn run_tool(tools: &dyn Toolchain, argv: &[String]) -> Result<String, String> {
    if !tools.which(&argv[0]) {
        return Err(format!(
            "{} is required to inspect host dependencies",
            argv[0]
        ));
    }
    let captured = tools
        .capture(argv)
        .map_err(|error| os_error_text(&error, &argv[0]))?;
    match captured.exit {
        Exit::Code(0) => decode(&captured.output),
        Exit::Code(code) => Err(format!(
            "Command '{}' returned non-zero exit status {code}.",
            tuple(argv)
        )),
        Exit::Signal(signal) => Err(format!(
            "Command '{}' died with {}.",
            tuple(argv),
            signal_name(signal)
        )),
    }
}

/// Python `repr(tuple_of_str)`.
fn tuple(argv: &[String]) -> String {
    let items: Vec<String> = argv.iter().map(|arg| repr(arg)).collect();
    format!("({})", items.join(", "))
}

/// `repr(signal.Signals(n))` for the signals every POSIX platform numbers
/// alike; others fall back to CPython's `unknown signal n`.
fn signal_name(signal: i32) -> String {
    let name = match signal {
        1 => "SIGHUP",
        2 => "SIGINT",
        3 => "SIGQUIT",
        4 => "SIGILL",
        5 => "SIGTRAP",
        6 => "SIGABRT",
        8 => "SIGFPE",
        9 => "SIGKILL",
        11 => "SIGSEGV",
        13 => "SIGPIPE",
        14 => "SIGALRM",
        15 => "SIGTERM",
        _ => return format!("unknown signal {signal}"),
    };
    format!("<Signals.{name}: {signal}>")
}

fn keep(pairs: Vec<(String, Json)>) -> Result<Json, String> {
    Ok(Json::Object(pairs))
}

fn constant(_: &str) -> Result<Json, String> {
    Ok(Json::Null)
}

/// `text=True`: strict UTF-8 with universal newlines.
pub(super) fn decode(output: &[u8]) -> Result<String, String> {
    match std::str::from_utf8(output) {
        Ok(text) => Ok(text.replace("\r\n", "\n").replace('\r', "\n")),
        Err(_) => {
            let hooks = Hooks {
                pairs: keep,
                constant,
            };
            // The shared decoder words an invalid byte as CPython's codec does.
            match loads(output, &hooks) {
                Err(DecodeError::Value(message)) => Err(message),
                _ => Err("'utf-8' codec can't decode output".to_owned()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_native_policy_tool_failures_use_python_wording() {
        let argv = ["otool".to_owned(), "-L".to_owned(), "a'b".to_owned()];
        assert_eq!(tuple(&argv), "('otool', '-L', \"a'b\")");
        assert_eq!(signal_name(9), "<Signals.SIGKILL: 9>");
        assert_eq!(signal_name(64), "unknown signal 64");
        assert_eq!(decode(b"a\r\nb\rc").ok().as_deref(), Some("a\nb\nc"));
        assert_eq!(
            decode(b"ab\xff").err().as_deref(),
            Some("'utf-8' codec can't decode byte 0xff in position 2: invalid start byte")
        );
    }

    #[test]
    fn migration_native_policy_windows_inspector_and_compiler_exe_lookup() {
        let root = std::env::temp_dir().join(format!(
            "xtask-native-win-lookup-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).expect("scratch directory");
        for name in ["llvm-readobj.exe", "objdump.exe", "g++.exe", "gcc.exe"] {
            std::fs::write(root.join(name), []).expect("stub executable");
        }
        for name in ["llvm-readobj", "objdump", "g++", "gcc"] {
            assert_eq!(
                windows_which(name, &root, ".EXE;.CMD"),
                Some(root.join(format!("{name}.exe")))
            );
        }
        assert_eq!(windows_which("llvm-readobj", &root, ".CMD"), None);
        assert_eq!(
            windows_which("llvm-readobj.exe", &root, ".CMD"),
            Some(root.join("llvm-readobj.exe"))
        );
        assert_eq!(windows_which("llvm-readobj.com", &root, ".EXE"), None);
        std::fs::remove_dir_all(root).expect("scratch cleanup");
    }
}
