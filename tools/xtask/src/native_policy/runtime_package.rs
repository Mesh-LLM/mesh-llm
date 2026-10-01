use super::runtime_package_manifest::Package;
use super::toolchain::Toolchain;
use crate::artifact::{self, ArtifactCommand};
use crate::repository::check_report::CheckReport;
use std::fs;
use std::path::{Path, PathBuf};

const USAGE: &str = "Usage: scripts/verify-native-runtime-package.sh [--portable] <artifact-dir-or-tar.gz> [...]\n\nVerifies MeshLLM native runtime artifacts:\n  - manifest schema and resolver fields\n  - artifact directory name matches runtime.id\n  - all runtime.libraries exist\n  - library_sha256 matches the primary library\n  - Linux platform.min_glibc is a valid major.minor floor and matches the\n    packaged ELF requirement exactly when present\n  - Linux ELF libraries and tools stay within the declared glibc floor\n  - Linux shared-library RUNPATH/RPATH is relocatable and resolves packaged deps\n  - Linux CUDA ELF dependencies are closed, same-architecture, and non-stub\n  - Windows non-system DLL imports are present in the artifact\n  - required archive checksum sidecar\n  - archive paths and links cannot escape the extraction directory\n\n--portable validates integrity, archive shape, manifest schema, paths, and\nchecksums without running host-specific binary dependency probes.\n";

pub(super) fn run(args: &[String], tools: &dyn Toolchain) -> CheckReport {
    let mut portable = false;
    let mut index = 0;
    while let Some(arg) = args.get(index) {
        match arg.as_str() {
            "--portable" => portable = true,
            "--" => {
                index += 1;
                break;
            }
            word if word.starts_with('-') => {
                return CheckReport::failure(
                    String::new(),
                    format!("unknown argument: {word}\n{USAGE}"),
                );
            }
            _ => break,
        }
        index += 1;
    }
    if index == args.len() {
        return CheckReport::failure(String::new(), USAGE.to_owned());
    }
    let mut stdout = String::new();
    for (position, input) in args[index..].iter().enumerate() {
        match verify_input(input, position, portable, tools) {
            Ok(message) => stdout.push_str(&message),
            Err(error) => return CheckReport::failure(stdout, format!("{error}\n")),
        }
    }
    CheckReport::success(stdout)
}

fn verify_input(
    input: &str,
    position: usize,
    portable: bool,
    tools: &dyn Toolchain,
) -> Result<String, String> {
    let source = Path::new(input);
    if source.is_dir() {
        return verify_dir(source, portable, tools);
    }
    if !input.ends_with(".tar.gz") && !input.ends_with(".tgz") {
        return Err(format!(
            "unsupported native runtime artifact input: {input}"
        ));
    }
    let checksum = artifact::check(ArtifactCommand::VerifyChecksum, &[input.to_owned()]);
    if checksum.code != 0 {
        return Err(checksum.stderr.trim_end_matches('\n').to_owned());
    }
    let temporary = TempExtraction::new(position)?;
    let extracted = artifact::check(
        ArtifactCommand::ExtractTar,
        &[
            input.to_owned(),
            temporary.path().to_string_lossy().into_owned(),
        ],
    );
    if extracted.code != 0 {
        return Err(extracted.stderr.trim_end_matches('\n').to_owned());
    }
    let entries: Vec<PathBuf> = fs::read_dir(temporary.path())
        .map_err(|error| error.to_string())?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let directory = match entries.as_slice() {
        [directory] if directory.is_dir() && !directory.is_symlink() => directory,
        _ => {
            return Err(format!(
                "expected archive to contain one top-level artifact directory: {input}"
            ));
        }
    };
    verify_dir(directory, portable, tools)
}

fn verify_dir(path: &Path, portable: bool, tools: &dyn Toolchain) -> Result<String, String> {
    let package = Package::read(path)?;
    if !portable {
        match package.os.as_str() {
            "linux" => super::runtime_package_linux::verify(&package, tools)?,
            "macos" => super::runtime_package_macos::verify(&package, tools)?,
            "windows" => {
                let arguments = vec![
                    "verify".to_owned(),
                    "--lib-dir".to_owned(),
                    path.join("lib").to_string_lossy().into_owned(),
                    "--scan-dir".to_owned(),
                    path.join("tools").to_string_lossy().into_owned(),
                ];
                let report = super::windows_deps::run(&arguments, tools);
                if report.code != 0 {
                    return Err(report.stderr.trim_end_matches('\n').to_owned());
                }
            }
            _ => unreachable!(),
        }
    }
    let label = if portable {
        "verified portable native runtime artifact"
    } else {
        "verified native runtime artifact"
    };
    Ok(format!("{label}: {}\n", path.display()))
}

struct TempExtraction(PathBuf);

impl TempExtraction {
    fn new(position: usize) -> Result<Self, String> {
        let clock = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "xtask-native-{}-{clock}-artifact-{position}",
            std::process::id()
        ));
        fs::create_dir(&path).map_err(|error| error.to_string())?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempExtraction {
    fn drop(&mut self) {
        let _cleanup = fs::remove_dir_all(&self.0);
    }
}
