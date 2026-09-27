//! The DLL-closure policy of `scripts/windows-native-runtime-deps.py`:
//! `HOST_DLLS`, `dependency_gaps`, `collect_dependencies` and
//! `verify_dependencies`. Directory scans reproduce `pathlib` (`is_dir`/
//! `is_file` raise every `stat` error but the ignored errnos), and paths are
//! kept as the legacy `str(pathlib.Path)` text.

use super::linux_deps_collect::copy2;
use super::linux_deps_elf::{casefold, file_name, join};
use super::toolchain::Toolchain;
use super::windows_deps_pe::{Raised, imported_dlls};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Loaded from the host: OS DLLs, the MSVC/UCRT runtime, and the GPU driver
/// entry points (`nvcuda.dll`, `vulkan-1.dll`). Toolkit DLLs such as cudart
/// and cublas are deliberately absent, so they must be packaged.
const HOST_DLLS: &[&str] = &[
    "advapi32.dll",
    "bcrypt.dll",
    "cfgmgr32.dll",
    "comdlg32.dll",
    "crypt32.dll",
    "d3d12.dll",
    "dbghelp.dll",
    "dxgi.dll",
    "gdi32.dll",
    "kernel32.dll",
    "msvcp140.dll",
    "msvcrt.dll",
    "nvcuda.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "psapi.dll",
    "rpcrt4.dll",
    "secur32.dll",
    "setupapi.dll",
    "shell32.dll",
    "shlwapi.dll",
    "ucrtbase.dll",
    "user32.dll",
    "version.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vulkan-1.dll",
    "winmm.dll",
    "ws2_32.dll",
];

/// Importer name to the imported DLL names it lacks, both sorted.
pub(super) type Gaps = BTreeMap<String, BTreeSet<String>>;

/// `ELOOP`, which `pathlib` ignores alongside `ENOENT`, `ENOTDIR`, `EBADF`.
const ELOOP: i32 = if cfg!(target_os = "linux") { 40 } else { 62 };

/// `is_host_dll`.
fn is_host_dll(name: &str) -> bool {
    let normalized = casefold(name);
    HOST_DLLS.contains(&normalized.as_str())
        || normalized.starts_with("api-ms-win-")
        || normalized.starts_with("ext-ms-win-")
}

/// `Path.is_dir()` / `Path.is_file()`: `false` for an ignored `stat` error,
/// the error text for any other.
fn stat_is(path: &str, directory: bool) -> Result<bool, Raised> {
    match std::fs::metadata(path) {
        Ok(meta) => Ok(if directory {
            meta.is_dir()
        } else {
            meta.is_file()
        }),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) || matches!(error.raw_os_error(), Some(9) | Some(ELOOP)) =>
        {
            Ok(false)
        }
        Err(error) => Err(os_error_text(&error, path)),
    }
}

/// `Path.iterdir()`: child paths in directory order.
fn iterdir(directory: &str) -> Result<Vec<String>, std::io::Error> {
    std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| join(directory, &entry.file_name().to_string_lossy())))
        .collect()
}

/// `Path.suffix.casefold() == ".dll"`.
fn has_dll_suffix(path: &str) -> bool {
    let name = file_name(path);
    match name.rfind('.') {
        Some(dot) if dot > 0 && dot + 1 < name.len() => casefold(&name[dot..]) == ".dll",
        _ => false,
    }
}

/// The `.dll` regular files among `paths`, keyed by casefolded name; later
/// spellings replace earlier ones as a dict comprehension does.
fn dll_files(paths: Vec<String>, into: &mut HashMap<String, String>) -> Result<(), Raised> {
    for path in paths {
        if stat_is(&path, false)? && has_dll_suffix(&path) {
            into.insert(casefold(file_name(&path)), path);
        }
    }
    Ok(())
}

/// A runtime library directory and the directories whose DLLs it must close.
pub(super) struct Runtime {
    pub(super) lib_dir: String,
    pub(super) scan_dirs: Vec<String>,
}

impl Runtime {
    /// `dependency_gaps`.
    pub(super) fn gaps(&self) -> Result<Gaps, Raised> {
        let mut packaged = HashMap::new();
        let listing =
            iterdir(&self.lib_dir).map_err(|error| os_error_text(&error, &self.lib_dir))?;
        dll_files(listing, &mut packaged)?;
        let mut libraries = HashMap::new();
        for scan_dir in &self.scan_dirs {
            if stat_is(scan_dir, true)? {
                let listing = iterdir(scan_dir).map_err(|error| os_error_text(&error, scan_dir))?;
                dll_files(listing, &mut libraries)?;
            }
        }
        let mut gaps = Gaps::new();
        for (name, library) in libraries.into_iter().collect::<BTreeMap<_, _>>() {
            let missing: BTreeSet<String> = imported_dlls(&library)?
                .into_iter()
                .filter(|dependency| {
                    !is_host_dll(dependency) && !packaged.contains_key(&casefold(dependency))
                })
                .collect();
            if !missing.is_empty() {
                gaps.insert(name, missing);
            }
        }
        Ok(gaps)
    }

    /// `verify_dependencies`.
    pub(super) fn verify(&self) -> Result<(), Raised> {
        let gaps = self.gaps()?;
        if gaps.is_empty() {
            return Ok(());
        }
        Err(format!(
            "unpackaged Windows runtime DLL dependencies: {}",
            details(&gaps)
        ))
    }

    /// `collect_dependencies`: the copied destinations, in copy order.
    pub(super) fn collect(
        &self,
        search_dirs: &[String],
        tools: &dyn Toolchain,
    ) -> Result<Vec<String>, Raised> {
        let mut directories = vec![self.lib_dir.clone()];
        directories.extend(search_dirs.iter().cloned());
        directories.extend(default_search_dirs(tools));
        let index = dll_index(&directories)?;
        let mut copied = Vec::new();
        loop {
            let gaps = self.gaps()?;
            if gaps.is_empty() {
                return Ok(copied);
            }
            let mut unresolved = Gaps::new();
            for (importer, dependencies) in gaps {
                for dependency in dependencies {
                    let Some(source) = index.get(&casefold(&dependency)) else {
                        unresolved
                            .entry(importer.clone())
                            .or_default()
                            .insert(dependency);
                        continue;
                    };
                    let destination = join(&self.lib_dir, file_name(source));
                    if !std::path::Path::new(&destination).exists() {
                        copy2(source, &destination)?;
                        copied.push(destination);
                    }
                }
            }
            if !unresolved.is_empty() {
                return Err(format!(
                    "unresolved Windows runtime DLL dependencies: {}",
                    details(&unresolved)
                ));
            }
        }
    }
}

/// `"; ".join(f"{importer}: {', '.join(sorted(dependencies))}")`.
fn details(gaps: &Gaps) -> String {
    gaps.iter()
        .map(|(importer, dependencies)| {
            let names: Vec<&str> = dependencies.iter().map(String::as_str).collect();
            format!("{importer}: {}", names.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// `default_search_dirs`: the compiler's directory, every `PATH` entry, and
/// the Vulkan SDK binaries.
fn default_search_dirs(tools: &dyn Toolchain) -> Vec<String> {
    let mut candidates = Vec::new();
    if let Some(compiler) = ["g++", "gcc"]
        .iter()
        .find_map(|compiler| tools.locate(compiler))
    {
        let parent = compiler
            .parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        candidates.push(python_path_display(&parent));
    }
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(
            std::env::split_paths(&path)
                .filter(|entry| !entry.as_os_str().is_empty())
                .map(|entry| python_path_display(&entry)),
        );
    }
    if let Some(sdk) = std::env::var_os("VULKAN_SDK").filter(|sdk| !sdk.is_empty()) {
        let sdk = python_path_display(std::path::Path::new(&sdk));
        candidates.push(join(&sdk, "Bin"));
        candidates.push(join(&sdk, "Bin32"));
    }
    candidates
}

/// `_dll_index`: the first `.dll` of each casefolded name across
/// `directories`; unlistable directories are skipped.
fn dll_index(directories: &[String]) -> Result<HashMap<String, String>, Raised> {
    let mut index: HashMap<String, String> = HashMap::new();
    for directory in directories {
        if !stat_is(directory, true)? {
            continue;
        }
        let Ok(entries) = iterdir(directory) else {
            continue;
        };
        for entry in entries {
            if stat_is(&entry, false)? && has_dll_suffix(&entry) {
                index.entry(casefold(file_name(&entry))).or_insert(entry);
            }
        }
    }
    Ok(index)
}
