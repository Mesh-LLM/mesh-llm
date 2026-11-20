//! Every DLL and executable in a Windows runtime package must be built for
//! the package's own architecture. The manifest fields and the import table
//! can agree while the binaries are for another machine: an ARM64 package
//! staged from an x64 build (`LLAMA_STAGE_BUILD_DIR` pointing at x64 output)
//! resolves all its imports, and an ARM64 host then fails to load it. The PE
//! header's machine field says which machine a binary is for.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_FILE_MACHINE_ARM64: u16 = 0xaa64;

/// Check the PE machine of every `.dll` and `.exe` under `lib/` and `tools/`
/// in the package at `root`, and of every one its manifest lists in
/// `declared`, whose manifest declares `arch` and `target`.
pub(super) fn verify<'a>(
    root: &Path,
    arch: &str,
    target: &str,
    declared: impl IntoIterator<Item = &'a str>,
) -> Result<(), String> {
    let expected = match arch {
        "x86_64" => IMAGE_FILE_MACHINE_AMD64,
        "aarch64" => IMAGE_FILE_MACHINE_ARM64,
        other => return Err(format!("no Windows PE machine for runtime arch {other}")),
    };
    // The file read for each binary, under the name the package gives it.
    let mut binaries = BTreeMap::<String, PathBuf>::new();
    let mut walked = Vec::new();
    for directory in ["lib", "tools"] {
        collect_binaries(root, &root.join(directory), &mut walked)?;
    }
    for relative in walked {
        let path = root.join(&relative);
        binaries.insert(relative, path);
    }
    // The manifest may list a binary outside `lib/` and `tools/`, or one that
    // is a symlink, which the walk does not follow: each is read through the
    // file it resolves to, which `Package::read` keeps inside the package.
    for relative in declared {
        if is_binary_name(Path::new(relative)) {
            let resolved = root
                .join(relative)
                .canonicalize()
                .map_err(|error| format!("{relative}: {error}"))?;
            binaries.insert(relative.to_owned(), resolved);
        }
    }
    let mut foreign = Vec::new();
    for (relative, path) in &binaries {
        let bytes = fs::read(path).map_err(|error| format!("{relative}: {error}"))?;
        let machine = pe_machine(&bytes).map_err(|error| format!("{relative}: {error}"))?;
        if machine != expected {
            foreign.push(format!("{relative} ({})", machine_name(machine)));
        }
    }
    if foreign.is_empty() {
        return Ok(());
    }
    Err(format!(
        "Windows runtime package for {target} contains binaries built for another machine: {}",
        foreign.join(", ")
    ))
}

/// The `.dll` and `.exe` files under `directory`, as forward-slash paths
/// relative to `root`. A missing directory holds none.
fn collect_binaries(root: &Path, directory: &Path, found: &mut Vec<String>) -> Result<(), String> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("{}: {error}", directory.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_dir() {
            collect_binaries(root, &path, found)?;
            continue;
        }
        if kind.is_file() && is_binary_name(&path) {
            let relative = path.strip_prefix(root).map_err(|error| error.to_string())?;
            found.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn is_binary_name(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("dll") || extension.eq_ignore_ascii_case("exe")
        })
}

/// The COFF machine field of a PE image: `MZ`, then the `PE\0\0` signature
/// at the offset stored at 0x3c, then the machine.
fn pe_machine(bytes: &[u8]) -> Result<u16, String> {
    if bytes.get(..2) != Some(b"MZ") {
        return Err("not a PE image (no MZ header)".to_owned());
    }
    let offset = bytes
        .get(0x3c..0x40)
        .map(|field| u32::from_le_bytes([field[0], field[1], field[2], field[3]]) as usize)
        .ok_or("truncated PE image")?;
    let header = bytes
        .get(offset..offset.saturating_add(6))
        .ok_or("truncated PE image")?;
    if &header[..4] != b"PE\0\0" {
        return Err("not a PE image (no PE signature)".to_owned());
    }
    Ok(u16::from_le_bytes([header[4], header[5]]))
}

fn machine_name(machine: u16) -> String {
    match machine {
        IMAGE_FILE_MACHINE_AMD64 => "x64".to_owned(),
        IMAGE_FILE_MACHINE_ARM64 => "ARM64".to_owned(),
        0x014c => "x86".to_owned(),
        0xa641 => "ARM64EC".to_owned(),
        other => format!("machine 0x{other:04x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest bytes `pe_machine` reads: an MZ stub whose 0x3c field
    /// points at a PE signature followed by `machine`.
    fn pe(machine: u16) -> Vec<u8> {
        let mut bytes = vec![0_u8; 0x46];
        bytes[..2].copy_from_slice(b"MZ");
        bytes[0x3c..0x40].copy_from_slice(&0x40_u32.to_le_bytes());
        bytes[0x40..0x44].copy_from_slice(b"PE\0\0");
        bytes[0x44..0x46].copy_from_slice(&machine.to_le_bytes());
        bytes
    }

    fn package(files: &[(&str, Vec<u8>)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for (relative, bytes) in files {
            let path = root.path().join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        root
    }

    #[test]
    fn an_arm64_package_with_x64_dlls_staged_into_it_is_refused() {
        let root = package(&[
            ("lib/llama.dll", pe(IMAGE_FILE_MACHINE_AMD64)),
            ("lib/ggml.dll", pe(IMAGE_FILE_MACHINE_ARM64)),
            ("lib/ggml-cpu.dll", pe(IMAGE_FILE_MACHINE_AMD64)),
            ("tools/llama-server.exe", pe(IMAGE_FILE_MACHINE_ARM64)),
        ]);
        let error = verify(root.path(), "aarch64", "aarch64-pc-windows-msvc", []).unwrap_err();
        assert!(
            error.contains("lib/ggml-cpu.dll (x64), lib/llama.dll (x64)"),
            "{error}"
        );
        assert!(!error.contains("lib/ggml.dll"), "{error}");
    }

    #[test]
    fn a_package_built_for_its_own_machine_passes() {
        let arm64 = package(&[
            ("lib/llama.dll", pe(IMAGE_FILE_MACHINE_ARM64)),
            ("lib/nested/kernel.dll", pe(IMAGE_FILE_MACHINE_ARM64)),
            ("lib/notes.txt", b"not a binary".to_vec()),
        ]);
        verify(arm64.path(), "aarch64", "aarch64-pc-windows-msvc", []).unwrap();
        let x64 = package(&[("lib/llama.dll", pe(IMAGE_FILE_MACHINE_AMD64))]);
        verify(x64.path(), "x86_64", "x86_64-pc-windows-msvc", []).unwrap();
    }

    #[test]
    fn an_x64_package_names_an_arm64ec_or_x86_binary() {
        let root = package(&[("lib/a.dll", pe(0xa641)), ("tools/b.exe", pe(0x014c))]);
        let error = verify(root.path(), "x86_64", "x86_64-pc-windows-msvc", []).unwrap_err();
        assert!(
            error.contains("lib/a.dll (ARM64EC), tools/b.exe (x86)"),
            "{error}"
        );
    }

    #[test]
    fn a_file_named_like_a_dll_that_is_not_a_pe_image_is_an_error() {
        let root = package(&[("lib/llama.dll", b"abc".to_vec())]);
        let error = verify(root.path(), "x86_64", "x86_64-pc-windows-msvc", []).unwrap_err();
        assert!(error.contains("lib/llama.dll: not a PE image"), "{error}");
    }

    #[test]
    fn a_declared_library_outside_lib_is_checked_too() {
        let root = package(&[
            ("lib/ggml.dll", pe(IMAGE_FILE_MACHINE_ARM64)),
            ("payload/llama.dll", pe(IMAGE_FILE_MACHINE_AMD64)),
        ]);
        verify(root.path(), "aarch64", "aarch64-pc-windows-msvc", []).unwrap();
        let error = verify(
            root.path(),
            "aarch64",
            "aarch64-pc-windows-msvc",
            ["payload/llama.dll", "lib/ggml.dll"],
        )
        .unwrap_err();
        assert!(error.contains("payload/llama.dll (x64)"), "{error}");
        assert!(!error.contains("lib/ggml.dll"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn a_declared_symlink_is_checked_through_its_target() {
        let root = package(&[("payload/llama-x64.dll", pe(IMAGE_FILE_MACHINE_AMD64))]);
        fs::create_dir_all(root.path().join("lib")).unwrap();
        std::os::unix::fs::symlink(
            "../payload/llama-x64.dll",
            root.path().join("lib/llama.dll"),
        )
        .unwrap();
        let error = verify(
            root.path(),
            "aarch64",
            "aarch64-pc-windows-msvc",
            ["lib/llama.dll"],
        )
        .unwrap_err();
        assert!(error.contains("lib/llama.dll (x64)"), "{error}");
    }
}
