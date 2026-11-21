//! Admit the baseline's actual native dependency closure without executing it.
use crate::command::DynResult;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

// CUDA vendor libraries can exceed 512 MiB; keep native-image inspection bounded
// independently of the much smaller automation inputs and receipts.
const MAX_NATIVE_IMAGE_BYTES: u64 = 1024 * 1024 * 1024;

struct Imports {
    libraries: Vec<String>,
    search: Vec<String>,
    macho: bool,
}

pub(super) fn verify(executable: &Path, runtime: &Path) -> DynResult<()> {
    let root = runtime.canonicalize()?;
    let executable = admitted_executable(executable, &root)?;
    let mut pending = vec![executable.clone()];
    pending.extend(runtime_libraries(&root)?);
    let mut visited = BTreeSet::new();
    let executable_dir = executable
        .parent()
        .ok_or("baseline executable parent missing")?;
    let main = inspect(&executable)?;
    let inherited = search_paths(&main.search, executable_dir, executable_dir, &root)?;
    while let Some(path) = pending.pop() {
        if !visited.insert(path.clone()) {
            continue;
        }
        if visited.len() > 2048 {
            return Err("standard runtime native closure exceeds object bound".into());
        }
        let image = inspect(&path)?;
        let origin = path.parent().ok_or("baseline library parent missing")?;
        let mut search = search_paths(&image.search, origin, executable_dir, &root)?;
        // Mach-O uses the run-path stack. ELF shared objects must carry their
        // own pinned search paths; RUNPATH is not transitive.
        if image.macho {
            search.extend(inherited.iter().cloned());
        }
        for library in &image.libraries {
            if system_library(library, image.macho) {
                continue;
            }
            pending.push(resolve(
                library,
                &search,
                origin,
                executable_dir,
                &root,
                image.macho,
            )?);
        }
    }
    Ok(())
}

pub(super) fn admitted_executable(executable: &Path, runtime: &Path) -> DynResult<PathBuf> {
    let path = contained(executable, &runtime.canonicalize()?)?;
    if !path.is_file() {
        return Err(
            "baseline executable must be a regular file in the pinned standard runtime".into(),
        );
    }
    Ok(path)
}

fn runtime_libraries(root: &Path) -> DynResult<Vec<PathBuf>> {
    let mut libraries = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    let mut count = 0usize;
    // Backends can be opened by ggml rather than DT_NEEDED. Verify their
    // dependencies as well, before accepting the tree as an offline baseline.
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            count += 1;
            if count > 250_000 {
                return Err("standard runtime exceeds entry bound".into());
            }
            if entry.file_type()?.is_dir() {
                directories.push(entry.path());
            } else if is_library(&entry.path()) {
                libraries.push(contained(&entry.path(), root)?);
            }
        }
    }
    Ok(libraries)
}

fn is_library(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.ends_with(".dylib") || name.ends_with(".so") || name.contains(".so.")
        })
}

fn inspect(path: &Path) -> DynResult<Imports> {
    if std::fs::symlink_metadata(path)?.len() > MAX_NATIVE_IMAGE_BYTES {
        return Err("standard runtime native image exceeds inspection bound".into());
    }
    let bytes = crate::automation::receipt_files::bounded(path, MAX_NATIVE_IMAGE_BYTES)?;
    match goblin::Object::parse(&bytes)? {
        goblin::Object::Elf(image) => Ok(Imports {
            libraries: image
                .libraries
                .iter()
                .map(|value| (*value).into())
                .collect(),
            search: if image.runpaths.is_empty() {
                image.rpaths
            } else {
                image.runpaths
            }
            .iter()
            .flat_map(|paths| paths.split(':').map(str::to_owned))
            .collect(),
            macho: false,
        }),
        goblin::Object::Mach(goblin::mach::Mach::Binary(image)) => Ok(mach_imports(&image)),
        goblin::Object::Mach(goblin::mach::Mach::Fat(images)) => {
            let cpu = if std::env::consts::ARCH == "aarch64" {
                goblin::mach::constants::cputype::CPU_TYPE_ARM64
            } else {
                goblin::mach::constants::cputype::CPU_TYPE_X86_64
            };
            let image = images
                .find(|arch| arch.is_ok_and(|arch| arch.cputype == cpu))
                .ok_or("standard runtime lacks the host Mach-O architecture")??;
            match image {
                goblin::mach::SingleArch::MachO(image) => Ok(mach_imports(&image)),
                _ => Err("standard runtime Mach-O archive is not a dynamic image".into()),
            }
        }
        _ => Err("standard baseline requires ELF or Mach-O native images".into()),
    }
}

fn mach_imports(image: &goblin::mach::MachO<'_>) -> Imports {
    Imports {
        libraries: image
            .libs
            .iter()
            .filter(|value| **value != "self")
            .map(|value| (*value).into())
            .collect(),
        search: image.rpaths.iter().map(|value| (*value).into()).collect(),
        macho: true,
    }
}

fn contained(path: &Path, root: &Path) -> DynResult<PathBuf> {
    let path = path.canonicalize()?;
    if !path.starts_with(root) {
        return Err("baseline native dependency escapes the pinned standard runtime".into());
    }
    Ok(path)
}

fn expand(path: &str, origin: &Path, executable: &Path) -> DynResult<PathBuf> {
    for (token, base) in [
        ("${ORIGIN}", origin),
        ("$ORIGIN", origin),
        ("@loader_path", origin),
        ("@executable_path", executable),
    ] {
        if path == token {
            return Ok(base.into());
        }
        if let Some(relative) = path.strip_prefix(&format!("{token}/")) {
            return Ok(base.join(relative));
        }
    }
    let path = Path::new(path);
    if !path.is_absolute() {
        return Err("baseline dependency uses an unbound loader search path".into());
    }
    Ok(path.into())
}

fn search_paths(
    paths: &[String],
    origin: &Path,
    executable: &Path,
    root: &Path,
) -> DynResult<Vec<PathBuf>> {
    paths
        .iter()
        .map(|path| contained(&expand(path, origin, executable)?, root))
        .collect()
}

fn resolve(
    library: &str,
    search: &[PathBuf],
    origin: &Path,
    executable: &Path,
    root: &Path,
    macho: bool,
) -> DynResult<PathBuf> {
    let candidate = if let Some(name) = library.strip_prefix("@rpath/") {
        search
            .iter()
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
    } else if library.contains('/') {
        Some(expand(library, origin, executable)?)
    } else if macho {
        None
    } else {
        search
            .iter()
            .map(|dir| dir.join(library))
            .find(|path| path.is_file())
    }
    .ok_or("baseline native dependency is absent from its pinned loader search path")?;
    let resolved = contained(&candidate, root)?;
    if !resolved.is_file() {
        return Err("baseline native dependency is not a regular file".into());
    }
    Ok(resolved)
}

fn system_library(library: &str, macho: bool) -> bool {
    if macho {
        (library.starts_with("/usr/lib/") || library.starts_with("/System/Library/"))
            && !library.split('/').any(|part| matches!(part, "." | ".."))
    } else {
        crate::native_policy::host_owned_linux_library(library)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn elf(path: &Path, library: Option<&str>, runpath: Option<&str>) {
        use goblin::elf::{dynamic, header, program_header};
        let mut bytes = vec![0u8; 1024];
        bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        bytes[16..18].copy_from_slice(&header::ET_DYN.to_le_bytes());
        bytes[18..20].copy_from_slice(&header::EM_AARCH64.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[32..40].copy_from_slice(&64u64.to_le_bytes());
        bytes[52..54].copy_from_slice(&64u16.to_le_bytes());
        bytes[54..56].copy_from_slice(&56u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&2u16.to_le_bytes());
        bytes[64..68].copy_from_slice(&program_header::PT_LOAD.to_le_bytes());
        bytes[96..104].copy_from_slice(&1024u64.to_le_bytes());
        bytes[104..112].copy_from_slice(&1024u64.to_le_bytes());
        bytes[120..124].copy_from_slice(&program_header::PT_DYNAMIC.to_le_bytes());
        bytes[128..136].copy_from_slice(&176u64.to_le_bytes());
        bytes[136..144].copy_from_slice(&176u64.to_le_bytes());
        let mut strings = vec![0u8];
        let mut tags = vec![(dynamic::DT_STRTAB, 512u64)];
        for (tag, value) in [
            (dynamic::DT_NEEDED, library),
            (dynamic::DT_RUNPATH, runpath),
        ] {
            if let Some(value) = value {
                tags.push((tag, strings.len() as u64));
                strings.extend_from_slice(value.as_bytes());
                strings.push(0);
            }
        }
        tags.push((dynamic::DT_STRSZ, strings.len() as u64));
        tags.push((dynamic::DT_NULL, 0));
        bytes[152..160].copy_from_slice(&((tags.len() * 16) as u64).to_le_bytes());
        bytes[160..168].copy_from_slice(&((tags.len() * 16) as u64).to_le_bytes());
        for (index, (tag, value)) in tags.into_iter().enumerate() {
            let offset = 176 + index * 16;
            bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
        }
        bytes[512..512 + strings.len()].copy_from_slice(&strings);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn large_native_image_preserves_imports_and_rejects_oversized_files() {
        let dir = tempfile::tempdir().unwrap();
        let library = dir.path().join("liblarge.so");
        elf(&library, Some("libggml.so"), Some("$ORIGIN"));
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&library)
            .unwrap();
        file.set_len(512 * 1024 * 1024 + 4096).unwrap();

        let imports = inspect(&library).unwrap();
        assert_eq!(imports.libraries, ["libggml.so"]);
        assert_eq!(imports.search, ["$ORIGIN"]);
        assert!(!imports.macho);

        file.set_len(MAX_NATIVE_IMAGE_BYTES + 1).unwrap();
        let error = inspect(&library).err().unwrap();
        assert_eq!(
            error.to_string(),
            "standard runtime native image exceeds inspection bound"
        );
    }

    #[test]
    fn inspected_elf_closure_requires_its_actual_needed_library_inside_the_pin() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("llama-server");
        let library = dir.path().join("libggml.so");
        elf(&binary, Some("libggml.so"), Some("$ORIGIN"));
        elf(&library, None, None);
        let imports = inspect(&binary).unwrap();
        assert_eq!(imports.libraries, ["libggml.so"]);
        assert_eq!(imports.search, ["$ORIGIN"]);
        verify(&binary, dir.path()).unwrap();
        std::fs::remove_file(library).unwrap();
        assert!(verify(&binary, dir.path()).is_err());
        elf(&binary, Some("libggml.so"), Some("/usr/lib"));
        assert!(verify(&binary, dir.path()).is_err());
    }

    #[test]
    fn baseline_executable_and_plugin_directory_must_be_in_the_hashed_tree() {
        let pin = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let binary = external.path().join("llama-server");
        elf(&binary, None, None);
        elf(&external.path().join("libggml-cpu.so"), None, None);
        assert!(verify(&binary, pin.path()).is_err());
        let pinned_binary = pin.path().join("llama-server");
        elf(&pinned_binary, None, None);
        verify(&pinned_binary, pin.path()).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&binary, pin.path().join("external-server")).unwrap();
            assert!(verify(&pin.path().join("external-server"), pin.path()).is_err());
        }
    }

    #[test]
    fn loader_resolution_rejects_external_and_unreachable_libraries() {
        let tree = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let root = tree.path().canonicalize().unwrap();
        std::fs::write(root.join("libggml.so"), b"library").unwrap();
        std::fs::write(external.path().join("libggml.so"), b"foreign").unwrap();
        let search = search_paths(&["$ORIGIN".into()], &root, &root, &root).unwrap();
        assert_eq!(
            resolve("libggml.so", &search, &root, &root, &root, false).unwrap(),
            root.join("libggml.so")
        );
        assert!(resolve("libggml.so", &[], &root, &root, &root, false).is_err());
        assert!(
            search_paths(
                &[external.path().to_string_lossy().into()],
                &root,
                &root,
                &root
            )
            .is_err()
        );
        assert!(
            resolve(
                external.path().join("libggml.so").to_str().unwrap(),
                &search,
                &root,
                &root,
                &root,
                false
            )
            .is_err()
        );
        assert!(!system_library("libggml.so", false));
        assert!(system_library("libc.so.6", false));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(
                external.path().join("libggml.so"),
                root.join("libforeign.so"),
            )
            .unwrap();
            assert!(resolve("libforeign.so", &search, &root, &root, &root, false).is_err());
        }
    }
}
