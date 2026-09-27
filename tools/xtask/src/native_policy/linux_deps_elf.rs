//! ELF discovery for `native linux-runtime-deps`: the legacy script's
//! `_iter_files`, `_elf_image` (two `readelf` runs through the
//! [`Toolchain`] adapter), `_sha256` and architecture checks. Paths are kept
//! as the legacy `str(pathlib.Path)` text so diagnostics match byte for byte.

use super::toolchain::{Exit, Toolchain, decode};
use crate::ci_plan::catalog::os_error_text;
use crate::repository::python_text::{splitlines, strip};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

/// A failure the legacy `main` printed (or, for an uncaught exception, the
/// final traceback line) before exiting 1.
pub(super) type Raised = String;

/// One inspected ELF file.
pub(super) struct ElfImage {
    pub(super) path: String,
    pub(super) needed: Vec<String>,
    pub(super) soname: Option<String>,
    pub(super) elf_class: String,
    pub(super) machine: String,
}

impl ElfImage {
    pub(super) fn name(&self) -> &str {
        file_name(&self.path)
    }

    /// `{path.name, soname}`, name first.
    pub(super) fn aliases(&self) -> Vec<&str> {
        let mut aliases = vec![self.name()];
        if let Some(soname) = self.soname.as_deref().filter(|soname| !soname.is_empty())
            && soname != self.name()
        {
            aliases.push(soname);
        }
        aliases
    }
}

/// `Path.name`.
pub(super) fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// `str(Path.parent)`.
pub(super) fn parent(path: &str) -> &str {
    match path.rfind('/') {
        Some(0) => "/",
        Some(index) => &path[..index],
        None => ".",
    }
}

/// `str(Path(base) / name)` for a displayed base.
pub(super) fn join(base: &str, name: &str) -> String {
    match base {
        "." => name.to_owned(),
        "/" => format!("/{name}"),
        _ => format!("{base}/{name}"),
    }
}

/// `str.casefold()` for the characters that can fold onto ASCII names.
pub(super) fn casefold(text: &str) -> String {
    text.to_lowercase().replace('ſ', "s").replace('ß', "ss")
}

/// `_iter_files`: regular non-symlink files below each existing directory
/// (symlinked directories are not entered), first spelling of each resolved
/// file kept, ordered by path text.
pub(super) fn iter_files(directories: &[String]) -> Vec<String> {
    let mut files = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for directory in directories {
        if !Path::new(directory).is_dir() {
            continue;
        }
        let mut entries = Vec::new();
        walk(directory, &mut entries);
        entries.sort_by(|left, right| left.split('/').cmp(right.split('/')));
        for entry in entries {
            let Ok(resolved) = std::fs::canonicalize(&entry) else {
                continue;
            };
            if seen.insert(resolved) {
                files.push(entry);
            }
        }
    }
    files.sort();
    files
}

fn walk(directory: &str, files: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = join(directory, &entry.file_name().to_string_lossy());
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            walk(&path, files);
        } else if kind.is_file() {
            files.push(path);
        }
    }
}

/// `_is_elf`.
fn is_elf(path: &str) -> Result<bool, Raised> {
    let failure =
        |error: std::io::Error| format!("read ELF file {path}: {}", os_error_text(&error, path));
    let mut handle = std::fs::File::open(path).map_err(failure)?;
    let mut magic = Vec::with_capacity(4);
    handle
        .by_ref()
        .take(4)
        .read_to_end(&mut magic)
        .map_err(failure)?;
    Ok(magic == b"\x7fELF")
}

/// `_sha256`.
pub(super) fn sha256(path: &str) -> Result<String, Raised> {
    let mut file = std::fs::File::open(path).map_err(|error| os_error_text(&error, path))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| os_error_text(&error, path))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

/// `_readelf(*arguments, path=path)`.
fn readelf(tools: &dyn Toolchain, flag: &str, path: &str) -> Result<String, Raised> {
    let argv = ["readelf".to_owned(), flag.to_owned(), path.to_owned()];
    let split = tools.output(&argv).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => {
            "readelf is required to inspect Linux native runtime ELF files".to_owned()
        }
        _ => os_error_text(&error, "readelf"),
    })?;
    let stdout = decode(&split.stdout).map_err(unicode_error)?;
    let stderr = decode(&split.stderr).map_err(unicode_error)?;
    if matches!(split.exit, Exit::Code(0)) {
        return Ok(stdout);
    }
    let details = if stderr.is_empty() { &stdout } else { &stderr };
    Err(format!(
        "cannot inspect ELF file {path}: {}",
        strip(details)
    ))
}

fn unicode_error(message: String) -> Raised {
    format!("UnicodeDecodeError: {message}")
}

/// `_elf_image`: `None` for a non-ELF file.
pub(super) fn elf_image(tools: &dyn Toolchain, path: &str) -> Result<Option<ElfImage>, Raised> {
    if !is_elf(path)? {
        return Ok(None);
    }
    let header = readelf(tools, "-h", path)?;
    let mut elf_class = String::new();
    let mut machine = String::new();
    for line in splitlines(&header) {
        if let Some(value) = line.strip_prefix("  Class:") {
            strip(value).clone_into(&mut elf_class);
        } else if let Some(value) = line.strip_prefix("  Machine:") {
            strip(value).clone_into(&mut machine);
        }
    }
    if elf_class.is_empty() || machine.is_empty() {
        return Err(format!("ELF header is missing class or machine: {path}"));
    }
    let mut image = ElfImage {
        path: path.to_owned(),
        needed: Vec::new(),
        soname: None,
        elf_class,
        machine,
    };
    for line in splitlines(&readelf(tools, "-d", path)?) {
        match dynamic_entry(line) {
            Some(("NEEDED", value)) => image.needed.push(value.to_owned()),
            Some((_, value)) => image.soname = Some(value.to_owned()),
            None => {}
        }
    }
    Ok(Some(image))
}

/// `re.search(r"\((NEEDED|SONAME)\).*\[(.*)\]", line)`: the leftmost tag,
/// then the text between the last `[` and the last `]` that follow it.
fn dynamic_entry(line: &str) -> Option<(&'static str, &str)> {
    let (start, tag) = ["(NEEDED)", "(SONAME)"]
        .iter()
        .filter_map(|tag| line.find(tag).map(|index| (index, *tag)))
        .min()?;
    let rest = &line[start + tag.len()..];
    let close = rest.rfind(']')?;
    let open = rest[..close].rfind('[')?;
    Some((&tag[1..tag.len() - 1], &rest[open + 1..close]))
}

/// `_architecture_matches`.
pub(super) fn architecture_matches(image: &ElfImage, arch: Option<&str>) -> bool {
    let expected = match arch {
        None => return true,
        Some("x86_64") => ("ELF64", "Advanced Micro Devices X86-64"),
        Some("aarch64") => ("ELF64", "AArch64"),
        Some(_) => ("ELF32", "ARM"),
    };
    (image.elf_class.as_str(), image.machine.as_str()) == expected
}

/// `_check_arch`.
pub(super) fn check_arch(image: &ElfImage, arch: Option<&str>, label: &str) -> Result<(), Raised> {
    if architecture_matches(image, arch) {
        return Ok(());
    }
    Err(format!(
        "wrong architecture for {label}: {} is {}/{}, expected {}",
        image.path,
        image.elf_class,
        image.machine,
        arch.unwrap_or("None")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_native_policy_dynamic_entries_follow_the_legacy_regex() {
        let line = " 0x1 (NEEDED)             Shared library: [libcudart.so.12]";
        assert_eq!(dynamic_entry(line), Some(("NEEDED", "libcudart.so.12")));
        assert_eq!(dynamic_entry(" (SONAME) x [a] [b]"), Some(("SONAME", "b")));
        assert_eq!(
            dynamic_entry(" (SONAME) x [a]b] ["),
            Some(("SONAME", "a]b"))
        );
        assert_eq!(dynamic_entry(" (NEEDED) no brackets"), None);
        assert_eq!(parent("lib/a.so"), "lib");
        assert_eq!(parent("a.so"), ".");
        assert_eq!(casefold("LIBſTDC++.SO.6"), "libstdc++.so.6");
    }
}
