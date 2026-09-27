//! The packaged-runtime dependency policy of
//! `scripts/linux-native-runtime-deps.py`: the package index, host-owned
//! library rules, dependency gaps and `verify`.

use super::linux_deps_elf::{
    ElfImage, Raised, casefold, check_arch, elf_image, iter_files, parent, sha256,
};
use super::toolchain::Toolchain;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Libraries the Linux host (or the NVIDIA driver) owns.
const HOST_LIBRARY_NAMES: &[&str] = &[
    "ld-linux.so.2",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
    "linux-vdso.so.1",
    "libc.so.6",
    "libcrypt.so.1",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libatomic.so.1",
    "libgomp.so.1",
    "libm.so.6",
    "libnsl.so.1",
    "libnuma.so.1",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libstdc++.so.6",
    "libutil.so.1",
    "libz.so.1",
    "libcuda.so",
    "libcuda.so.1",
    "libnvidia-ml.so.1",
];

/// Importer name to its unpackaged dependencies.
pub(super) type Gaps = BTreeMap<String, BTreeSet<String>>;

/// The inputs shared by every subcommand.
pub(super) struct Package<'a> {
    pub(super) tools: &'a dyn Toolchain,
    pub(super) lib_dir: String,
    /// `[lib_dir, *scan_dir]`.
    pub(super) scan_dirs: Vec<String>,
    pub(super) arch: Option<String>,
}

fn host_owned(name: &str) -> bool {
    let lower = casefold(name);
    HOST_LIBRARY_NAMES.contains(&lower.as_str())
        || lower.starts_with("linux-vdso")
        || lower.starts_with("libnvidia-")
}

impl Package<'_> {
    pub(super) fn arch(&self) -> Option<&str> {
        self.arch.as_deref()
    }

    /// `_package_index`: every packaged ELF image, architecture-checked and
    /// free of conflicting aliases.
    pub(super) fn images(&self) -> Result<Vec<ElfImage>, Raised> {
        let mut directories = vec![self.lib_dir.clone()];
        directories.extend(self.scan_dirs.iter().cloned());
        let mut images = Vec::new();
        let mut aliases: HashMap<String, (String, String)> = HashMap::new();
        for path in iter_files(&directories) {
            let Some(image) = elf_image(self.tools, &path)? else {
                continue;
            };
            check_arch(&image, self.arch(), "packaged runtime")?;
            let digest = sha256(&path)?;
            for alias in image.aliases() {
                match aliases.get(alias) {
                    Some((previous, known)) if *known != digest => {
                        return Err(format!(
                            "conflicting ELF libraries provide {alias}: {previous} and {path}"
                        ));
                    }
                    Some(_) => {}
                    None => {
                        aliases.insert(alias.to_owned(), (path.clone(), digest.clone()));
                    }
                }
            }
            images.push(image);
        }
        Ok(images)
    }

    pub(super) fn in_lib_dir(&self, image: &ElfImage) -> bool {
        parent(&image.path) == self.lib_dir
    }

    /// `dependency_gaps`.
    pub(super) fn gaps(&self) -> Result<Gaps, Raised> {
        let images = self.images()?;
        let mut packaged_host_owned: Vec<&str> = images
            .iter()
            .filter(|image| self.in_lib_dir(image) && host_owned(image.name()))
            .map(ElfImage::name)
            .collect();
        packaged_host_owned.sort_unstable();
        if !packaged_host_owned.is_empty() {
            return Err(format!(
                "host-owned Linux libraries must not be packaged: {}",
                packaged_host_owned.join(", ")
            ));
        }
        let packaged: BTreeSet<&str> = images.iter().map(ElfImage::name).collect();
        let mut gaps = Gaps::new();
        for image in &images {
            let missing: BTreeSet<String> = image
                .needed
                .iter()
                .filter(|name| !host_owned(name) && !packaged.contains(name.as_str()))
                .cloned()
                .collect();
            if !missing.is_empty() {
                gaps.insert(image.name().to_owned(), missing);
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
            "unpackaged Linux runtime ELF dependencies: {}",
            details(&gaps)
        ))
    }
}

/// `"; ".join(f"{importer}: {', '.join(sorted(deps))}" ...)`.
pub(super) fn details(gaps: &Gaps) -> String {
    gaps.iter()
        .map(|(importer, dependencies)| {
            let names: Vec<&str> = dependencies.iter().map(String::as_str).collect();
            format!("{importer}: {}", names.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}
