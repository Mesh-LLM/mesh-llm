//! `collect` of `scripts/linux-native-runtime-deps.py`: copies reviewed
//! CUDA redistributables from search directories into the library dir until
//! the package is closed, refusing stubs, other architectures, conflicting
//! providers and unreviewed libraries.

use super::linux_deps_elf::{
    ElfImage, Raised, architecture_matches, casefold, check_arch, elf_image, iter_files, join,
    sha256,
};
use super::linux_deps_policy::{Gaps, Package, details};
use crate::ci_plan::catalog::os_error_text;
use std::collections::HashMap;
use std::path::Path;

/// The reviewed CUDA 12.9 / 13.1 redistributable library stems.
const CUDA_REDISTRIBUTABLES: &[&str] = &["libcudart", "libcublas", "libcublasLt", "libnvJitLink"];

impl Package<'_> {
    /// `collect_dependencies`: the copied destinations, in copy order.
    pub(super) fn collect(
        &self,
        search_dirs: &[String],
        cuda_major: &str,
    ) -> Result<Vec<String>, Raised> {
        let paths = iter_files(search_dirs);
        let (stubs, real): (Vec<String>, Vec<String>) =
            paths.into_iter().partition(|path| is_stub(path));
        let search = self.candidates(&real)?;
        let stub_index = self.candidates(&stubs)?;
        let mut copied: Vec<String> = Vec::new();
        loop {
            let gaps = self.gaps()?;
            if gaps.is_empty() {
                return Ok(copied);
            }
            let before = copied.len();
            let mut unresolved = Gaps::new();
            for (importer, dependencies) in &gaps {
                for dependency in dependencies {
                    let Some(candidates) = search.get(dependency) else {
                        if let Some(stub) = stub_index.get(dependency) {
                            return Err(format!(
                                "CUDA stub library cannot satisfy runtime dependency {dependency}: {}",
                                stub[0].path
                            ));
                        }
                        unresolved
                            .entry(importer.clone())
                            .or_default()
                            .insert(dependency.clone());
                        continue;
                    };
                    validate_cuda_redistributable(dependency, cuda_major)?;
                    let source = self.select_provider(dependency, candidates)?;
                    let destination = self.copy_dependency(source, dependency)?;
                    if !copied.contains(&destination) {
                        copied.push(destination);
                    }
                }
            }
            if !unresolved.is_empty() {
                return Err(format!(
                    "unresolved Linux runtime ELF dependencies: {}",
                    details(&unresolved)
                ));
            }
            if copied.len() == before {
                return Err(format!(
                    "Linux runtime dependency collection made no progress: {}",
                    details(&gaps)
                ));
            }
        }
    }

    /// `_candidate_index`.
    fn candidates(&self, paths: &[String]) -> Result<HashMap<String, Vec<ElfImage>>, Raised> {
        let mut index: HashMap<String, Vec<ElfImage>> = HashMap::new();
        for path in paths {
            let Some(image) = elf_image(self.tools, path)? else {
                continue;
            };
            let aliases: Vec<String> = image.aliases().into_iter().map(str::to_owned).collect();
            for alias in aliases {
                index.entry(alias).or_default().push(clone_image(&image));
            }
        }
        Ok(index)
    }

    /// `_select_provider`.
    fn select_provider<'i>(
        &self,
        dependency: &str,
        candidates: &'i [ElfImage],
    ) -> Result<&'i ElfImage, Raised> {
        let matching: Vec<&ElfImage> = candidates
            .iter()
            .filter(|image| architecture_matches(image, self.arch()))
            .collect();
        if matching.is_empty() {
            let available: Vec<String> = candidates
                .iter()
                .map(|image| format!("{} ({}/{})", image.path, image.elf_class, image.machine))
                .collect();
            return Err(format!(
                "wrong architecture for search dependency {dependency}; expected {}, found: {}",
                self.arch().unwrap_or("None"),
                available.join(", ")
            ));
        }
        let mut by_digest: Vec<(String, &ElfImage)> = Vec::new();
        for image in matching {
            let digest = sha256(&image.path)?;
            if !by_digest.iter().any(|(known, _)| *known == digest) {
                by_digest.push((digest, image));
            }
        }
        if by_digest.len() > 1 {
            let paths: Vec<&str> = by_digest
                .iter()
                .map(|(_, image)| image.path.as_str())
                .collect();
            return Err(format!(
                "conflicting ELF libraries provide {dependency}: {}",
                paths.join(", ")
            ));
        }
        Ok(by_digest[0].1)
    }

    /// `_copy_dependency`: `shutil.copy2` keeps permissions and times.
    fn copy_dependency(&self, source: &ElfImage, dependency: &str) -> Result<String, Raised> {
        check_arch(source, self.arch(), "search dependency")?;
        let destination = join(&self.lib_dir, dependency);
        let target = Path::new(&destination);
        if target.is_symlink() {
            return Err(format!(
                "packaged dependency destination must not be a symlink: {destination}"
            ));
        }
        if target.exists() {
            if sha256(&destination)? != sha256(&source.path)? {
                return Err(format!(
                    "conflicting packaged dependency {dependency}: {destination} and {}",
                    source.path
                ));
            }
            return Ok(destination);
        }
        copy2(&source.path, &destination)?;
        Ok(destination)
    }
}

pub(super) fn copy2(source: &str, destination: &str) -> Result<(), Raised> {
    let failure = |error: std::io::Error, path: &str| os_error_text(&error, path);
    std::fs::copy(source, destination).map_err(|error| failure(error, source))?;
    let modified = std::fs::metadata(source)
        .and_then(|meta| meta.modified())
        .map_err(|error| failure(error, source))?;
    std::fs::File::options()
        .write(true)
        .open(destination)
        .and_then(|file| file.set_modified(modified))
        .map_err(|error| failure(error, destination))
}

fn clone_image(image: &ElfImage) -> ElfImage {
    ElfImage {
        path: image.path.clone(),
        needed: image.needed.clone(),
        soname: image.soname.clone(),
        elf_class: image.elf_class.clone(),
        machine: image.machine.clone(),
    }
}

/// `_is_stub`.
fn is_stub(path: &str) -> bool {
    path.split('/').any(|part| casefold(part) == "stubs")
}

/// `_validate_cuda_redistributable`.
fn validate_cuda_redistributable(dependency: &str, cuda_major: &str) -> Result<(), Raised> {
    let allowed = CUDA_REDISTRIBUTABLES.iter().any(|stem| {
        dependency
            .strip_prefix(stem)
            .and_then(|rest| rest.strip_prefix(".so."))
            .and_then(|rest| rest.strip_prefix(cuda_major))
            .is_some_and(|rest| {
                rest.is_empty()
                    || rest
                        .strip_prefix('.')
                        .is_some_and(|tail| !tail.is_empty() && !tail.contains('\n'))
            })
    });
    if allowed {
        return Ok(());
    }
    Err(format!(
        "Linux CUDA runtime dependency is not in the reviewed redistributable allowlist for CUDA {cuda_major}: {dependency}"
    ))
}
