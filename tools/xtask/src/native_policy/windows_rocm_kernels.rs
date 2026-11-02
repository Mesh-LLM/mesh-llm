//! Advertised Windows ROCm targets require checksum-bound rocBLAS code objects.
use crate::command::DynResult;
use std::{collections::BTreeMap, ffi::OsStr, fs, io::Write, path::Path};

const LIBRARY: &str = "lib/rocblas/library";

pub(super) fn admit(
    os: &str,
    root: &Path,
    files: &mut BTreeMap<String, String>,
    targets: Vec<String>,
) -> DynResult<Vec<String>> {
    if os != "windows"
        || targets.is_empty()
        || !files
            .keys()
            .any(|path| path.eq_ignore_ascii_case("lib/rocblas.dll"))
    {
        return Ok(targets);
    }
    let paths = code_objects(root)?;
    // Use the same contained regular-file validation and hashing as the DLLs.
    // The manifest consumers then verify the objects that justify admission.
    let objects = super::files::hashes(root, paths.iter())?;
    let (kept, dropped): (Vec<_>, Vec<_>) = targets.into_iter().partition(|target| {
        objects.keys().any(|path| {
            path.rsplit('/').next().is_some_and(|name| {
                name.match_indices(target.as_str()).any(|(start, _)| {
                    let end = start + target.len();
                    let bytes = name.as_bytes();
                    (start == 0 || !bytes[start - 1].is_ascii_alphanumeric())
                        && (end == bytes.len() || !bytes[end].is_ascii_alphanumeric())
                })
            })
        })
    });
    if kept.is_empty() {
        return Err(format!(
            "no rocBLAS kernels for any ROCm target ({}) in {LIBRARY}; the runtime would match no GPU",
            dropped.join(" ")
        )
        .into());
    }
    if !dropped.is_empty() {
        writeln!(
            crate::cli_output::stderr(),
            "not advertising {}: no packaged rocBLAS code objects in {LIBRARY}",
            dropped.join(" ")
        )?;
    }
    files.extend(objects);
    Ok(kept)
}

fn code_objects(root: &Path) -> DynResult<Vec<String>> {
    let entries = fs::read_dir(root.join(LIBRARY)).map_err(|error| {
        format!("cannot inspect packaged rocBLAS kernels in {LIBRARY}: {error}")
    })?;
    let mut paths = Vec::new();
    for entry in entries {
        if let Some(path) = code_object_path(&entry?.file_name())? {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn code_object_path(name: &OsStr) -> DynResult<Option<String>> {
    let name = name
        .to_str()
        .ok_or("packaged rocBLAS kernel filename is not UTF-8")?;
    Ok(Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("co") || extension.eq_ignore_ascii_case("hsaco")
        })
        .then(|| format!("{LIBRARY}/{name}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_policy::runtime_package_manifest::Package;
    use serde_json::json;

    #[cfg(unix)]
    #[test]
    fn code_object_inventory_refuses_non_utf8_names_without_lossy_conversion() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};
        // Some filesystems (including APFS) cannot store invalid UTF-8 names.
        // Exercise admission directly rather than depending on that capability.
        let name = OsString::from_vec(b"invalid-\xff.co".to_vec());
        assert!(
            code_object_path(&name)
                .unwrap_err()
                .to_string()
                .contains("filename is not UTF-8")
        );
    }

    #[test]
    fn package_consumer_verifies_the_code_objects_used_for_admission() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("runtime-fixture");
        fs::create_dir_all(root.join(LIBRARY)).unwrap();
        for path in [
            "lib/llama.dll",
            "lib/rocblas.dll",
            "lib/rocblas/library/kernel-gfx1100.co",
        ] {
            fs::write(root.join(path), b"abc").unwrap();
        }
        let libraries = vec!["lib/llama.dll".into(), "lib/rocblas.dll".into()];
        let mut files = super::super::files::hashes(&root, libraries.iter()).unwrap();
        let targets = admit("windows", &root, &mut files, vec!["gfx1100".into()]).unwrap();
        let manifest = json!({
            "schema_version":2,
            "runtime":{
                "id":"runtime-fixture", "release_version":"0.80.0", "skippy_abi":"1.2.3",
                "platform":{"os":"windows","arch":"x86_64","target":"x86_64-pc-windows-msvc"},
                "backend":{"kind":"rocm","rocm":{"gpu_arches":targets}},
                "libraries":libraries, "files":files
            },
            "build":{"primary_library":"lib/llama.dll","library_sha256":files["lib/llama.dll"]}
        });
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let package = Package::read(&root).unwrap();
        assert_eq!(package.libraries, libraries);
        fs::write(root.join(LIBRARY).join("kernel-gfx1100.co"), b"tampered").unwrap();
        let error = Package::read(&root)
            .err()
            .expect("changed kernel must fail");
        assert!(
            error.contains("file checksum mismatch for lib/rocblas/library/kernel-gfx1100.co"),
            "{error}"
        );
    }
}
