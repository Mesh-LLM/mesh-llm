use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

#[derive(Deserialize)]
struct Product {
    schema_version: u32,
    contract: String,
    mesh_version: String,
    backend: String,
    host: Entry,
    runtime: Entry,
}

#[derive(Deserialize)]
struct Entry {
    path: String,
}

fn contained(root: &Path, raw: &str, directory: bool) -> DynResult<PathBuf> {
    let path = Path::new(raw);
    if raw.is_empty()
        || raw.contains(['\\', '\0', '\r', '\n', '\t', ':'])
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err("product path must be a portable relative path".into());
    }
    let candidate = root.join(path);
    let resolved = candidate.canonicalize()?;
    if !resolved.starts_with(root)
        || fs::symlink_metadata(&candidate)?.file_type().is_symlink()
        || resolved.is_dir() != directory
        || (!directory && !resolved.is_file())
    {
        return Err("product path is missing, linked, or outside the product".into());
    }
    Ok(resolved)
}

fn inspect(root: &Path, binary: &str, expected: &str) -> DynResult<String> {
    let root = root.canonicalize()?;
    let product: Product = serde_json::from_slice(&fs::read(root.join("product-manifest.json"))?)?;
    if product.schema_version != 2
        || product.contract != "mesh-llm-product-v2"
        || product.mesh_version.is_empty()
        || product.mesh_version.contains(['\t', '\n', '\r'])
        || !matches!(
            product.backend.as_str(),
            "cpu" | "metal" | "cuda" | "cuda-blackwell" | "rocm" | "hip" | "vulkan"
        )
        || (!expected.is_empty() && product.backend != expected)
        || product.host.path != binary
    {
        return Err("product manifest identity mismatch".into());
    }
    contained(&root, &product.host.path, false)?;
    contained(&root, "host-imports.json", false)?;
    let runtime = contained(&root, &product.runtime.path, true)?;
    let parts: Vec<_> = Path::new(&product.runtime.path).components().collect();
    if parts.len() != 2 || parts[0] != Component::Normal("native-runtimes".as_ref()) {
        return Err("runtime must be a direct child of native-runtimes".into());
    }
    let actual = fs::read_dir(&root)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let expected = [
        binary,
        "host-imports.json",
        "native-runtimes",
        "product-manifest.json",
    ]
    .into_iter()
    .map(std::ffi::OsString::from)
    .collect::<BTreeSet<_>>();
    if actual != expected
        || fs::read_dir(&root)?.any(|entry| entry.is_ok_and(|entry| entry.path().is_symlink()))
    {
        return Err("product top-level contents are not canonical".into());
    }
    let runtimes = fs::read_dir(root.join("native-runtimes"))?.collect::<Result<Vec<_>, _>>()?;
    if runtimes.len() != 1 || runtimes[0].path().canonicalize()? != runtime {
        return Err("product must contain only its selected runtime".into());
    }
    Ok(format!(
        "{}\t{}\t{}\t{}\n",
        product.mesh_version, product.backend, product.host.path, product.runtime.path
    ))
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [root, binary, backend] = args else {
        return Err(
            "usage: automation smoke-inputs <product-root> <binary-name> <expected-backend>".into(),
        );
    };
    crate::cli_output::stdout().write_all(inspect(Path::new(root), binary, backend)?.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_product_when_exact_contents() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join("native-runtimes/fixture"))?;
        fs::write(root.path().join("mesh-llm"), "host")?;
        fs::write(root.path().join("host-imports.json"), "{}")?;
        fs::write(
            root.path().join("product-manifest.json"),
            r#"{"schema_version":2,"contract":"mesh-llm-product-v2","mesh_version":"1.0.0","backend":"cpu","host":{"path":"mesh-llm"},"runtime":{"path":"native-runtimes/fixture"}}"#,
        )?;
        assert_eq!(
            inspect(root.path(), "mesh-llm", "cpu")?,
            "1.0.0\tcpu\tmesh-llm\tnative-runtimes/fixture\n"
        );
        assert!(inspect(root.path(), "mesh-llm", "metal").is_err());
        fs::write(root.path().join("extra"), "unexpected")?;
        assert!(inspect(root.path(), "mesh-llm", "cpu").is_err());
        Ok(())
    }
    #[test]
    fn paths_reject_escape_and_controls() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        for raw in ["../escape", "/escape", "C:/escape", "a\\b", "a\tb"] {
            assert!(contained(root.path(), raw, false).is_err());
        }
        Ok(())
    }
}
