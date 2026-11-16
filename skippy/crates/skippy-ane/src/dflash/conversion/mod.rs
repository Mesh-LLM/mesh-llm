//! Offline Rust conversion of the paired Qwen3.6 DFlash GGUF to Core ML.

mod gguf;
mod graph;
#[cfg(target_os = "macos")]
mod placement;
mod weights;

use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mil_spec::write_mlpackage_stream;
use serde_json::json;
use sha2::{Digest, Sha256};

use self::gguf::{Pair, TAPS};
use self::weights::Weights;

/// Inputs and output paths for the paired Qwen3.6 DFlash Core ML conversion.
pub struct ConversionOptions {
    pub draft: PathBuf,
    pub target: PathBuf,
    pub out: PathBuf,
    pub embeddings_out: PathBuf,
    pub context: usize,
    pub rows: usize,
}

/// Convert a paired draft and target GGUF into an ANE package and compiled model.
///
/// The compiled model, embeddings and manifest are installed beside `out`.
/// Existing output paths are rejected.
pub fn convert(args: &ConversionOptions) -> Result<()> {
    if args.out.extension().is_none_or(|ext| ext != "mlpackage") {
        bail!("--out must end in .mlpackage");
    }
    if args.context < args.rows || args.rows != 7 {
        bail!("the paired DFlash serving graph requires seven rows and context >= rows");
    }
    let compiled = args.out.with_extension("mlmodelc");
    let metadata = args.out.with_extension("json");
    validate_output_destinations(&[&args.out, &compiled, &metadata, &args.embeddings_out])?;
    let parent = output_parent(&args.out);
    let staging =
        tempfile::tempdir_in(parent).context("create ANE conversion staging directory")?;
    let package = staging.path().join("draft.mlpackage");
    let embeddings = staging.path().join("embeddings.f16");
    let pair = Pair::open(&args.draft, &args.target)?;
    let mask_embedding = pair.target_embeddings(&embeddings)?;
    let weight_path = package.join("Data/com.apple.CoreML/weights/weight.bin");
    std::fs::create_dir_all(weight_path.parent().expect("weight path has parent"))?;
    let mut weights = Weights::create(&pair, &weight_path)?;
    let spec = graph::build(&mut weights, &mask_embedding, args.context, args.rows)?;
    weights.finish()?;
    write_mlpackage_stream(&package, &spec, Some(&weight_path))?;
    let compiled_temp = compile_package(&package)?;
    #[cfg(target_os = "macos")]
    let placement = placement::inspect(&compiled_temp)?;
    let manifest = json!({
        "draft_sha256": sha256_file(&args.draft)?,
        "target_sha256": sha256_file(&args.target)?,
        "embeddings_sha256": sha256_file(&embeddings)?,
        "context": args.context,
        "rows": args.rows,
        "body_only": false,
        "int8": true,
        "token_output": true,
        "taps": TAPS,
        "compiled_model": compiled.file_name().context("compiled model has no name")?.to_string_lossy(),
    });
    #[cfg(target_os = "macos")]
    let manifest = {
        let mut manifest = manifest;
        let fields = manifest.as_object_mut().expect("manifest is an object");
        fields.insert(
            "neural_operations".into(),
            json!(placement.neural_operations),
        );
        fields.insert(
            "neural_operations_on_ane".into(),
            json!(placement.neural_operations_on_ane),
        );
        manifest
    };
    let staged_metadata = staging.path().join("draft.json");
    std::fs::write(&staged_metadata, serde_json::to_vec_pretty(&manifest)?)?;
    install_artifacts(&[
        (&package, &args.out),
        (&compiled_temp, &compiled),
        (&embeddings, &args.embeddings_out),
        (&staged_metadata, &metadata),
    ])?;
    Ok(())
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn validate_output_destinations(paths: &[&Path]) -> Result<()> {
    let mut destinations = HashSet::new();
    for path in paths {
        let parent = output_parent(path);
        let canonical_parent = std::fs::canonicalize(parent)
            .with_context(|| format!("output directory does not exist: {}", parent.display()))?;
        if !canonical_parent.is_dir() {
            bail!("output parent is not a directory: {}", parent.display());
        }
        let name = path
            .file_name()
            .with_context(|| format!("output path has no file name: {}", path.display()))?;
        if !destinations.insert(canonical_parent.join(name)) {
            bail!(
                "output destinations refer to the same path: {}",
                path.display()
            );
        }
        match std::fs::symlink_metadata(path) {
            Ok(_) => bail!("output already exists: {}", path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
        }
    }
    Ok(())
}

fn install_artifacts(artifacts: &[(&Path, &Path)]) -> Result<()> {
    let mut installed = Vec::new();
    for &(source, destination) in artifacts {
        if let Err(error) = install_artifact(source, destination) {
            let mut cleanup_errors = Vec::new();
            for path in installed.into_iter().rev() {
                if let Err(cleanup_error) = remove_installed_output(path) {
                    cleanup_errors.push(cleanup_error.to_string());
                }
            }
            if !cleanup_errors.is_empty() {
                return Err(error.context(format!("cleanup failed: {}", cleanup_errors.join("; "))));
            }
            return Err(error);
        }
        installed.push(destination);
    }
    Ok(())
}

fn install_artifact(source: &Path, destination: &Path) -> Result<()> {
    match std::fs::symlink_metadata(destination) {
        Ok(_) => bail!("output already exists: {}", destination.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspect {}", destination.display()));
        }
    }
    std::fs::rename(source, destination)
        .with_context(|| format!("install {}", destination.display()))
}

fn remove_installed_output(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect installed output {}", path.display()))?;
    if metadata.is_dir() {
        std::fs::remove_dir_all(path)
            .with_context(|| format!("remove incomplete output {}", path.display()))
    } else {
        std::fs::remove_file(path)
            .with_context(|| format!("remove incomplete output {}", path.display()))
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut stream = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

#[cfg(target_os = "macos")]
fn compile_package(path: &Path) -> Result<PathBuf> {
    use std::sync::mpsc;

    use anyhow::anyhow;
    use block2::RcBlock;
    use objc2_core_ml::MLModel;
    use objc2_foundation::{NSError, NSString, NSURL};

    let source = path.to_str().context("ANE package path is not UTF-8")?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(source));
    let (tx, rx) = mpsc::channel::<Result<PathBuf>>();
    let callback = RcBlock::new(move |compiled: *mut NSURL, error: *mut NSError| {
        let result = unsafe { compiled.as_ref() }
            .and_then(NSURL::path)
            .map(|path| PathBuf::from(path.to_string()))
            .ok_or_else(|| anyhow!("Core ML compilation failed: {error:p}"));
        let _ = tx.send(result);
    });
    unsafe { MLModel::compileModelAtURL_completionHandler(&url, &callback) };
    rx.recv()
        .context("Core ML compiler did not return a result")?
}

#[cfg(not(target_os = "macos"))]
fn compile_package(_path: &Path) -> Result<PathBuf> {
    bail!("Core ML compilation requires macOS");
}

#[cfg(test)]
mod tests {
    use super::{install_artifacts, validate_output_destinations};

    #[test]
    fn output_aliases_are_rejected_before_conversion() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("alias")).unwrap();
        let package = temp.path().join("draft.mlpackage");
        let metadata = temp.path().join("draft.json");
        let alias = temp.path().join("alias/../draft.json");
        let error = validate_output_destinations(&[&package, &metadata, &alias]).unwrap_err();
        assert!(error.to_string().contains("same path"));
    }

    #[test]
    fn missing_output_parent_is_rejected_before_conversion() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("missing/embeddings.f16");
        let error = validate_output_destinations(&[&output]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("output directory does not exist")
        );
    }

    #[test]
    fn failed_later_install_removes_only_outputs_this_attempt_installed() {
        let temp = tempfile::tempdir().unwrap();
        let package = temp.path().join("staged.mlpackage");
        std::fs::create_dir(&package).unwrap();
        let installed = temp.path().join("draft.mlpackage");
        let compiled = temp.path().join("staged.mlmodelc");
        std::fs::create_dir(&compiled).unwrap();
        let installed_compiled = temp.path().join("draft.mlmodelc");
        let existing = temp.path().join("embeddings.f16");
        std::fs::write(&existing, b"existing").unwrap();
        let error = install_artifacts(&[
            (&package, &installed),
            (&compiled, &installed_compiled),
            (&temp.path().join("missing"), &existing),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("output already exists"));
        assert!(!installed.exists());
        assert!(!installed_compiled.exists());
        assert_eq!(std::fs::read(existing).unwrap(), b"existing");
    }
}
