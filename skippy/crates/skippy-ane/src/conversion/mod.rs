//! Offline Rust conversion of the paired Qwen3.6 DFlash GGUF to Core ML.

mod gguf;
mod graph;
#[cfg(target_os = "macos")]
mod placement;
mod weights;

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
    for path in [&args.out, &compiled, &metadata, &args.embeddings_out] {
        if path.exists() {
            bail!("output already exists: {}", path.display());
        }
    }
    let parent = args.out.parent().unwrap_or_else(|| Path::new("."));
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
    std::fs::rename(&package, &args.out).context("install ANE model package")?;
    std::fs::rename(&compiled_temp, &compiled).context("install compiled ANE model")?;
    std::fs::rename(&embeddings, &args.embeddings_out).context("install ANE embeddings")?;
    std::fs::write(&metadata, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
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
