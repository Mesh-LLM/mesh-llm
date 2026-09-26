use std::{
    collections::BTreeSet,
    env,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

fn frame(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn bundled_recipe(crate_dir: &std::path::Path) -> (String, String) {
    let roster_path = crate_dir.join("src/inference/skippy/split-certified.json");
    println!("cargo:rerun-if-changed={}", roster_path.display());
    let roster: serde_json::Value = serde_json::from_slice(
        &fs::read(&roster_path).expect("read bundled split certification roster"),
    )
    .expect("parse bundled split certification roster");
    let recipe = roster
        .get("native_recipe")
        .and_then(serde_json::Value::as_object)
        .expect("split certification roster native_recipe");
    let field = |name: &str| {
        recipe
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("split certification roster native_recipe.{name}"))
            .to_string()
    };
    (field("llama_upstream_sha"), field("patch_queue_sha256"))
}

fn series_patches(patch_dir: &Path, subdir: &str) -> Vec<(String, PathBuf)> {
    // Keep this series validation and ordering contract in lockstep with
    // scripts/generate-split-certified.py::_series_patches. Together they
    // define the v2 patch_queue_sha256 consumed below.
    let directory = patch_dir.join(subdir);
    if !directory.exists() {
        return Vec::new();
    }

    let series_path = directory.join("series");
    println!("cargo:rerun-if-changed={}", series_path.display());
    let names = fs::read_to_string(&series_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", series_path.display()))
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert!(
        !names.is_empty() && names.iter().all(|name| !name.is_empty()),
        "patch series is empty or contains blank entries: {}",
        series_path.display(),
    );
    let listed = names.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        listed.len(),
        names.len(),
        "patch series contains duplicate entries: {}",
        series_path.display(),
    );
    assert!(
        names.iter().all(|name| Path::new(name)
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .is_some_and(|file_name| file_name == name)),
        "patch series contains an unsafe entry: {}",
        series_path.display(),
    );

    let actual = fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        .map(|entry| entry.expect("read patch entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "patch")
        })
        .map(|path| {
            path.file_name()
                .expect("patch file name")
                .to_string_lossy()
                .into_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual,
        listed,
        "patch series does not exactly cover its directory: {}",
        series_path.display(),
    );

    names
        .into_iter()
        .map(|name| (format!("{subdir}/{name}"), directory.join(name)))
        .collect()
}

fn main() {
    let crate_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("crate directory"));
    let root = crate_dir.join("../..");
    let upstream_path = root.join("third_party/llama.cpp/upstream.txt");
    let patch_dir = root.join("third_party/llama.cpp/patches");
    if !upstream_path.is_file() && !patch_dir.is_dir() {
        // crates.io packages cannot contain files outside this crate. The
        // repository CI validates the source recipe before packaging; a
        // packaged crate retains the exact recipe embedded in its roster.
        let (upstream, patch_digest) = bundled_recipe(&crate_dir);
        println!("cargo:rustc-env=MESH_LLAMA_UPSTREAM_SHA={upstream}");
        println!("cargo:rustc-env=MESH_SKIPPY_PATCH_QUEUE_SHA256={patch_digest}");
        return;
    }
    assert!(
        upstream_path.is_file() && patch_dir.is_dir(),
        "llama.cpp recipe is incomplete: both the upstream pin and patch directory are required",
    );
    println!("cargo:rerun-if-changed={}", upstream_path.display());
    println!("cargo:rerun-if-changed={}", patch_dir.display());

    let upstream = fs::read_to_string(&upstream_path)
        .expect("read llama.cpp upstream pin")
        .trim()
        .to_string();
    let mut patches = fs::read_dir(&patch_dir)
        .expect("read llama.cpp patch directory")
        .map(|entry| entry.expect("read patch entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "patch")
        })
        .map(|path| {
            let name = path
                .file_name()
                .expect("patch file name")
                .to_string_lossy()
                .into_owned();
            (name, path)
        })
        .collect::<Vec<_>>();
    patches.sort_by(|left, right| left.0.cmp(&right.0));
    patches.extend(series_patches(&patch_dir, "model_support"));
    patches.extend(series_patches(&patch_dir, "generated"));
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-llm-skippy-patch-queue-v2\0");
    hasher.update((patches.len() as u64).to_le_bytes());
    for (relative_path, path) in patches {
        println!("cargo:rerun-if-changed={}", path.display());
        frame(&mut hasher, relative_path.as_bytes());
        frame(&mut hasher, &fs::read(&path).expect("read llama.cpp patch"));
    }
    println!("cargo:rustc-env=MESH_LLAMA_UPSTREAM_SHA={upstream}");
    let mut patch_digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut patch_digest, "{byte:02x}").expect("write patch digest");
    }
    println!("cargo:rustc-env=MESH_SKIPPY_PATCH_QUEUE_SHA256={patch_digest}");
}
