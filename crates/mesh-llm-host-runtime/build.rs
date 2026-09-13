use std::{env, fmt::Write, fs, path::PathBuf};

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
        .collect::<Vec<_>>();
    patches.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"mesh-llm-skippy-patch-queue-v1\0");
    hasher.update((patches.len() as u64).to_le_bytes());
    for path in patches {
        println!("cargo:rerun-if-changed={}", path.display());
        frame(
            &mut hasher,
            path.file_name()
                .expect("patch file name")
                .to_string_lossy()
                .as_bytes(),
        );
        frame(&mut hasher, &fs::read(&path).expect("read llama.cpp patch"));
    }
    println!("cargo:rustc-env=MESH_LLAMA_UPSTREAM_SHA={upstream}");
    let mut patch_digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut patch_digest, "{byte:02x}").expect("write patch digest");
    }
    println!("cargo:rustc-env=MESH_SKIPPY_PATCH_QUEUE_SHA256={patch_digest}");
}
