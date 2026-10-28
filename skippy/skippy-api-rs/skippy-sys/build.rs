//! Builds skippy_api (core + llama backend) with CMake, compiles the cxx
//! facade against it, and tells Cargo what to link.
//!
//! Inputs, all optional:
//!   SKIPPY_API_DIR            skippy_api checkout; default ../../skippy_api
//!   SKIPPY_LLAMA_SOURCE_DIR   llama.cpp checkout; default: the nearest .deps/llama.cpp
//!                             in a directory above this crate (mesh-llm keeps it at the repo root)
//!   SKIPPY_CMAKE_PROFILE      CMake build type; default Release

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    let skippy_api = dir_from_env("SKIPPY_API_DIR", || manifest.join("../../skippy_api"));
    let llama = dir_from_env("SKIPPY_LLAMA_SOURCE_DIR", || {
        manifest
            .ancestors()
            .map(|ancestor| ancestor.join(".deps/llama.cpp"))
            .find(|candidate| candidate.join("include/llama.h").is_file())
            .unwrap_or_else(|| manifest.join("../../../.deps/llama.cpp"))
    });
    let profile = env::var("SKIPPY_CMAKE_PROFILE").unwrap_or_else(|_| "Release".into());

    for name in ["SKIPPY_API_DIR", "SKIPPY_LLAMA_SOURCE_DIR", "SKIPPY_CMAKE_PROFILE"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for path in [
        manifest.join("cpp/bridge.h"),
        manifest.join("cpp/bridge.cpp"),
        manifest.join("src/lib.rs"),
        skippy_api.join("CMakeLists.txt"),
        skippy_api.join("include"),
        skippy_api.join("src"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let build_root = cmake::Config::new(&skippy_api)
        .profile(&profile)
        .define("SKIPPY_BUILD_LLAMA", "ON")
        .define("SKIPPY_LLAMA_SOURCE_DIR", &llama)
        .define("SKIPPY_API_BUILD_TESTS", "OFF")
        .define("SKIPPY_API_BUILD_BENCHMARKS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .build_target("skippy_llama")
        .build()
        .join("build");

    let libraries = find_libraries(&build_root);
    link(&libraries, "skippy_llama", Kind::Static);
    link(&libraries, "skippy_core", Kind::Static);
    link(&libraries, "llama", Kind::Discovered);
    libraries.keys().filter(|name| name.starts_with("ggml")).for_each(|name| link(&libraries, name, Kind::Discovered));

    // Dependent packages cannot inherit cargo:rustc-link-arg; publish the build
    // root through the links-key metadata (Cargo.toml `links = "skippy_api"`),
    // so a dependent's build.rs can read DEP_SKIPPY_API_ROOT and emit its own
    // -Wl,-rpath for the shared llama/ggml libraries (i386 review).
    println!("cargo:root={}", build_root.display());
    // Shared llama/ggml libraries stay in the build tree; binaries find them by rpath.
    let mut rpaths: Vec<&Path> =
        libraries.values().filter(|library| library.shared).map(|library| library.dir.as_path()).collect();
    rpaths.sort();
    rpaths.dedup();
    for dir in rpaths {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }

    cxx_build::bridge("src/lib.rs")
        .file("cpp/bridge.cpp")
        .include(skippy_api.join("include"))
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .std("c++2b")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wpedantic")
        .compile("skippy_sys_bridge");
}

fn dir_from_env(name: &str, default: impl FnOnce() -> PathBuf) -> PathBuf {
    let path = env::var_os(name).map_or_else(default, PathBuf::from);
    path.canonicalize()
        .unwrap_or_else(|error| panic!("{name}: {} is not a readable directory: {error}", path.display()))
}

struct Library {
    dir: PathBuf,
    shared: bool,
}

#[derive(Clone, Copy)]
enum Kind {
    Static,
    Discovered,
}

/// Every lib<name>.{a,so,dylib} under `root`, by name. A shared build wins
/// over a static one of the same name.
fn find_libraries(root: &Path) -> BTreeMap<String, Library> {
    let mut found = BTreeMap::new();
    collect_libraries(root, &mut found);
    assert!(
        found.contains_key("skippy_llama") && found.contains_key("llama"),
        "CMake build under {} produced no skippy_llama/llama libraries",
        root.display()
    );
    found
}

fn collect_libraries(dir: &Path, found: &mut BTreeMap<String, Library>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_libraries(&path, found);
            continue;
        }
        let Some((name, shared)) = library_name(&path) else { continue };
        if found.get(&name).is_none_or(|existing| shared && !existing.shared) {
            found.insert(name, Library { dir: dir.to_path_buf(), shared });
        }
    }
}

fn library_name(path: &Path) -> Option<(String, bool)> {
    let file = path.file_name()?.to_str()?;
    let stem = file.strip_prefix("lib")?;
    if let Some(name) = stem.strip_suffix(".a") {
        return Some((name.to_owned(), false));
    }
    [".so", ".dylib"].iter().find_map(|suffix| stem.strip_suffix(suffix)).map(|name| (name.to_owned(), true))
}

fn link(libraries: &BTreeMap<String, Library>, name: &str, kind: Kind) {
    let library = libraries.get(name).unwrap_or_else(|| panic!("library {name} not produced by the CMake build"));
    println!("cargo:rustc-link-search=native={}", library.dir.display());
    let modifier = if matches!(kind, Kind::Discovered) && library.shared { "dylib" } else { "static" };
    println!("cargo:rustc-link-lib={modifier}={name}");
}
