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
        // Native dependencies travel inside the rlib. Link arguments and rpaths
        // from this build script do not propagate to a downstream executable.
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BACKEND_DL", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .build_target("skippy_llama")
        .build()
        .join("build");

    let libraries = find_libraries(&build_root);
    link(&libraries, "skippy_llama");
    link(&libraries, "skippy_core");
    link(&libraries, "llama");
    libraries.keys().filter(|name| name.starts_with("ggml")).for_each(|name| link(&libraries, name));
    link_system_dependencies(&build_root, &libraries);

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

fn link_system_dependencies(build_root: &Path, libraries: &BTreeMap<String, Library>) {
    let target = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    if target == "macos" {
        if libraries.contains_key("ggml-metal") {
            for framework in ["Foundation", "Metal", "MetalKit"] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        let cache = fs::read_to_string(build_root.join("CMakeCache.txt")).expect("CMake cache");
        if cache.lines().any(|line| line == "GGML_ACCELERATE:BOOL=ON") || libraries.contains_key("ggml-blas") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }
    // CMake discovers the host OpenMP runtime; retain that exact link closure
    // instead of assuming libgomp on every platform/compiler.
    let cache = fs::read_to_string(build_root.join("CMakeCache.txt")).expect("CMake cache");
    if let Some(names) = cache
        .lines()
        .find_map(|line| line.strip_prefix("OpenMP_CXX_LIB_NAMES:STRING="))
        .filter(|names| !names.ends_with("NOTFOUND"))
    {
        for name in names.split(';').filter(|name| !name.is_empty()) {
            let prefix = format!("OpenMP_{name}_LIBRARY:FILEPATH=");
            if let Some(path) = cache.lines().find_map(|line| line.strip_prefix(&prefix)) {
                let dir = Path::new(path).parent().expect("OpenMP library directory");
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
            println!("cargo:rustc-link-lib={name}");
        }
    }
}

struct Library {
    dir: PathBuf,
}

/// Static archives under `root`. Ignore stale shared outputs from a previous
/// configuration: discovering those would silently restore the loader defect.
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
        let Some(name) = library_name(&path) else { continue };
        found.insert(name, Library { dir: dir.to_path_buf() });
    }
}

fn library_name(path: &Path) -> Option<String> {
    let file = path.file_name()?.to_str()?;
    let stem = file.strip_prefix("lib")?;
    if let Some(name) = stem.strip_suffix(".a") {
        return Some(name.to_owned());
    }
    None
}

fn link(libraries: &BTreeMap<String, Library>, name: &str) {
    let library = libraries.get(name).unwrap_or_else(|| panic!("library {name} not produced by the CMake build"));
    println!("cargo:rustc-link-search=native={}", library.dir.display());
    println!("cargo:rustc-link-lib=static={name}");
}
