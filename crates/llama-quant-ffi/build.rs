fn main() {
    print_rerun_envs();

    if std::env::var_os("CARGO_FEATURE_DYNAMIC_RUNTIME").is_some() {
        return;
    }

    let link_mode =
        std::env::var("LLAMA_STAGE_LINK_MODE").or_else(|_| std::env::var("SKIPPY_LLAMA_LINK_MODE"));
    if link_mode.as_deref() == Ok("dynamic") {
        link_dynamic_runtime();
        return;
    }

    let workspace_root = workspace_root();
    let target = std::env::var("TARGET").unwrap_or_default();
    let backend = std::env::var("LLAMA_STAGE_BACKEND")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BACKEND"))
        .unwrap_or_else(|_| default_backend(&target).to_string());
    let build_dir = configured_build_dir(&workspace_root, &backend);
    ensure_static_native_ready(&workspace_root, &build_dir, &target, &backend);
    emit_static_link(&build_dir, &target);
}

fn print_rerun_envs() {
    for key in [
        "LLAMA_STAGE_BUILD_DIR",
        "LLAMA_STAGE_LIB_DIR",
        "LLAMA_STAGE_LINK_MODE",
        "SKIPPY_LLAMA_BUILD_DIR",
        "SKIPPY_LLAMA_LIB_DIR",
        "SKIPPY_LLAMA_LINK_MODE",
        "LLAMA_STAGE_BACKEND",
        "SKIPPY_LLAMA_BACKEND",
        "SKIPPY_LLAMA_AUTO_BUILD",
        "MESH_LLM_AUTO_BUILD_LLAMA",
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "LLVMInstallDir",
        "VULKAN_SDK",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
}

fn link_dynamic_runtime() {
    if let Ok(lib_dir) =
        std::env::var("LLAMA_STAGE_LIB_DIR").or_else(|_| std::env::var("SKIPPY_LLAMA_LIB_DIR"))
    {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    println!("cargo:rustc-link-lib=dylib=llama-common");
    println!("cargo:rustc-link-lib=dylib=llama");
}

fn workspace_root() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("../..")
}

fn configured_build_dir(workspace_root: &std::path::Path, backend: &str) -> std::path::PathBuf {
    std::env::var("LLAMA_STAGE_BUILD_DIR")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BUILD_DIR"))
        .map(std::path::PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| {
            workspace_root.join(format!(
                ".deps/llama-build/build-stage-abi-static-{backend}"
            ))
        })
}

fn default_backend(target: &str) -> &'static str {
    if target.contains("apple") {
        "metal"
    } else {
        "cpu"
    }
}

fn ensure_static_native_ready(
    workspace_root: &std::path::Path,
    build_dir: &std::path::Path,
    target: &str,
    backend: &str,
) {
    if required_static_archives_exist(build_dir) {
        return;
    }
    if !native_auto_build_enabled() {
        panic!(
            "patched llama.cpp quant archives are missing from {}; run `just llama-build`, set LLAMA_STAGE_BUILD_DIR, or enable SKIPPY_LLAMA_AUTO_BUILD=1",
            build_dir.display()
        );
    }
    if target.contains("windows") {
        panic!(
            "patched llama.cpp quant archives are missing from {}; automatic native preparation is not supported for Windows from build.rs yet",
            build_dir.display()
        );
    }

    let prepare = workspace_root.join("scripts/prepare-llama.sh");
    let build = workspace_root.join("scripts/build-llama.sh");
    println!("cargo:rerun-if-changed={}", prepare.display());
    println!("cargo:rerun-if-changed={}", build.display());
    if !prepare.exists() || !build.exists() {
        panic!(
            "patched llama.cpp quant archives are missing from {}, and mesh-llm build scripts were not found under {}",
            build_dir.display(),
            workspace_root.display()
        );
    }

    run_native_script(
        workspace_root,
        &prepare,
        ["pinned"].as_slice(),
        backend,
        build_dir,
    );
    run_native_script(workspace_root, &build, [].as_slice(), backend, build_dir);

    if !required_static_archives_exist(build_dir) {
        panic!(
            "patched llama.cpp quant build finished but required archives are still missing from {}",
            build_dir.display()
        );
    }
}

fn native_auto_build_enabled() -> bool {
    for key in ["SKIPPY_LLAMA_AUTO_BUILD", "MESH_LLM_AUTO_BUILD_LLAMA"] {
        if let Ok(value) = std::env::var(key) {
            return !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            );
        }
    }
    true
}

fn run_native_script(
    workspace_root: &std::path::Path,
    script: &std::path::Path,
    args: &[&str],
    backend: &str,
    build_dir: &std::path::Path,
) {
    let mut command = std::process::Command::new("bash");
    command.current_dir(workspace_root).arg(script).args(args);
    command.env("LLAMA_WORKDIR", workspace_root.join(".deps/llama.cpp"));
    command.env("LLAMA_BUILD_DIR", build_dir);
    command.env("LLAMA_STAGE_BUILD_DIR", build_dir);
    command.env("LLAMA_STAGE_LINK_MODE", "static");
    command.env("LLAMA_STAGE_BACKEND", backend);
    let status = command.status().unwrap_or_else(|error| {
        panic!("failed to run {}: {error}", script.display());
    });
    if !status.success() {
        panic!("{} failed with status {status}", script.display());
    }
}

fn emit_static_link(build_dir: &std::path::Path, target: &str) {
    for dir in static_search_dirs(build_dir)
        .iter()
        .filter(|dir| dir.exists())
    {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    let cmake_cache = build_dir.join("CMakeCache.txt");
    if cmake_cache.exists() {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }
    emit_archive_reruns(build_dir);

    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    let has_cuda = static_archive_exists(
        build_dir,
        "ggml/src/ggml-cuda/libggml-cuda.a",
        "ggml/src/ggml-cuda/ggml-cuda.lib",
    );
    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    let has_hip = static_archive_exists(
        build_dir,
        "ggml/src/ggml-hip/libggml-hip.a",
        "ggml/src/ggml-hip/ggml-hip.lib",
    );
    if has_hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    let has_vulkan = static_archive_exists(
        build_dir,
        "ggml/src/ggml-vulkan/libggml-vulkan.a",
        "ggml/src/ggml-vulkan/ggml-vulkan.lib",
    );
    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if static_archive_exists(
        build_dir,
        "ggml/src/ggml-blas/libggml-blas.a",
        "ggml/src/ggml-blas/ggml-blas.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if static_archive_exists(
        build_dir,
        "ggml/src/ggml-metal/libggml-metal.a",
        "ggml/src/ggml-metal/ggml-metal.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");
    emit_system_links(
        build_dir,
        &cmake_cache,
        target,
        has_cuda,
        has_hip,
        has_vulkan,
    );
}

fn static_search_dirs(build_dir: &std::path::Path) -> [std::path::PathBuf; 9] {
    [
        build_dir.join("common"),
        build_dir.join("src"),
        build_dir.join("ggml/src"),
        build_dir.join("ggml/src/ggml-cpu"),
        build_dir.join("ggml/src/ggml-blas"),
        build_dir.join("ggml/src/ggml-cuda"),
        build_dir.join("ggml/src/ggml-hip"),
        build_dir.join("ggml/src/ggml-metal"),
        build_dir.join("ggml/src/ggml-vulkan"),
    ]
}

fn emit_archive_reruns(build_dir: &std::path::Path) {
    for (unix_archive, msvc_archive) in [
        ("src/libllama.a", "src/llama.lib"),
        ("common/libllama-common.a", "common/llama-common.lib"),
        (
            "common/libllama-common-base.a",
            "common/llama-common-base.lib",
        ),
        ("ggml/src/libggml.a", "ggml/src/ggml.lib"),
        ("ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"),
        (
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "ggml/src/ggml-cpu/ggml-cpu.lib",
        ),
        ("ggml/src/libggml-cpu.a", "ggml/src/ggml-cpu.lib"),
        (
            "ggml/src/ggml-blas/libggml-blas.a",
            "ggml/src/ggml-blas/ggml-blas.lib",
        ),
        (
            "ggml/src/ggml-cuda/libggml-cuda.a",
            "ggml/src/ggml-cuda/ggml-cuda.lib",
        ),
        (
            "ggml/src/ggml-hip/libggml-hip.a",
            "ggml/src/ggml-hip/ggml-hip.lib",
        ),
        (
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ),
        (
            "ggml/src/ggml-vulkan/libggml-vulkan.a",
            "ggml/src/ggml-vulkan/ggml-vulkan.lib",
        ),
    ] {
        for archive in [unix_archive, msvc_archive]
            .iter()
            .map(|path| build_dir.join(path))
            .filter(|archive| archive.exists())
        {
            println!("cargo:rerun-if-changed={}", archive.display());
        }
    }
}

fn emit_system_links(
    build_dir: &std::path::Path,
    cmake_cache: &std::path::Path,
    target: &str,
    has_cuda: bool,
    has_hip: bool,
    has_vulkan: bool,
) {
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if static_archive_exists(
            build_dir,
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ) {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        link_linux_openmp(cmake_cache);
        if has_cuda {
            link_linux_cuda_libs(cmake_cache);
        }
        if has_hip {
            link_linux_hip_libs();
        }
        if has_vulkan {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    }
}

fn link_linux_openmp(cmake_cache: &std::path::Path) {
    let Ok(cache) = std::fs::read_to_string(cmake_cache) else {
        return;
    };
    if cmake_cache_value(&cache, "GGML_OPENMP_ENABLED").as_deref() != Some("ON") {
        return;
    }

    let mut libraries = Vec::new();
    for cache_key in ["OpenMP_C_LIB_NAMES", "OpenMP_CXX_LIB_NAMES"] {
        if let Some(names) = cmake_cache_value(&cache, cache_key) {
            for library in names
                .split(';')
                .filter(|library| !library.is_empty() && *library != "pthread")
            {
                if !libraries.iter().any(|existing| existing == library) {
                    libraries.push(library.to_string());
                }
            }
        }
    }
    if libraries.is_empty() {
        libraries.push("gomp".to_string());
    }
    for library in libraries {
        link_linux_lib_from_cache(cmake_cache, &format!("OpenMP_{library}_LIBRARY"), &library);
    }
}

fn required_static_archives_exist(build_dir: &std::path::Path) -> bool {
    [
        &["src/libllama.a", "src/llama.lib"][..],
        &["common/libllama-common.a", "common/llama-common.lib"],
        &[
            "common/libllama-common-base.a",
            "common/llama-common-base.lib",
        ],
        &["ggml/src/libggml.a", "ggml/src/ggml.lib"],
        &["ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"],
        &[
            "ggml/src/libggml-cpu.a",
            "ggml/src/ggml-cpu.lib",
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "ggml/src/ggml-cpu/ggml-cpu.lib",
        ],
    ]
    .iter()
    .all(|candidates| {
        candidates
            .iter()
            .any(|candidate| build_dir.join(candidate).exists())
    })
}

fn static_archive_exists(
    build_dir: &std::path::Path,
    unix_archive: &str,
    msvc_archive: &str,
) -> bool {
    build_dir.join(unix_archive).exists() || build_dir.join(msvc_archive).exists()
}

fn link_linux_cuda_libs(cmake_cache: &std::path::Path) {
    for (cache_key, lib) in [
        ("CUDA_cuda_driver_LIBRARY", "cuda"),
        ("CUDA_cudart_LIBRARY", "cudart"),
        ("CUDA_cublas_LIBRARY", "cublas"),
        ("CUDA_cublasLt_LIBRARY", "cublasLt"),
    ] {
        link_linux_lib_from_cache(cmake_cache, cache_key, lib);
    }
}

fn link_linux_hip_libs() {
    for search_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib"] {
        if std::path::Path::new(search_path).is_dir() {
            println!("cargo:rustc-link-search=native={search_path}");
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_linux_lib_from_cache(cmake_cache: &std::path::Path, cache_key: &str, lib: &str) {
    if let Ok(cache) = std::fs::read_to_string(cmake_cache)
        && let Some(path) = cmake_cache_value(&cache, cache_key)
    {
        let path = std::path::PathBuf::from(path);
        if path.exists()
            && let Some(parent) = path.parent()
        {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

fn cmake_cache_value(cache: &str, key: &str) -> Option<String> {
    cache.lines().find_map(|line| {
        let (lhs, rhs) = line.split_once('=')?;
        let (name, _) = lhs.split_once(':')?;
        (name == key).then(|| rhs.to_string())
    })
}
