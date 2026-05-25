fn main() {
    if target_os_is("macos") {
        build_metal();
    }

    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build_cuda();
    }

    if std::env::var_os("CARGO_FEATURE_HIP").is_some() {
        build_hip();
    }

    if std::env::var_os("CARGO_FEATURE_INTEL").is_some() {
        build_intel();
    }
}

fn target_os_is(os: &str) -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok(os)
}

fn build_metal() {
    let object = out_path("mesh_llm_gpu_bench_metal.o");
    run_or_panic({
        let mut command = std::process::Command::new("clang");
        command.arg("-O3").arg("-fobjc-arc").arg("-fPIC").arg("-c");
        add_macos_target_flags(&mut command);
        command
            .arg("native/metal/membench_metal.m")
            .arg("-o")
            .arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_metal");

    println!("cargo:rerun-if-changed=native/metal/membench_metal.m");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
}

fn add_macos_target_flags(command: &mut std::process::Command) {
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        Ok(arch) => panic!("unsupported macOS Metal benchmark target architecture: {arch}"),
        Err(err) => panic!("CARGO_CFG_TARGET_ARCH is required for Metal benchmark build: {err}"),
    };
    command.arg("-arch").arg(arch);

    if let Some(sdk_path) = macos_sdk_path() {
        command.arg("-isysroot").arg(sdk_path);
    }

    let deployment_target =
        std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "13.0".to_string());
    command.arg(format!("-mmacosx-version-min={deployment_target}"));
}

fn macos_sdk_path() -> Option<String> {
    let output = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "--show-sdk-path"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (!path.is_empty()).then(|| path.to_string())
}

fn native_source(dir: &str, name: &str) -> String {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir
        .join("native")
        .join(dir)
        .join(name)
        .display()
        .to_string()
}

fn out_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(name)
}

fn write_wrapper(name: &str, source: &str, symbol: &str) -> std::path::PathBuf {
    let wrapper = out_path(name);
    let body = format!(
        "#define main {symbol}_program_main\n#include \"{source}\"\n#undef main\nextern \"C\" int {symbol}(void) {{ char arg0[] = \"{symbol}\"; char arg1[] = \"--json\"; char *argv[] = {{ arg0, arg1, nullptr }}; return {symbol}_program_main(2, argv); }}\n"
    );
    std::fs::write(&wrapper, body).unwrap();
    println!("cargo:rerun-if-changed={source}");
    wrapper
}

fn run_or_panic(mut command: std::process::Command) {
    let status = command.status().unwrap_or_else(|err| {
        panic!(
            "failed to run native benchmark compiler {:?}: {err}",
            command
        )
    });
    assert!(
        status.success(),
        "native benchmark compiler {:?} failed with {status}",
        command
    );
}

fn archive_static_lib(object: &std::path::Path, lib_name: &str) {
    if cfg!(windows) {
        cc::Build::new().object(object).compile(lib_name);
        return;
    }

    let lib_path = out_path(&format!("lib{lib_name}.a"));
    run_or_panic({
        let mut command = std::process::Command::new("ar");
        command.arg("crus").arg(&lib_path).arg(object);
        command
    });
    println!("cargo:rustc-link-search=native={}", out_path("").display());
    println!("cargo:rustc-link-lib=static={lib_name}");
}

fn target_is_windows_msvc() -> bool {
    std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
}

fn target_uses_static_crt() -> bool {
    std::env::var("CARGO_CFG_TARGET_FEATURE")
        .map(|features| features.split(',').any(|feature| feature == "crt-static"))
        .unwrap_or(false)
}

fn add_windows_cuda_crt_flags(command: &mut std::process::Command) {
    if target_is_windows_msvc() {
        let runtime = if target_uses_static_crt() {
            "/MT"
        } else {
            "/MD"
        };
        command.arg("-Xcompiler").arg(runtime);
    }
}

fn add_windows_hip_crt_flags(command: &mut std::process::Command) {
    if target_is_windows_msvc() {
        let runtime = if target_uses_static_crt() {
            "-fms-runtime-lib=static"
        } else {
            "-fms-runtime-lib=dll"
        };
        command.arg(runtime);
    }
}

fn build_cuda() {
    let source = native_source("cuda", "membench-fingerprint.cu");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_cuda_wrapper.cu",
        &source,
        "mesh_llm_gpu_bench_cuda_main",
    );
    let object = out_path("mesh_llm_gpu_bench_cuda.o");
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(nvcc);
        command.arg("-O3").arg("-std=c++17");
        add_windows_cuda_crt_flags(&mut command);
        if !cfg!(windows) {
            command.arg("-Xcompiler").arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn build_hip() {
    let source = native_source("hip", "membench-fingerprint.hip");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_hip_wrapper.hip",
        &source,
        "mesh_llm_gpu_bench_hip_main",
    );
    let object = out_path("mesh_llm_gpu_bench_hip.o");
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(hipcc);
        command.arg("-O3").arg("-std=c++17");
        add_windows_hip_crt_flags(&mut command);
        if !cfg!(windows) {
            command.arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_hip");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

fn build_intel() {
    let source = native_source("intel", "membench-fingerprint-intel.cpp");
    let wrapper = write_wrapper(
        "mesh_llm_gpu_bench_intel_wrapper.cpp",
        &source,
        "mesh_llm_gpu_bench_intel_main",
    );
    let object = out_path("mesh_llm_gpu_bench_intel.o");
    let icpx = std::env::var("ICPX").unwrap_or_else(|_| "icpx".to_string());
    run_or_panic({
        let mut command = std::process::Command::new(icpx);
        command.arg("-O3").arg("-fsycl");
        if !cfg!(windows) {
            command.arg("-fPIC");
        }
        command.arg("-c").arg(&wrapper).arg("-o").arg(&object);
        command
    });
    archive_static_lib(&object, "mesh_llm_gpu_bench_intel");
    println!("cargo:rustc-link-lib=dylib=sycl");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
