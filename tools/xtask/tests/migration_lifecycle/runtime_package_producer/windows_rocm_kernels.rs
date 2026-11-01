//! Windows ROCm packages advertise only the targets whose rocBLAS kernels they carry (#2768).
use super::{Fixture, owner};
use std::fs;

fn windows_rocm_sdk(fixture: &Fixture) {
    for (library, files) in [
        (
            "rocblas",
            &[
                "TensileLibrary_lazy_gfx1100.dat",
                "TensileLibrary_Type_HH_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1100.co",
                "Kernels.so-000-gfx1201.hsaco",
                // A catalog without code objects is not a kernel.
                "TensileLibrary_lazy_gfx1102.dat",
                "TensileLibrary_Type_HH_Contraction_l_Ailk_Bjlk_Cijk_Dijk_fallback.dat",
            ][..],
        ),
        // hipBLASLt has gfx1103 kernels in the HIP SDK; they do not count,
        // since the failure on such a GPU is rocBLAS's.
        ("hipblaslt", &["Kernels.so-000-gfx1103.hsaco"][..]),
    ] {
        let dir = fixture.path().join("sdk/bin").join(library).join("library");
        fs::create_dir_all(&dir).unwrap();
        for file in files {
            fs::write(dir.join(file), "").unwrap();
        }
        fs::write(
            fixture
                .path()
                .join("stage/lib")
                .join(format!("{library}.dll")),
            "",
        )
        .unwrap();
    }
}

fn package_kernels(targets: &str) -> (bool, String, String) {
    let functions = owner(
        "copy_windows_rocm_kernel_libraries() {",
        "hip_offload_arch_args() {",
    );
    let fixture = Fixture::new();
    windows_rocm_sdk(&fixture);
    let body = format!(
        "{functions}\nROCM_PATH=\"$TEST_ROOT/sdk\"\nLLAMA_STAGE_AMDGPU_TARGETS='{targets}'\ncopy_windows_rocm_kernel_libraries \"$TEST_ROOT/stage/lib\"\nadvertise_windows_rocm_targets_with_kernels \"$TEST_ROOT/stage/lib\"\nprintf '%s\\n' \"$LLAMA_STAGE_AMDGPU_TARGETS\"\n"
    );
    let result = fixture.run(&body, &[]);
    fixture.directory.close().unwrap();
    result
}

#[test]
fn windows_rocm_package_drops_targets_without_rocblas_kernels_from_the_manifest_targets() {
    let (ok, out, error) = package_kernels("gfx90a;gfx1100, gfx1102;gfx1103;gfx1201");
    assert!(ok, "{error}");
    assert_eq!(out, "gfx1100;gfx1201\n");
    assert!(
        error
            .contains("not advertising gfx90a gfx1102 gfx1103: the HIP SDK has no rocBLAS kernels"),
        "{error}"
    );
}

#[test]
fn windows_rocm_package_keeps_targets_that_all_have_rocblas_kernels() {
    let (ok, out, error) = package_kernels("gfx1100;gfx1201");
    assert!(ok, "{error}");
    assert_eq!(out, "gfx1100;gfx1201\n");
    assert!(error.is_empty(), "{error}");
}

#[test]
fn windows_rocm_package_refuses_targets_with_no_rocblas_kernels_at_all() {
    let (ok, _, error) = package_kernels("gfx1103");
    assert!(!ok);
    assert!(
        error.contains("no rocBLAS kernels for any ROCm target (gfx1103)"),
        "{error}"
    );
}
