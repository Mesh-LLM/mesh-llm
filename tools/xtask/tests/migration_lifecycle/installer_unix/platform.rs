//! Platform and CUDA release selection use real installer functions with inert probe outputs.
use super::fixture::{Fixture, stderr, stdout};

#[test]
fn installer_arm_platform_support_keeps_cpu_fallback_and_jetson_cuda_recommendation() {
    let fixture = Fixture::new();
    let report = fixture.run(
        "export MESH_LLM_TEST_UNAME_S=Linux MESH_LLM_TEST_UNAME_M=aarch64 MESH_LLM_TEST_CUDA_MAJOR='' MESH_LLM_TEST_TEGRA_MODEL='fixture ordinary host'\nplatform_support_status\nsupported_flavors\nrecommended_flavor\nasset_name cpu\nasset_name cuda",
    );
    assert!(!report.success());
    assert_eq!(
        stdout(&report),
        "supported\ncuda cpu\ncpu\nmesh-llm-aarch64-unknown-linux-gnu.tar.gz\n"
    );
    assert!(stderr(&report).contains("could not determine a supported CUDA major version"));
    assert!(stderr(&report).contains("--flavor cpu"));
    assert!(!stderr(&report).contains("--flavor vulkan"));
    let report = fixture.run(
        "export MESH_LLM_TEST_UNAME_S=Linux MESH_LLM_TEST_UNAME_M=aarch64 MESH_LLM_TEST_TEGRA_MODEL='NVIDIA Jetson AGX Orin'\nrecommended_flavor",
    );
    assert!(report.success(), "{report:?}");
    assert_eq!(stdout(&report), "cuda\n");
    let report = fixture.run(
        "export MESH_LLM_TEST_UNAME_S=Linux MESH_LLM_TEST_UNAME_M=armv7l\nplatform_support_status\nplatform_error_message",
    );
    assert!(report.success(), "{report:?}");
    assert!(stdout(&report).contains("recognized-unsupported\n"));
    assert!(stdout(&report).contains("Linux/arm"));
}

#[test]
fn installer_cuda_assets_bind_architecture_and_supported_major_from_probe_or_override() {
    for arch in ["aarch64", "x86_64"] {
        for override_major in [true, false] {
            let fixture = Fixture::new();
            let setup = if override_major {
                "export MESH_LLM_TEST_CUDA_MAJOR=13"
            } else {
                "detect_cuda_major() { printf '13\\n'; }"
            };
            let report = fixture.run(&format!(
                "export MESH_LLM_TEST_UNAME_S=Linux MESH_LLM_TEST_UNAME_M={arch}\n{setup}\nasset_name cuda"
            ));
            assert!(report.success(), "{report:?}");
            assert_eq!(
                stdout(&report),
                format!("mesh-llm-{arch}-unknown-linux-gnu-cuda-13.tar.gz\n")
            );
        }
    }
}

#[test]
fn installer_cuda_detection_clamps_driver_and_requires_a_compatible_complete_toolkit() {
    for (driver, libraries, expected) in [
        (13, None, "13"),
        (13, Some([13, 13, 13]), "13"),
        (12, Some([12, 12, 12]), "12"),
        (12, Some([13, 13, 13]), "12"),
        (14, Some([14, 14, 14]), "13"),
        (13, Some([13, 12, 13]), "13"),
        (11, Some([13, 13, 13]), ""),
    ] {
        let fixture = Fixture::new();
        fixture.stub(
            "nvidia-smi",
            &format!("#!/bin/bash\nprintf 'CUDA Version: {driver}.0\\n'\n"),
        );
        let body = libraries.map_or_else(
            || "#!/bin/bash\nexit 1\n".to_owned(),
            |majors| {
                let lines = ["libcudart", "libcublas", "libcublasLt"]
                    .into_iter()
                    .zip(majors)
                    .map(|(library, major)| format!("printf '{library}.so.{major}\\n'\n"))
                    .collect::<String>();
                format!("#!/bin/bash\n{lines}")
            },
        );
        fixture.stub("ldconfig", &body);
        let report = fixture.run("detect_cuda_major");
        assert!(report.success(), "{report:?}");
        assert_eq!(stdout(&report), format!("{expected}\n"));
    }
}

#[test]
fn installer_cuda_detection_reads_matching_library_versions_without_a_driver() {
    let fixture = Fixture::new();
    fixture.stub("ldconfig", "#!/bin/bash\nprintf '%s\\n' 'libcudart.so.12' 'libcudart.so.13' 'libcublas.so.13' 'libcublasLt.so.13'\n");
    let report = fixture.run("detect_cuda_major");
    assert!(report.success(), "{report:?}");
    assert_eq!(stdout(&report), "13\n");
}
