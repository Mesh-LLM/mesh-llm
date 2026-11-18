use super::{assert_mutation_rejected, document, input, job, named, steps, support, text};
use std::{fs, process::Command};

#[test]
fn backend_projection_preserves_exact_catalog_rows_and_native_capacity() {
    let catalog: serde_json::Value =
        serde_json::from_slice(&fs::read(support::root().join("ci/slices.yml")).unwrap()).unwrap();
    let rows = catalog["runtime_rows"]
        .as_array()
        .expect("runtime catalog rows");
    let rows = rows
        .iter()
        .filter(|row| row["platform"] == "linux")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 4);
    for component in ["runtime", "product"] {
        let doc = document(&format!("ci-linux-{component}-slice.yml"));
        let select = named(job(&doc, "runner_policy"), "Select requested backend rows");
        for backend in ["", "cpu", "cuda", "rocm", "vulkan", "unknown"] {
            let fixture = support::Fixture::new();
            let output_path = fixture.path().join("outputs");
            let mut command = Command::new("bash");
            command
                .args(["-c", text(select, "run").unwrap()])
                .current_dir(fixture.path())
                .env("BACKEND", backend)
                .env("RUNTIME_MATRIX", serde_json::to_string(&rows).unwrap())
                .env("GITHUB_OUTPUT", &output_path);
            let output = fixture.run(command);
            assert_eq!(
                output.status.success(),
                backend != "unknown",
                "{backend}: {output:?}"
            );
            if backend == "unknown" {
                continue;
            }
            let contents = fs::read_to_string(output_path).unwrap();
            let selected: serde_json::Value =
                serde_json::from_str(contents.trim().strip_prefix("runtime_matrix=").unwrap())
                    .unwrap();
            let expected = rows
                .iter()
                .filter(|row| backend.is_empty() || row["backend"] == backend)
                .copied()
                .collect::<Vec<_>>();
            assert_eq!(selected, serde_json::to_value(expected).unwrap());
        }
    }
}

#[test]
fn artifact_runtime_image_verification_preserves_each_backend_argument() {
    let document = document("ci-linux-runtime-slice.yml");
    let step = named(
        job(&document, "linux_runtime"),
        "Verify prebuilt native environment",
    );
    assert_eq!(
        text(step.get("env").unwrap(), "VERIFY_BACKEND"),
        Some("${{ matrix.runtime.verify_backend }}")
    );
    for (value, expected) in [
        ("public", "public\n"),
        ("public cuda", "public\ncuda\n"),
        ("public rocm", "public\nrocm\n"),
        ("public vulkan", "public\nvulkan\n"),
    ] {
        let fixture = support::Fixture::new();
        fixture.executable("verify-runner-image", "printf '%s\\n' \"$@\" > arguments");
        let mut command = Command::new("/bin/bash");
        command
            .env_clear()
            .current_dir(fixture.path())
            .env("PATH", fixture.path().join("bin"))
            .env("VERIFY_BACKEND", value);
        command.args(["-c", text(step, "run").unwrap()]);
        let output = fixture.run(command);
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            fs::read_to_string(fixture.path().join("arguments")).unwrap(),
            expected
        );
    }
}

fn cuda_versions(document: &super::Node) -> bool {
    let runtime = job(document, "windows_runtime");
    let environment = runtime.get("env").unwrap();
    let version = "${{ vars.CUDA_VERSION || '12.6.3' }}";
    text(environment, "WINDOWS_CUDA_VERSION") == Some(version)
        && text(environment, "MESH_CUDA_VERSION") == Some(version)
        && input(named(runtime, "Install CUDA toolkit"), "cuda")
            == Some("${{ env.WINDOWS_CUDA_VERSION }}")
}

#[test]
fn artifact_windows_cuda_installation_and_native_build_share_one_version() {
    let file = "ci-windows-runtime-slice.yml";
    assert!(cuda_versions(&document(file)));
    for key in ["WINDOWS_CUDA_VERSION", "MESH_CUDA_VERSION"] {
        assert_mutation_rejected(
            file,
            &format!("{key}: ${{{{ vars.CUDA_VERSION || '12.6.3' }}}}"),
            &format!("{key}: 12.9.2"),
            cuda_versions,
        );
    }
    assert_mutation_rejected(
        file,
        "cuda: ${{ env.WINDOWS_CUDA_VERSION }}",
        "cuda: 12.9.2",
        cuda_versions,
    );
}

fn no_dependency_cache(document: &super::Node) -> bool {
    document
        .get("env")
        .is_none_or(|env| env.get("CACHE_NAMESPACE").is_none())
        && document
            .get("jobs")
            .unwrap()
            .entries()
            .iter()
            .all(|(_, job)| {
                job.get("env")
                    .is_none_or(|env| env.get("CACHE_NAMESPACE").is_none())
                    && job.get("steps").is_none_or(|_| {
                        steps(job).iter().all(|step| {
                            !text(step, "uses")
                                .is_some_and(|uses| uses.starts_with("actions/cache@"))
                                && !matches!(input(step, "cache"), Some("pnpm" | "npm"))
                        })
                    })
            })
}

#[test]
fn artifact_web_and_console_use_baked_stores_and_resolved_website_dependencies() {
    for (file, jobs) in [
        ("ci-ui-artifact-slice.yml", vec!["ui_artifact"]),
        ("ci-web-slice.yml", vec!["ui_quality", "ui_e2e"]),
    ] {
        let document = document(file);
        assert!(no_dependency_cache(&document));
        for name in jobs {
            let store = named(
                job(&document, name),
                "Point pnpm at the image's baked store",
            );
            assert_eq!(
                text(store, "working-directory"),
                Some("${{ steps.layout.outputs.ui_dir }}")
            );
            assert_eq!(
                text(store, "run"),
                Some("pnpm config set store-dir /home/runner/.local/share/pnpm/store")
            );
        }
        for quoted in ["pnpm", "'pnpm'", "\"pnpm\"", "npm", "'npm'", "\"npm\""] {
            assert_mutation_rejected(
                file,
                "run: pnpm config set store-dir /home/runner/.local/share/pnpm/store",
                &format!(
                    "run: pnpm config set store-dir /home/runner/.local/share/pnpm/store\n        with:\n          cache: {quoted}"
                ),
                no_dependency_cache,
            );
        }
    }
    let web = document("ci-web-slice.yml");
    let install = named(job(&web, "website"), "Install website dependencies");
    assert_eq!(
        text(install, "working-directory"),
        Some("${{ steps.layout.outputs.website_dir }}")
    );
    assert_eq!(text(install, "run"), Some("npm ci"));
}
