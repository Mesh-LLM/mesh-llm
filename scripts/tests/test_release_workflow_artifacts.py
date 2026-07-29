from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def job_block(workflow: str, job_name: str, next_job_name: str) -> str:
    start = workflow.index(f"  {job_name}:")
    end = workflow.index(f"  {next_job_name}:", start)
    return workflow[start:end]


class ReleaseWorkflowArtifactTests(unittest.TestCase):
    def test_release_entrypoint_rejects_untrusted_refs(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(workflow, "metadata", "build")

        manual_guard = metadata.index("Require the trusted release ref")
        checkout = metadata.index("uses: actions/checkout@")
        selector = metadata.index(
            "uses: ./.github/actions/select-ci-runners",
        )
        self.assertLess(manual_guard, checkout)
        self.assertLess(checkout, selector)
        self.assertIn(
            '"$GITHUB_EVENT_NAME" == "workflow_dispatch"',
            metadata,
        )
        self.assertIn(
            '"$GITHUB_REF" != "refs/heads/main"',
            metadata,
        )
        self.assertIn(
            'git merge-base --is-ancestor "$GITHUB_SHA" '
            "refs/remotes/origin/main",
            metadata,
        )
        self.assertIn("Reject an existing manual release tag", metadata)
        self.assertIn(
            "Manual release tag already exists and is immutable",
            metadata,
        )

    def test_release_depot_policy_is_main_ref_only_and_selected_once(
        self,
    ) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(workflow, "metadata", "build")

        self.assertIn("use_depot:", workflow[: workflow.index("\njobs:\n")])
        self.assertIn(
            "uses: ./.github/actions/select-ci-runners",
            metadata,
        )
        self.assertIn("ref: ${{ github.ref }}", metadata)
        self.assertIn(
            "depot_main_enabled: ${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            metadata,
        )
        self.assertIn(
            "manual_use_depot: ${{ inputs.use_depot == true }}",
            metadata,
        )
        self.assertEqual(
            workflow.count("uses: ./.github/actions/select-ci-runners"),
            1,
        )

    def test_release_routes_only_initial_non_secret_linux_lanes(
        self,
    ) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        host = job_block(workflow, "build", "compose_cpu_products")
        sdk_runtime = job_block(
            workflow,
            "build_native_sdk_runtime",
            "build_native_runtime",
        )
        native_runtime = job_block(
            workflow,
            "build_native_runtime",
            "build_native_runtime_linux_aarch64_cuda",
        )
        rocm = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_rocm",
            "build_native_runtime_linux_x86_64_vulkan",
        )
        vulkan = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_vulkan",
            "build_swift_sdk_artifact",
        )
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("runs-on: ${{ matrix.os }}", host)
        self.assertIn("RELEASE_ATTESTATION_SIGNING_KEY", host)
        for producer in (sdk_runtime, native_runtime):
            self.assertIn(
                "matrix.target == 'x86_64-unknown-linux-gnu'",
                producer,
            )
            self.assertIn(
                "needs.metadata.outputs.runner_8",
                producer,
            )
        for producer in (rocm, vulkan):
            self.assertIn(
                "runs-on: ${{ needs.metadata.outputs.runner_16 }}",
                producer,
            )
            self.assertIn(
                "allow_depot_remote_cache: "
                "${{ needs.metadata.outputs.allow_depot_remote_cache }}",
                producer,
            )
        self.assertIn("runs-on: ubuntu-24.04", publish)
        self.assertNotIn("needs.metadata.outputs.runner", publish)

    def test_inference_smoke_consumes_composed_product(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("ci-release-linux-inference-product"),
            2,
        )
        self.assertNotIn("release-linux-inference-binary", workflow)
        self.assertIn(
            "uses: ./.github/actions/compose-product-input",
            workflow,
        )
        self.assertIn("output_dir: product-input", workflow)
        self.assertIn(
            "path: ${{ steps.compose.outputs.archive_path }}",
            workflow,
        )

    def test_release_permissions_are_least_privilege(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        header = workflow[: workflow.index("\njobs:\n")]
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn(
            "permissions:\n  contents: read\n  packages: read",
            header,
        )
        self.assertNotIn("contents: write", header)
        self.assertNotIn("packages: write", header)
        self.assertIn(
            "    permissions:\n      contents: write",
            publish,
        )
        self.assertNotIn("packages: write", publish)

    def test_publish_fan_in_stops_when_release_is_cancelled(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("if: ${{ !cancelled()", publish)
        self.assertNotIn("always()", publish)

    def test_release_assets_and_manual_tags_are_immutable(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("Release tag already exists and cannot be reused", publish)
        self.assertNotIn("reusing it", publish)
        self.assertIn("overwrite_files: false", publish)
        self.assertIn("persist-credentials: false", publish)
        self.assertNotIn("persist-credentials: true", publish)
        self.assertIn(
            'git push "$release_remote" "refs/tags/$RELEASE_TAG"',
            publish,
        )
        self.assertNotIn(
            'git push origin "refs/tags/$RELEASE_TAG"',
            publish,
        )

    def test_release_push_token_is_isolated_to_the_push_step(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )
        generate_start = publish.index(
            "- name: Generate native runtime release manifest",
        )
        prepare_start = publish.index(
            "- name: Prepare dispatched release tag",
        )
        push_start = publish.index(
            "- name: Push dispatched release tag",
        )
        release_start = publish.index(
            "- name: Publish GitHub release",
        )
        generate = publish[generate_start:prepare_start]
        prepare = publish[prepare_start:push_start]
        push = publish[push_start:release_start]

        token_binding = "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}"
        self.assertNotIn(token_binding, generate)
        self.assertNotIn(token_binding, prepare)
        self.assertIn(token_binding, push)
        self.assertEqual(publish.count(token_binding), 1)
        self.assertNotIn("release_remote=", prepare)
        self.assertNotIn("git push", prepare)
        self.assertIn(
            'git push "$release_remote" "refs/tags/$RELEASE_TAG"',
            push,
        )

    def test_arm64_smoke_requires_integrity_and_safe_extraction(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        smoke = job_block(
            workflow,
            "smoke_linux_arm64_artifact",
            "compose_linux_aarch64_cuda",
        )

        self.assertIn("scripts/verify-checksum-sidecar.py", smoke)
        self.assertIn("scripts/safe-extract-tar.py", smoke)
        self.assertNotIn("tar -xzf", smoke)
        self.assertNotIn("command -v sha256sum", smoke)

    def test_native_sdk_assets_are_staged_flat_for_publishing(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "build_native_sdk_runtime",
            "build_native_runtime",
        )
        upload = producer[producer.index("- name: Upload native SDK runtime") :]
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("- name: Stage flat native SDK release assets", producer)
        self.assertIn(
            "native SDK release asset basename collision",
            producer,
        )
        self.assertIn("path: release-native-sdk-assets/*", upload)
        self.assertNotIn("dist/native-sdk/", upload)
        self.assertNotIn("dist/native-sdk-crates/", upload)
        self.assertIn("files: release-artifacts/*", publish)

    def test_windows_host_publishes_prebuilt_attestation_verifier(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "windows_host_input",
            "compose_windows_gpu",
        )

        self.assertIn(
            "uses: ./.github/actions/prepare-windows-host-input",
            producer,
        )
        self.assertIn("profile: release", producer)
        self.assertIn(
            "attestation_signing_key_file:",
            producer,
        )
        self.assertIn(
            "attestation_public_key_file:",
            producer,
        )
        self.assertIn("path: host-input/*", producer)
        self.assertNotIn("prepare-native-runtime-input", producer)
        self.assertNotIn("compose-product-input", producer)

    def test_windows_composers_use_shared_verified_product_action(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        jobs = (
            ("compose_windows_gpu", "compose_windows_cpu"),
            ("compose_windows_cpu", "build_native_runtime_windows_cpu"),
        )

        for job_name, next_job_name in jobs:
            with self.subTest(job=job_name):
                job = job_block(workflow, job_name, next_job_name)
                composition = job.index(
                    "uses: ./.github/actions/compose-product-input",
                )
                packaging = job.index(
                    "- name: Package verified Windows",
                )
                self.assertLess(composition, packaging)
                self.assertIn(
                    "binary_name: mesh-llm.exe",
                    job,
                )
                self.assertIn(
                    "attestation_verifier: host-input/release-attestation-verifier.exe",
                    job,
                )
                self.assertIn(
                    "version: ${{ needs.metadata.outputs.tag }}",
                    job,
                )
                expected_backend = (
                    "backend: ${{ matrix.backend }}"
                    if job_name == "compose_windows_gpu"
                    else "backend: cpu"
                )
                self.assertIn(expected_backend, job)
                self.assertIn('readiness_smoke: "true"', job)
                self.assertIn(
                    "MESH_LLM_PRECOMPOSED_PRODUCT_DIR: ${{ steps.compose.outputs.product_dir }}",
                    job,
                )
                self.assertIn(
                    'MESH_RELEASE_ATTESTATION_PREVERIFIED: "1"',
                    job,
                )
                self.assertNotIn("Verify immutable runtime archive", job)
                self.assertNotIn("tar -xzf", job)
                self.assertNotIn("ci-client-readiness-smoke.sh", job)
                self.assertNotIn("cargo run", job)
                self.assertNotIn("dtolnay/rust-toolchain", job)
                self.assertNotIn("sccache-action", job)

    def test_windows_cuda12_label_rejects_other_toolkit_majors(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "build_native_runtime_windows_gpu",
            "publish",
        )

        validation = producer.index("- name: Validate CUDA 12 artifact contract")
        installation = producer.index("- name: Install CUDA toolkit")
        self.assertLess(validation, installation)
        self.assertIn("$cudaMajor -ne '12'", producer)
        self.assertIn(
            "release-native-runtime-windows-x86_64-cuda12",
            producer,
        )

    def test_linux_cuda_composition_uses_hosted_runner(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        composer = job_block(
            workflow,
            "compose_linux_cuda",
            "compose_linux_rocm",
        )
        job_header = composer[: composer.index("    steps:")]

        self.assertIn(
            "    runs-on: ${{ needs.metadata.outputs.runner_4 }}",
            job_header,
        )
        self.assertNotIn("self-hosted", job_header)
        self.assertNotIn("USE_SELF_HOSTED", job_header)

    def test_release_uses_shared_host_and_runtime_producers(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("uses: ./.github/actions/prepare-host-input"),
            2,
        )
        self.assertEqual(
            workflow.count(
                "uses: ./.github/actions/prepare-windows-host-input",
            ),
            1,
        )
        self.assertGreaterEqual(
            workflow.count("uses: ./.github/actions/prepare-native-runtime-input"),
            5,
        )
        self.assertEqual(
            workflow.count("uses: ./.github/actions/compose-product-input"),
            8,
        )
        self.assertNotIn(
            "scripts/ci-client-readiness-smoke.sh host-input/mesh-llm runtime-root",
            workflow,
        )

    def test_release_product_jobs_do_not_restore_compiler_caches(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        product_jobs = (
            "compose_linux_aarch64_cuda",
            "compose_linux_cuda",
            "compose_linux_rocm",
            "compose_linux_vulkan",
        )

        for index, job_name in enumerate(product_jobs):
            start = workflow.index(f"  {job_name}:")
            next_starts = [
                workflow.find(f"  {other_job}:", start + 1)
                for other_job in product_jobs[index + 1 :]
            ]
            next_starts = [position for position in next_starts if position >= 0]
            end = min(next_starts) if next_starts else len(workflow)
            job = workflow[start:end]
            self.assertIn(
                "uses: ./.github/actions/compose-product-input",
                job,
            )
            self.assertNotIn(
                "uses: ./.github/actions/configure-sccache-gha",
                job,
            )
            self.assertNotIn("uses: actions/cache@", job)


if __name__ == "__main__":
    unittest.main()
