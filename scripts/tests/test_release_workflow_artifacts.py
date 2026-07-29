from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


class ReleaseWorkflowArtifactTests(unittest.TestCase):
    def test_macos_cpu_composer_installs_attestation_linker(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        composer = self.job_block(workflow, "compose_cpu_products", "inference_smoke_tests")

        self.assertIn("Install macOS attestation verifier linker", composer)
        self.assertIn("if: runner.os == 'macOS'", composer)
        self.assertIn("run: brew install lld", composer)

    def test_container_product_composers_run_bash(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        for step_name in (
            "Compose aarch64 CUDA release bundle from producer inputs",
            "Compose CUDA release bundle from producer inputs",
            "Compose ROCm release bundle from producer inputs",
            "Compose Vulkan release bundle from producer inputs",
        ):
            step_start = workflow.index(f"- name: {step_name}")
            env_start = workflow.index("        env:", step_start)
            self.assertIn("        shell: bash", workflow[step_start:env_start])

    def test_windows_composers_reuse_checksum_verified_host_verifier(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn(
            "Copy-Item target\\debug\\xtask.exe "
            "host-input\\release-attestation-verifier.exe -Force",
            workflow,
        )
        self.assertIn(
            "host-input\\release-attestation-verifier.exe.sha256",
            workflow,
        )
        self.assertEqual(
            workflow.count("MESH_RELEASE_ATTESTATION_VERIFIER:"),
            2,
        )

    def test_unix_composition_restores_downloaded_host_executable_bit(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        readiness_command = (
            "scripts/ci-client-readiness-smoke.sh "
            "host-input/mesh-llm runtime-root"
        )

        self.assertEqual(workflow.count(readiness_command), 6)
        self.assertEqual(
            workflow.count("chmod +x host-input/mesh-llm"),
            workflow.count(readiness_command),
        )

    def test_inference_smoke_consumes_composed_product(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("release-linux-inference-product"),
            2,
        )
        self.assertNotIn("release-linux-inference-binary", workflow)
        for required_path in (
            "smoke-input/mesh-llm",
            "smoke-input/host-imports.json",
            'smoke-input/native-runtimes/$runtime_name',
            "smoke-input/product-manifest.json",
        ):
            self.assertIn(required_path, workflow)

        self.assertIn(
            "python3 scripts/compose-product-bundle.py",
            workflow,
        )

    @staticmethod
    def job_block(workflow: str, start_job: str, next_job: str) -> str:
        start = workflow.index(f"  {start_job}:")
        end = workflow.index(f"  {next_job}:", start)
        return workflow[start:end]


if __name__ == "__main__":
    unittest.main()
