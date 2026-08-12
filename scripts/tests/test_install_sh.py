from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import textwrap
import unittest
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "install.sh"


class InstallScriptTests(unittest.TestCase):
    def test_download_release_archive_prefers_platform_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            self._write_file_with_checksum(assets_dir / platform_asset, "platform\n")
            self._write_file_with_checksum(assets_dir / "mesh-bundle.tar.gz", "fallback\n")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                INSTALL_VERBOSE=1
                download_release_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"asset={platform_asset}", result.stdout)
            self.assertIn(f"archive={tmp_path / platform_asset}", result.stdout)

    def test_release_url_honors_test_asset_base_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()

            result = self._run_helper(
                tmp_path,
                install_dir,
                """
                RELEASE_URL_BASE=https://example.invalid/assets/
                release_url mesh-llm-aarch64-unknown-linux-gnu.tar.gz
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.strip(),
                "https://example.invalid/assets/mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
            )

    def test_release_target_helpers_keep_linux_aarch64_flavor_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()

            result = self._run_helper(
                tmp_path,
                install_dir,
                """
                export MESH_LLM_TEST_UNAME_S=Linux
                export MESH_LLM_TEST_UNAME_M=aarch64
                # No CUDA evidence of any kind on this host.
                detect_cuda_major() { printf '\\n'; }
                printf 'support=%s\\n' "$(platform_support_status)"
                printf 'flavors=%s\\n' "$(supported_flavors)"
                printf 'recommended=%s\\n' "$(recommended_flavor)"
                printf 'cpu=%s\\n' "$(asset_name cpu)"
                printf 'cuda=%s\\n' "$(asset_name cuda || true)"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("support=supported", result.stdout)
            self.assertIn("flavors=cuda cpu", result.stdout)
            self.assertIn("recommended=cpu", result.stdout)
            self.assertIn("cpu=mesh-llm-aarch64-unknown-linux-gnu.tar.gz", result.stdout)
            # With no CUDA evidence at all there is no publishable cuda asset name.
            # Releases stopped publishing the legacy bare -cuda archive when the
            # lanes split into -cuda-12/-cuda-13, so asset_name must refuse rather
            # than emit a name that always 404s.
            self.assertIn("cuda=\n", result.stdout)
            self.assertNotIn("mesh-llm-aarch64-unknown-linux-gnu-cuda.tar.gz", result.stdout)
            self.assertIn("could not determine a supported CUDA major version", result.stderr)
            self.assertIn("--flavor cpu", result.stderr)
            self.assertNotIn("--flavor vulkan", result.stderr)

    def test_asset_name_uses_detected_cuda_major_lane(self) -> None:
        for arch in ("aarch64", "x86_64"):
            with self.subTest(arch=arch):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    install_dir = tmp_path / "bin"
                    install_dir.mkdir()

                    result = self._run_helper(
                        tmp_path,
                        install_dir,
                        f"""
                        export MESH_LLM_TEST_UNAME_S=Linux
                        export MESH_LLM_TEST_UNAME_M={arch}
                        detect_cuda_major() {{ printf '13\\n'; }}
                        asset_name cuda
                        """,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout.strip(),
                        f"mesh-llm-{arch}-unknown-linux-gnu-cuda-13.tar.gz",
                    )

    def test_detect_cuda_major_falls_back_to_nvidia_smi_driver_version(self) -> None:
        # Inference-only hosts have a driver but no toolkit, so every toolkit
        # probe comes up empty and only the nvidia-smi header carries a version.
        cases = (
            ("CUDA Version: 13.0", "13"),
            ("CUDA Version: 12.4", "12"),
            # A driver newer than any lane we publish clamps to the newest lane.
            ("CUDA Version: 14.2", "13"),
            # A driver too old for any published lane yields nothing.
            ("CUDA Version: 11.8", ""),
        )
        for header, expected in cases:
            with self.subTest(header=header):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    install_dir = tmp_path / "bin"
                    install_dir.mkdir()
                    probe_root = tmp_path / "cuda-probe-root"
                    probe_root.mkdir()
                    wrappers = tmp_path / "wrappers"
                    wrappers.mkdir()
                    for absent in ("nvcc", "ldconfig"):
                        stub = wrappers / absent
                        stub.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
                        stub.chmod(0o755)
                    nvidia_smi = wrappers / "nvidia-smi"
                    nvidia_smi.write_text(
                        "#!/usr/bin/env bash\n"
                        f"printf '%s\\n' '| NVIDIA-SMI 595.84  Driver Version: 595.84  {header} |'\n",
                        encoding="utf-8",
                    )
                    nvidia_smi.chmod(0o755)

                    result = self._run_helper(
                        tmp_path,
                        install_dir,
                        f"""
                        export PATH={shlex_quote(str(wrappers))}:$PATH
                        export MESH_LLM_TEST_CUDA_PROBE_ROOT={shlex_quote(str(probe_root))}
                        detect_cuda_major
                        """,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_release_target_helpers_recommend_cuda_on_jetson_orin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()

            result = self._run_helper(
                tmp_path,
                install_dir,
                """
                export MESH_LLM_TEST_UNAME_S=Linux
                export MESH_LLM_TEST_UNAME_M=aarch64
                export MESH_LLM_TEST_TEGRA_MODEL='NVIDIA Jetson AGX Orin'
                recommended_flavor
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "cuda")

    def test_checksum_from_sidecar_does_not_depend_on_awk_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            wrappers = tmp_path / "wrappers"
            wrappers.mkdir()
            fake_awk = wrappers / "awk"
            fake_awk.write_text(
                "#!/usr/bin/env bash\n"
                "echo 'mawk interval expressions are unavailable' >&2\n"
                "exit 2\n",
                encoding="utf-8",
            )
            fake_awk.chmod(0o755)
            digest = "ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789"
            sidecar = tmp_path / "archive.tar.gz.sha256"
            sidecar.write_text(f"{digest}  archive.tar.gz\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                export PATH={shlex_quote(str(wrappers))}:$PATH
                checksum_from_sidecar {shlex_quote(str(sidecar))}
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), digest.lower())

    def test_detect_cuda_major_reads_ldconfig_libcudart_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            wrappers = tmp_path / "wrappers"
            wrappers.mkdir()
            probe_root = tmp_path / "cuda-probe-root"
            probe_root.mkdir()
            ldconfig = wrappers / "ldconfig"
            ldconfig.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' 'libcudart.so.12 (libc6,AArch64) => /usr/local/cuda/lib64/libcudart.so.12'\n",
                encoding="utf-8",
            )
            ldconfig.chmod(0o755)

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                export PATH={shlex_quote(str(wrappers))}:$PATH
                export MESH_LLM_TEST_CUDA_PROBE_ROOT={shlex_quote(str(probe_root))}
                detect_cuda_major
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "12")

    def test_release_target_helpers_report_unsupported_armv7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()

            result = self._run_helper(
                tmp_path,
                install_dir,
                """
                export MESH_LLM_TEST_UNAME_S=Linux
                export MESH_LLM_TEST_UNAME_M=armv7l
                printf 'support=%s\\n' "$(platform_support_status)"
                platform_error_message
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("support=recognized-unsupported", result.stdout)
            self.assertIn("Linux/arm", result.stdout)

    def test_download_release_archive_falls_back_to_mesh_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            self._write_file_with_checksum(assets_dir / "mesh-bundle.tar.gz", "fallback\n")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                INSTALL_VERBOSE=1
                download_release_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("asset=mesh-bundle.tar.gz", result.stdout)
            self.assertIn(f"archive={tmp_path / 'mesh-bundle.tar.gz'}", result.stdout)
            self.assertIn("Using runtime-enabled mesh bundle fallback", result.stdout)

    def test_download_release_archive_fails_without_old_or_new_release_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                download_release_archive "{tmp_path}" "mesh-llm-aarch64-apple-darwin.tar.gz"
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("could not download release archive", result.stderr)

    def test_install_bundle_preserves_existing_binary_when_replacement_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            installed_binary = install_dir / "mesh-llm"
            installed_binary.write_text("existing binary\n", encoding="utf-8")
            bundle_dir = tmp_path / "mesh-bundle"
            bundle_dir.mkdir()
            (bundle_dir / "README.txt").write_text("malformed bundle\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"install_bundle {shlex_quote(str(bundle_dir))}",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "did not contain an installable mesh-llm binary",
                result.stderr,
            )
            self.assertEqual(
                installed_binary.read_text(encoding="utf-8"),
                "existing binary\n",
            )

    def test_install_bundle_accepts_complete_legacy_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            bundle_dir = tmp_path / "legacy-mesh-bundle"
            bundle_dir.mkdir()
            host = bundle_dir / "mesh-llm"
            host.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' 'mesh-llm 0.74.0'\n",
                encoding="utf-8",
            )
            host.chmod(0o755)

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"install_bundle {shlex_quote(str(bundle_dir))}",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("mesh-llm 0.74.0", (install_dir / "mesh-llm").read_text(encoding="utf-8"))
            self.assertIn("supported legacy MeshLLM 0.74.0", result.stderr)

    def test_install_bundle_rejects_partial_composed_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            bundle_dir = tmp_path / "partial-mesh-bundle"
            bundle_dir.mkdir()
            host = bundle_dir / "mesh-llm"
            host.write_text("partial host\n", encoding="utf-8")
            host.chmod(0o755)
            (bundle_dir / "product-manifest.json").write_text("{}\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"install_bundle {shlex_quote(str(bundle_dir))}",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete composed native runtime bundle", result.stderr)

    def test_install_bundle_rejects_post_contract_legacy_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            bundle_dir = tmp_path / "broken-current-bundle"
            bundle_dir.mkdir()
            host = bundle_dir / "mesh-llm"
            host.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' 'mesh-llm 0.75.0'\n",
                encoding="utf-8",
            )
            host.chmod(0o755)

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"install_bundle {shlex_quote(str(bundle_dir))}",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires product-manifest.json", result.stderr)

    def test_install_bundle_replaces_existing_nonempty_runtime_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            existing_runtime = install_dir / "native-runtimes" / "old-runtime"
            existing_runtime.mkdir(parents=True)
            (existing_runtime / "old-library").write_text("old\n", encoding="utf-8")
            (install_dir / "mesh-llm").write_text("old host\n", encoding="utf-8")

            bundle_dir = self._write_installable_bundle(tmp_path, "new")
            result = self._run_helper(
                tmp_path,
                install_dir,
                f"install_bundle {shlex_quote(str(bundle_dir))}",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (install_dir / "mesh-llm").read_text(encoding="utf-8"),
                "new host\n",
            )
            self.assertTrue(
                (
                    install_dir
                    / "native-runtimes"
                    / "new-runtime"
                    / "lib"
                    / "libllama.so"
                ).is_file()
            )
            self.assertFalse(existing_runtime.exists())

    def test_install_bundle_rolls_back_when_a_staged_move_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            existing_runtime = install_dir / "native-runtimes" / "old-runtime"
            existing_runtime.mkdir(parents=True)
            (existing_runtime / "old-library").write_text("old runtime\n", encoding="utf-8")
            old_host = install_dir / "mesh-llm"
            old_host.write_text("old host\n", encoding="utf-8")
            old_host.chmod(0o755)
            old_manifest = install_dir / "product-manifest.json"
            old_manifest.write_text("old manifest\n", encoding="utf-8")

            bundle_dir = self._write_installable_bundle(tmp_path, "new")
            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                failed_runtime_move=0
                mv() {{
                    if [[ "$failed_runtime_move" == 0 && "$1" == *".mesh-llm-stage."*/native-runtimes ]]; then
                        failed_runtime_move=1
                        return 42
                    fi
                    command mv "$@"
                }}
                install_bundle {shlex_quote(str(bundle_dir))}
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(old_host.read_text(encoding="utf-8"), "old host\n")
            self.assertEqual(
                (existing_runtime / "old-library").read_text(encoding="utf-8"),
                "old runtime\n",
            )
            self.assertEqual(
                old_manifest.read_text(encoding="utf-8"),
                "old manifest\n",
            )

    def test_install_bundle_preserves_existing_install_when_staging_mktemp_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            old_host = install_dir / "mesh-llm"
            old_host.write_text("old host\n", encoding="utf-8")
            bundle_dir = self._write_installable_bundle(tmp_path, "new")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                mktemp() {{ return 41; }}
                install_bundle {shlex_quote(str(bundle_dir))}
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(old_host.read_text(encoding="utf-8"), "old host\n")
            self.assertEqual(list(install_dir.glob(".mesh-llm-*")), [])

    def test_install_bundle_rolls_back_when_terminated_during_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            existing_runtime = install_dir / "native-runtimes" / "old-runtime"
            existing_runtime.mkdir(parents=True)
            (existing_runtime / "old-library").write_text("old runtime\n", encoding="utf-8")
            old_host = install_dir / "mesh-llm"
            old_host.write_text("old host\n", encoding="utf-8")
            old_host.chmod(0o755)
            old_manifest = install_dir / "product-manifest.json"
            old_manifest.write_text("old manifest\n", encoding="utf-8")
            bundle_dir = self._write_installable_bundle(tmp_path, "new")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                sent_term=0
                mv() {{
                    if [[ "$sent_term" == 0 && "$1" == *".mesh-llm-stage."*/native-runtimes ]]; then
                        sent_term=1
                        kill -TERM "$$"
                        return 143
                    fi
                    command mv "$@"
                }}
                install_bundle {shlex_quote(str(bundle_dir))}
                """,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(old_host.read_text(encoding="utf-8"), "old host\n")
            self.assertEqual(
                (existing_runtime / "old-library").read_text(encoding="utf-8"),
                "old runtime\n",
            )
            self.assertEqual(old_manifest.read_text(encoding="utf-8"), "old manifest\n")
            self.assertEqual(list(install_dir.glob(".mesh-llm-*")), [])

    def test_main_runs_setup_interactively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8"), "setup\n")
            self.assertFalse(tools.exists())
            self.assertIn("↓ Fetching mesh-llm release...", result.stdout)
            self.assertIn("Installed mesh-llm to", result.stdout)
            self.assertNotIn("Release channel:", result.stdout)
            self.assertNotIn("Verified checksum:", result.stdout)
            self.assertNotIn("Running post-install setup:", result.stdout)

    def test_main_verbose_output_keeps_download_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(
                tmp,
                interactive=True,
                args=["--verbose"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8"), "setup --verbose\n")
            self.assertFalse(tools.exists())
            self.assertIn("Release channel: stable", result.stdout)
            self.assertIn("Verified checksum:", result.stdout)
            self.assertIn("Installed mesh-llm-aarch64-apple-darwin.tar.gz", result.stdout)
            self.assertIn("Running post-install setup:", result.stdout)
            self.assertNotIn("↓ Fetching mesh-llm release...", result.stdout)

    def test_main_env_verbose_forwards_to_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(
                tmp,
                interactive=True,
                extra_exports="INSTALL_VERBOSE=1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8"), "setup --verbose\n")
            self.assertFalse(tools.exists())
            self.assertIn("Release channel: stable", result.stdout)

    def test_main_prints_setup_command_when_noninteractive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(calls.exists())
            self.assertFalse(tools.exists())
            self.assertIn("↓ Fetching mesh-llm release...", result.stdout)
            self.assertIn("Run this next:", result.stdout)
            self.assertIn("/mesh-llm setup", result.stdout)
            self.assertNotIn("Release channel:", result.stdout)

    def test_main_prints_setup_command_when_no_setup_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=True, args=["--no-setup"])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(calls.exists())
            self.assertFalse(tools.exists())
            self.assertIn("Run this next:", result.stdout)
            self.assertIn("/mesh-llm setup", result.stdout)

    def test_main_downloads_recommended_cuda_asset_on_jetson_orin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, _calls, _tools = self._run_main(
                tmp,
                interactive=False,
                args=["--no-setup"],
                os_name="Linux",
                arch="aarch64",
                tegra_model="NVIDIA Jetson AGX Orin",
                archive_name="mesh-llm-aarch64-unknown-linux-gnu-cuda-13.tar.gz",
                extra_exports="detect_cuda_major() { printf '13\\n'; }",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Installed mesh-llm to", result.stdout)
            self.assertNotIn("mesh-llm-aarch64-unknown-linux-gnu-cuda-13.tar.gz", result.stdout)

    def test_legacy_service_flags_pass_through_to_setup_without_shell_service_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(
                tmp,
                interactive=True,
                args=["--service", "--no-start-service"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8"), "setup --service\n")
            self.assertFalse(tools.exists())
            self.assertNotIn("runtime install", calls.read_text(encoding="utf-8"))
            self.assertNotIn("runtime prune", calls.read_text(encoding="utf-8"))
            self.assertIn("forwarding it to `mesh-llm setup --service`", result.stderr)

    def _run_main(
        self,
        tmp_dir: str,
        *,
        interactive: bool,
        args: list[str] | None = None,
        os_name: str = "Darwin",
        arch: str = "arm64",
        tegra_model: str = "",
        archive_name: str = "mesh-llm-aarch64-apple-darwin.tar.gz",
        extra_exports: str = "",
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        tmp_path = Path(tmp_dir)
        install_dir = tmp_path / "bin"
        install_dir.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        calls = tmp_path / "mesh-llm-calls.log"
        tools = tmp_path / "service-tools.log"
        archive_path = assets_dir / archive_name
        self._write_release_archive(archive_path, calls)
        wrappers = self._write_service_wrappers(tmp_path / "wrappers", tools)
        joined_args = " ".join(shlex_quote(arg) for arg in (args or []))
        result = self._run_helper(
            tmp_path,
            install_dir,
            f"""
            export PATH={wrappers}:$PATH
            release_url() {{
                printf 'file://{assets_dir}/%s\\n' "$1"
            }}
            {extra_exports}
            export MESH_LLM_TEST_INTERACTIVE={'1' if interactive else '0'}
            export MESH_LLM_TEST_UNAME_S={shlex_quote(os_name)}
            export MESH_LLM_TEST_UNAME_M={shlex_quote(arch)}
            export MESH_LLM_TEST_TEGRA_MODEL={shlex_quote(tegra_model)}
            main --install-dir {shlex_quote(str(install_dir))} {joined_args}
            """,
        )
        return result, calls, tools

    def _run_helper(
        self,
        tmp_path: Path,
        install_dir: Path,
        body: str,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["INSTALL_DIR"] = str(install_dir)
        script = textwrap.dedent(
            f"""
            set -euo pipefail
            source {SCRIPT}
            INSTALL_DIR={shlex_quote(str(install_dir))}
            {body}
            """,
        )
        return subprocess.run(
            ["bash", "-c", script],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _write_file_with_checksum(self, path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8")
        digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        path.with_name(f"{path.name}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")

    def _write_installable_bundle(self, root: Path, marker: str) -> Path:
        bundle = root / f"{marker}-mesh-bundle"
        runtime = bundle / "native-runtimes" / f"{marker}-runtime" / "lib"
        runtime.mkdir(parents=True)
        mesh_llm = bundle / "mesh-llm"
        mesh_llm.write_text(f"{marker} host\n", encoding="utf-8")
        mesh_llm.chmod(0o755)
        (bundle / "product-manifest.json").write_text(
            f"{marker} manifest\n",
            encoding="utf-8",
        )
        (runtime.parent / "manifest.json").write_text(
            f"{marker} runtime manifest\n",
            encoding="utf-8",
        )
        (runtime / "libllama.so").write_text(
            f"{marker} runtime\n",
            encoding="utf-8",
        )
        return bundle

    def _write_release_archive(self, archive_path: Path, calls: Path) -> None:
        with tempfile.TemporaryDirectory() as bundle_tmp:
            bundle_root = Path(bundle_tmp) / "mesh-bundle"
            bundle_root.mkdir()
            mesh_llm = bundle_root / "mesh-llm"
            mesh_llm.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"printf '%s\\n' \"$*\" >> {calls}\n",
                encoding="utf-8",
            )
            mesh_llm.chmod(0o755)
            (bundle_root / "product-manifest.json").write_text("{}\n", encoding="utf-8")
            runtime = bundle_root / "native-runtimes" / "test-runtime"
            (runtime / "lib").mkdir(parents=True)
            (runtime / "manifest.json").write_text("{}\n", encoding="utf-8")
            (runtime / "lib" / "libllama.so").write_bytes(b"runtime")
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(bundle_root, arcname="mesh-bundle")
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        archive_path.with_name(f"{archive_path.name}.sha256").write_text(
            f"{digest}  {archive_path.name}\n",
            encoding="utf-8",
        )

    def _write_service_wrappers(self, directory: Path, log_path: Path) -> str:
        directory.mkdir()
        for name in ("systemctl", "launchctl"):
            script_path = directory / name
            script_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"echo {name} >> {log_path}\n"
                "exit 0\n",
                encoding="utf-8",
            )
            script_path.chmod(0o755)
        return str(directory)


def shlex_quote(value: str) -> str:
    return subprocess.list2cmdline([value]) if os.name == "nt" else shlex.quote(value)


if __name__ == "__main__":
    unittest.main()
