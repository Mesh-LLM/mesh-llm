from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "install.sh"


class InstallScriptTests(unittest.TestCase):
    def test_download_release_archive_prefers_platform_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            (assets_dir / platform_asset).write_text("platform\n", encoding="utf-8")
            (assets_dir / "native-runtimes.json").write_text("{}\n", encoding="utf-8")
            (assets_dir / "mesh-bundle.tar.gz").write_text("fallback\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                download_release_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"asset={platform_asset}", result.stdout)
            self.assertIn(f"archive={tmp_path / platform_asset}", result.stdout)

    def test_download_release_archive_falls_back_to_runtime_mesh_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            assets_dir = tmp_path / "assets"
            assets_dir.mkdir()
            platform_asset = "mesh-llm-aarch64-apple-darwin.tar.gz"
            (assets_dir / "native-runtimes.json").write_text("{}\n", encoding="utf-8")
            (assets_dir / "mesh-bundle.tar.gz").write_text("fallback\n", encoding="utf-8")

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{assets_dir}/%s\\n' "$1"
                }}
                download_release_archive "{tmp_path}" "{platform_asset}"
                printf 'asset=%s\\narchive=%s\\n' "$DOWNLOADED_ASSET" "$DOWNLOADED_ARCHIVE"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("asset=mesh-bundle.tar.gz", result.stdout)
            self.assertIn(f"archive={tmp_path / 'mesh-bundle.tar.gz'}", result.stdout)
            self.assertIn("Using runtime-enabled mesh bundle", result.stdout)

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

    def test_missing_native_runtime_manifest_is_silent_and_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            calls = tmp_path / "calls.log"
            self._write_fake_mesh_llm(
                install_dir / "mesh-llm",
                f"""
                if [[ "$*" == "runtime install --help" ]]; then
                    exit 0
                fi
                echo "$*" >> {calls}
                exit 0
                """,
            )

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{tmp_path}/missing-native-runtimes.json\\n'
                }}
                install_recommended_native_runtime "{tmp_path}"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            self.assertFalse(calls.exists())

    def test_old_binary_without_runtime_command_skips_manifest_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            release_url_calls = tmp_path / "release-url-calls.log"
            self._write_fake_mesh_llm(
                install_dir / "mesh-llm",
                """
                exit 2
                """,
            )

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    echo called >> {release_url_calls}
                    return 1
                }}
                install_recommended_native_runtime "{tmp_path}"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(release_url_calls.exists())

    def test_runtime_capable_binary_installs_available_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            manifest = tmp_path / "native-runtimes-source.json"
            manifest.write_text('{"runtimes":[]}\n', encoding="utf-8")
            calls = tmp_path / "calls.log"
            self._write_fake_mesh_llm(
                install_dir / "mesh-llm",
                f"""
                if [[ "$*" == "runtime install --help" ]]; then
                    exit 0
                fi
                echo "$*" >> {calls}
                exit 0
                """,
            )

            result = self._run_helper(
                tmp_path,
                install_dir,
                f"""
                release_url() {{
                    printf 'file://{manifest}\\n'
                }}
                install_recommended_native_runtime "{tmp_path}"
                """,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"runtime install --manifest {tmp_path / 'native-runtimes.json'}",
                calls.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "runtime prune --active-only",
                calls.read_text(encoding="utf-8"),
            )

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
            INSTALL_DIR={install_dir}
            {body}
            """
        )
        return subprocess.run(
            ["bash", "-c", script],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def _write_fake_mesh_llm(self, path: Path, body: str) -> None:
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + textwrap.dedent(body),
            encoding="utf-8",
        )
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
