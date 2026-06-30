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

    def test_main_runs_setup_interactively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=True)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8"), "setup\n")
            self.assertFalse(tools.exists())

    def test_main_prints_setup_command_when_noninteractive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=False)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(calls.exists())
            self.assertFalse(tools.exists())
            self.assertIn("Run this next:", result.stdout)
            self.assertIn("/mesh-llm setup", result.stdout)

    def test_main_prints_setup_command_when_no_setup_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result, calls, tools = self._run_main(tmp, interactive=True, args=["--no-setup"])

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(calls.exists())
            self.assertFalse(tools.exists())
            self.assertIn("Run this next:", result.stdout)
            self.assertIn("/mesh-llm setup", result.stdout)

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
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        tmp_path = Path(tmp_dir)
        install_dir = tmp_path / "bin"
        install_dir.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        calls = tmp_path / "mesh-llm-calls.log"
        tools = tmp_path / "service-tools.log"
        archive_path = assets_dir / "mesh-llm-aarch64-apple-darwin.tar.gz"
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
            export MESH_LLM_TEST_INTERACTIVE={'1' if interactive else '0'}
            export MESH_LLM_TEST_UNAME_S=Darwin
            export MESH_LLM_TEST_UNAME_M=arm64
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
