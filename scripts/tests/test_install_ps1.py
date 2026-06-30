from __future__ import annotations

import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Thread
import unittest
from typing import Final
import zipfile


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "install.ps1"
PWSH: Final = shutil.which("pwsh")
HOST_ASSET: Final = "mesh-llm-x86_64-pc-windows-msvc.zip"
FORBIDDEN_POLICY_STRINGS: Final = (
    "runtime install",
    "runtime prune",
    "New-Service",
    "sc.exe",
)


class InstallPs1StaticTests(unittest.TestCase):
    def test_script_removes_runtime_and_service_policy_strings(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        for forbidden in FORBIDDEN_POLICY_STRINGS:
            self.assertNotIn(forbidden, contents)

    def test_script_keeps_host_archive_and_setup_handoff_flags(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(HOST_ASSET, contents)
        self.assertIn("[switch]$NoSetup", contents)
        self.assertIn("Run this next:", contents)
        self.assertIn("mesh-llm.exe setup", contents)
        self.assertIn("Legacy compatibility flag", contents)
        self.assertNotIn("Get-RecommendedFlavor", contents)
        self.assertNotIn("Choose-Flavor", contents)
        self.assertNotIn("native-runtimes.json", contents)


@unittest.skipUnless(PWSH, "pwsh not installed")
class InstallPs1BehaviorTests(unittest.TestCase):
    def test_interactive_install_runs_setup_and_warns_for_legacy_flavor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, calls = self._run_install(
                tmp_path,
                interactive=True,
                args=["-Flavor", "cuda"],
            )

            self.assertEqual(result.returncode, 0, self._combined_output(result))
            self.assertEqual(self._read_calls(calls), ["--version", "setup"])
            self.assertIn("Installing Windows x64 host binary", result.stdout)
            self.assertIn("Ignoring legacy -Flavor 'cuda'", self._combined_output(result))

    def test_noninteractive_install_prints_setup_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, calls = self._run_install(tmp_path, interactive=False)

            self.assertEqual(result.returncode, 0, self._combined_output(result))
            self.assertEqual(self._read_calls(calls), ["--version"])
            self.assertIn("Run this next:", result.stdout)
            self.assertIn('mesh-llm.exe" setup', result.stdout)

    def test_no_setup_prints_command_without_running_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, calls = self._run_install(
                tmp_path,
                interactive=True,
                args=["-NoSetup"],
            )

            self.assertEqual(result.returncode, 0, self._combined_output(result))
            self.assertEqual(self._read_calls(calls), ["--version"])
            self.assertIn("Run this next:", result.stdout)
            self.assertIn('mesh-llm.exe" setup', result.stdout)

    def _run_install(
        self,
        tmp_path: Path,
        *,
        interactive: bool,
        args: list[str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        install_dir = tmp_path / "bin"
        install_dir.mkdir()
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        calls = tmp_path / "mesh-llm-calls.log"
        self._write_release_archive(assets_dir / HOST_ASSET, calls)

        with AssetServer(assets_dir) as server:
            env = os.environ.copy()
            env["MESH_LLM_INSTALL_INTERACTIVE"] = "1" if interactive else "0"
            env["MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS"] = "1"
            env["MESH_LLM_INSTALL_URL_BASE"] = server.base_url
            command = [
                PWSH,
                "-NoProfile",
                "-File",
                str(SCRIPT),
                "-InstallDir",
                str(install_dir),
                "-NoPathUpdate",
                *(args or []),
            ]
            result = subprocess.run(
                command,
                cwd=tmp_path,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        return result, calls

    def _write_release_archive(self, archive_path: Path, calls: Path) -> None:
        script_contents = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\n' \"$*\" >> {calls}\n"
        )
        digest = self._write_zip_with_executable(
            archive_path,
            member_name="mesh-bundle/mesh-llm.exe",
            contents=script_contents,
        )
        archive_path.with_name(f"{archive_path.name}.sha256").write_text(
            f"{digest}  {archive_path.name}\n",
            encoding="utf-8",
        )

    def _write_zip_with_executable(
        self,
        archive_path: Path,
        *,
        member_name: str,
        contents: str,
    ) -> str:
        with zipfile.ZipFile(archive_path, "w") as archive:
            info = zipfile.ZipInfo(member_name)
            info.external_attr = 0o755 << 16
            archive.writestr(info, contents)
        return hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def _combined_output(self, result: subprocess.CompletedProcess[str]) -> str:
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    def _read_calls(self, calls: Path) -> list[str]:
        if not calls.exists():
            return []
        return [line for line in calls.read_text(encoding="utf-8").splitlines() if line]


class AssetServer:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            functools.partial(SimpleHTTPRequestHandler, directory=str(root)),
        )
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        address = self._server.server_address
        host = str(address[0])
        port = address[1]
        return f"http://{host}:{port}"

    def __enter__(self) -> AssetServer:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()


if __name__ == "__main__":
    unittest.main()
