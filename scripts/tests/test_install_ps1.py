from __future__ import annotations

import functools
import hashlib
import json
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

    def test_architecture_probe_handles_windows_powershell_null(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("::OSArchitecture.ToString()", contents)
        self.assertIn("$env:PROCESSOR_ARCHITECTURE", contents)

    def test_script_validates_product_bundle_before_mutation(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("Assert-ProductBundle", contents)
        self.assertIn("mesh-llm-product-v2", contents)
        self.assertIn("runtime.manifest_sha256", contents)
        self.assertIn("Get-DeterministicTreeSha256", contents)
        self.assertIn("Assert-SafeRelativePath", contents)
        self.assertIn('$ComposedProductMinVersion = [System.Version]::Parse("0.75.0")', contents)
        self.assertIn("Installing supported legacy MeshLLM", contents)
        self.assertIn("requires product-manifest.json and native-runtimes", contents)

    def test_tree_digest_normalizes_platform_path_separators(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("[System.IO.Path]::DirectorySeparatorChar", contents)
        self.assertIn("[System.IO.Path]::AltDirectorySeparatorChar", contents)
        self.assertIn("TrimStart($pathSeparators)", contents)

    def test_script_stages_replacement_and_removes_stale_host_imports(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("mesh-llm.exe.incoming", contents)
        self.assertIn("native-runtimes.incoming", contents)
        self.assertIn("Stage-IncomingBundle", contents)
        self.assertIn("Restore-InstallBackup", contents)
        self.assertIn("host-imports.json.backup", contents)
        self.assertIn("Remove-Item $hostImportsDestination -Force", contents)

    def test_script_stages_before_mutation_and_cleans_failed_staging(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        install_start = contents.index("function Install-MeshBinary")
        install_body = contents[install_start:]
        product_install_body = install_body[install_body.index("$paths = [PSCustomObject]") :]

        self.assertLess(
            product_install_body.index("Stage-IncomingBundle"),
            product_install_body.index("Move-IfExists"),
        )
        self.assertLess(
            product_install_body.index("Stage-IncomingBundle"),
            product_install_body.index("Remove-StaleBinaries"),
        )
        self.assertLess(
            product_install_body.index("Remove-InstallStagingPath -Path $paths.MeshBinaryStaging"),
            product_install_body.index("Restore-InstallBackup"),
        )
        self.assertLess(
            install_body.index("Remove-InstallBackups -Paths $paths"),
            install_body.index("Remove-StaleBinaries"),
        )


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
            self.assertIn("Installing Windows x64 MeshLLM product bundle", result.stdout)
            self.assertIn("Ignoring legacy -Flavor 'cuda'", self._combined_output(result))

    def test_noninteractive_install_prints_setup_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result, calls = self._run_install(tmp_path, interactive=False)

            self.assertEqual(result.returncode, 0, self._combined_output(result))
            self.assertEqual(self._read_calls(calls), ["--version"])
            self.assertIn("Run this next:", result.stdout)
            self.assertIn('mesh-llm.exe" setup', result.stdout)
            self.assertTrue(
                (tmp_path / "bin/native-runtimes/test-runtime/manifest.json").is_file()
            )

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

    def test_tampered_product_manifest_fails_before_existing_install_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            existing_binary = install_dir / "mesh-llm.exe"
            existing_binary.write_text("existing\n", encoding="utf-8")
            (install_dir / "product-manifest.json").write_text(
                '{"existing":true}\n', encoding="utf-8"
            )

            result, _calls = self._run_install(
                tmp_path,
                interactive=False,
                archive_options={"tamper_host_digest": True},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("host.sha256 mismatch", self._combined_output(result))
            self.assertEqual(existing_binary.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(
                (install_dir / "product-manifest.json").read_text(encoding="utf-8"),
                '{"existing":true}\n',
            )

    def test_stale_host_imports_is_removed_when_bundle_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            (install_dir / "host-imports.json").write_text("stale\n", encoding="utf-8")

            result, _calls = self._run_install(tmp_path, interactive=False)

            self.assertEqual(result.returncode, 0, self._combined_output(result))
            self.assertFalse((install_dir / "host-imports.json").exists())

    def test_failed_runtime_replacement_restores_previous_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            install_dir.mkdir()
            existing_binary = install_dir / "mesh-llm.exe"
            existing_binary.write_text("existing\n", encoding="utf-8")
            existing_runtime = install_dir / "native-runtimes" / "old-runtime"
            existing_runtime.mkdir(parents=True)
            (existing_runtime / "manifest.json").write_text("old-runtime\n", encoding="utf-8")
            (install_dir / "product-manifest.json").write_text("old-manifest\n", encoding="utf-8")

            result, _calls = self._run_install(
                tmp_path,
                interactive=False,
                extra_env={"MESH_LLM_INSTALL_TEST_FAIL_AFTER_RUNTIME_REPLACE": "1"},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(existing_binary.read_text(encoding="utf-8"), "existing\n")
            self.assertTrue((install_dir / "native-runtimes" / "old-runtime").is_dir())
            self.assertEqual(
                (install_dir / "product-manifest.json").read_text(encoding="utf-8"),
                "old-manifest\n",
            )

    def test_stale_cleanup_failure_does_not_roll_back_committed_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            install_dir = tmp_path / "bin"
            stale_binary = install_dir / "rpc-server.exe"
            stale_binary.mkdir(parents=True)
            (stale_binary / "child").write_text("prevent non-recursive removal\n", encoding="utf-8")

            result, _calls = self._run_install(tmp_path, interactive=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((install_dir / "mesh-llm.exe").is_file())
            self.assertTrue(
                (install_dir / "native-runtimes/test-runtime/manifest.json").is_file()
            )
            manifest = json.loads(
                (install_dir / "product-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["runtime"]["id"], "test-runtime")

    def _run_install(
        self,
        tmp_path: Path,
        *,
        interactive: bool,
        args: list[str] | None = None,
        archive_options: dict[str, bool] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        install_dir = tmp_path / "bin"
        install_dir.mkdir(exist_ok=True)
        assets_dir = tmp_path / "assets"
        assets_dir.mkdir()
        calls = tmp_path / "mesh-llm-calls.log"
        self._write_release_archive(
            assets_dir / HOST_ASSET,
            calls,
            **(archive_options or {}),
        )

        with AssetServer(assets_dir) as server:
            env = os.environ.copy()
            env["MESH_LLM_INSTALL_INTERACTIVE"] = "1" if interactive else "0"
            env["MESH_LLM_INSTALL_TEST_ALLOW_NONWINDOWS"] = "1"
            env["MESH_LLM_INSTALL_URL_BASE"] = server.base_url
            env.update(extra_env or {})
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

    def _write_release_archive(
        self,
        archive_path: Path,
        calls: Path,
        *,
        tamper_host_digest: bool = False,
        include_host_imports: bool = False,
    ) -> None:
        script_contents = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"printf '%s\\n' \"$*\" >> {calls}\n"
        )
        digest = self._write_zip_with_executable(
            archive_path,
            member_name="mesh-bundle/mesh-llm.exe",
            contents=script_contents,
            tamper_host_digest=tamper_host_digest,
            include_host_imports=include_host_imports,
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
        tamper_host_digest: bool,
        include_host_imports: bool,
    ) -> str:
        runtime_manifest = json.dumps(
            {
                "runtime": {
                    "id": "test-runtime",
                    "mesh_version": "0.73.1",
                    "skippy_abi": "0.1.0",
                    "platform": {
                        "os": "windows",
                        "arch": "x86_64",
                        "target": "x86_64-pc-windows-msvc",
                    },
                    "backend": {"kind": "cpu"},
                    "rank": 0,
                    "libraries": ["lib/llama.dll"],
                    "url": None,
                    "sha256": None,
                    "signature": None,
                }
            },
            sort_keys=True,
        ) + "\n"
        runtime_files = {
            "manifest.json": runtime_manifest.encode(),
            "lib/llama.dll": b"runtime",
            "Z-file": b"uppercase",
            "a-file": b"lowercase",
        }
        runtime_digest = self._tree_sha256(runtime_files)
        host_digest = hashlib.sha256(contents.encode()).hexdigest()
        if tamper_host_digest:
            host_digest = "0" * 64
        product_manifest = json.dumps(
            {
                "schema_version": 2,
                "contract": "mesh-llm-product-v2",
                "mesh_version": "0.73.1",
                "backend": "cpu",
                "host": {"path": "mesh-llm.exe", "sha256": host_digest},
                "runtime": {
                    "id": "test-runtime",
                    "path": "native-runtimes/test-runtime",
                    "sha256": runtime_digest,
                    "manifest_sha256": hashlib.sha256(
                        runtime_files["manifest.json"]
                    ).hexdigest(),
                },
            },
            sort_keys=True,
        ) + "\n"
        with zipfile.ZipFile(archive_path, "w") as archive:
            info = zipfile.ZipInfo(member_name)
            info.external_attr = 0o755 << 16
            archive.writestr(info, contents)
            archive.writestr("mesh-bundle/product-manifest.json", product_manifest)
            for relative_path, data in runtime_files.items():
                archive.writestr(
                    f"mesh-bundle/native-runtimes/test-runtime/{relative_path}", data
                )
            if include_host_imports:
                archive.writestr("mesh-bundle/host-imports.json", '{"imports":[]}\n')
        return hashlib.sha256(archive_path.read_bytes()).hexdigest()

    def _tree_sha256(self, files: dict[str, bytes]) -> str:
        digest = hashlib.sha256()
        for relative_path in sorted(files):
            relative = relative_path.encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(hashlib.sha256(files[relative_path]).digest())
        return digest.hexdigest()

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
