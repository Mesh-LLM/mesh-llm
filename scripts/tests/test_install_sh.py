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
