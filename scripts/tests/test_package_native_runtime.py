from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package-native-runtime.sh"


class PackageNativeRuntimeTests(unittest.TestCase):
    def test_cuda_flavor_uses_mesh_cuda_version_major(self) -> None:
        self.assertEqual(
            self.backend_flavor("cuda", mesh_cuda_version="13.1.2"),
            "cuda13",
        )

    def test_explicit_cuda_toolkit_major_wins(self) -> None:
        self.assertEqual(
            self.backend_flavor(
                "cuda",
                mesh_cuda_version="13.1.2",
                toolkit_major="12",
            ),
            "cuda12",
        )

    def test_cuda_flavor_defaults_to_cuda_12(self) -> None:
        self.assertEqual(self.backend_flavor("cuda"), "cuda12")

    def backend_flavor(
        self,
        backend: str,
        *,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
    ) -> str:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("backend_flavor()")
        end = script.index("build_backend()", start)
        helpers = script[start:end]
        env = os.environ.copy()
        env["BACKEND"] = backend
        for name, value in (
            ("MESH_CUDA_VERSION", mesh_cuda_version),
            ("MESH_LLM_CUDA_TOOLKIT_MAJOR", toolkit_major),
        ):
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        result = subprocess.run(
            ["bash", "-c", f"set -euo pipefail\n{helpers}\nbackend_flavor"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
