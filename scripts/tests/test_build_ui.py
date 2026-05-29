from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-ui.sh"


class BuildUiScriptTests(unittest.TestCase):
    def test_mixed_case_release_profile_uses_release_ui_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui_dir = Path(tmp) / "ui"
            self._write_up_to_date_ui_fixture(ui_dir, profile="release", debug_ui="false")

            env = os.environ.copy()
            env["MESH_LLM_BUILD_PROFILE"] = "ReLeAsE"
            for key in ("VITE_BASE_PATH", "VITE_ROUTER_BASE_PATH", "VITE_STORAGE_NAMESPACE"):
                env.pop(key, None)

            result = subprocess.run(
                ["bash", str(SCRIPT), str(ui_dir)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profile: release", result.stdout)
        self.assertIn("debug UI: false", result.stdout)

    def _write_up_to_date_ui_fixture(self, ui_dir: Path, *, profile: str, debug_ui: str) -> None:
        ui_dir.mkdir()
        for relative in (
            "package.json",
            "pnpm-lock.yaml",
            "vite.config.ts",
            "tsconfig.json",
            "tsconfig.app.json",
            "tsconfig.node.json",
            "biome.json",
            "index.html",
        ):
            (ui_dir / relative).write_text("{}\n", encoding="utf-8")
        (ui_dir / "src").mkdir()
        (ui_dir / "public").mkdir()

        dist_dir = ui_dir / "dist"
        dist_dir.mkdir()
        (dist_dir / "asset.js").write_text("// built\n", encoding="utf-8")
        (dist_dir / ".mesh-llm-ui-build-env").write_text(
            f"MESH_LLM_BUILD_PROFILE={profile}\n"
            f"VITE_MESH_LLM_DEBUG_UI={debug_ui}\n"
            "VITE_BASE_PATH=\n"
            "VITE_ROUTER_BASE_PATH=\n"
            "VITE_STORAGE_NAMESPACE=\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
