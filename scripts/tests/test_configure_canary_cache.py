import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "configure-canary-cache.py"
SPEC = importlib.util.spec_from_file_location("canary_cache", SCRIPT)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class CanaryCacheTests(unittest.TestCase):
    def test_machine_home_wins_over_stale_legacy_service_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "mounted cache"
            (home / "hub").mkdir(parents=True)
            result = CACHE.configuration({"HF_HOME": str(home), "HF_CACHE": "/obsolete"})
            self.assertEqual(result["HF_CACHE"], str(home))
            self.assertEqual(result["HF_HUB_CACHE"], str(home / "hub"))
            self.assertEqual(result["HF_HUB_OFFLINE"], "1")

    def test_legacy_and_xdg_cache_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "huggingface"
            (home / "hub").mkdir(parents=True)
            for env in ({"HF_CACHE": str(home)}, {"XDG_CACHE_HOME": directory}):
                with self.subTest(env=env):
                    self.assertEqual(CACHE.configuration(env)["HF_HOME"], str(home))

    def test_missing_mount_fails_without_creating_cache_or_exporting_token(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "absent"
            output = Path(directory) / "github-env"
            env = {**os.environ, "HF_HOME": str(home), "HF_HUB_CACHE": str(home / "hub"),
                   "HF_TOKEN": "test-private-value", "GITHUB_ENV": str(output)}
            result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("check the runner mount/configuration", result.stderr)
            self.assertNotIn(env["HF_TOKEN"], result.stdout + result.stderr)
            self.assertFalse(home.exists())
            self.assertFalse(output.exists())

    def test_conflicting_hub_override_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "HF_HUB_CACHE"):
                CACHE.configuration({"HF_HOME": directory, "HF_HUB_CACHE": directory + "/other"})

    def test_exports_existing_credentials_with_mask_and_offline_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "hub").mkdir()
            output = home / "github-env"
            env = {**os.environ, "HF_HOME": directory, "HF_HUB_CACHE": str(home / "hub"),
                   "HF_TOKEN": "test-private-value", "HF_TOKEN_PATH": str(home / "token"),
                   "HF_HUB_OFFLINE": "0", "GITHUB_ENV": str(output)}
            result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines()[0], "::add-mask::test-private-value")
            values = dict(line.split("=", 1) for line in output.read_text().splitlines())
            self.assertEqual(values["HF_TOKEN"], env["HF_TOKEN"])
            self.assertEqual(values["HF_TOKEN_PATH"], env["HF_TOKEN_PATH"])
            self.assertEqual(values["HF_HUB_OFFLINE"], "1")
            self.assertFalse((home / "token").exists())

    def test_multiline_credentials_cannot_inject_actions_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "hub").mkdir()
            with self.assertRaisesRegex(ValueError, "invalid multiline value for HF_TOKEN"):
                CACHE.configuration({"HF_HOME": directory, "HF_TOKEN": "hidden\nINJECT=1"})


if __name__ == "__main__":
    unittest.main()
