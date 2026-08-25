import os
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE_SCRIPT = ROOT / "providers" / "apple" / "Packaging" / "package.sh"


class ApplePackageTests(unittest.TestCase):
    def test_rejects_coreai_reference_with_multiple_at_signs(self):
        environment = os.environ.copy()
        environment.update(
            {
                "MESH_APPLE_COREAI_MODEL_REF": (
                    "owner/repository@mutable@" + "a" * 40
                ),
                "MESH_APPLE_COREAI_MODEL_ID": "owner/repository",
                "MESH_APPLE_COREAI_MODEL_VERSION": "1.0.0",
            }
        )
        result = subprocess.run(
            [str(PACKAGE_SCRIPT)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("owner/repository@", result.stderr)


if __name__ == "__main__":
    unittest.main()
