import json
import pathlib
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "generate-native-runtime-release-manifest.sh"
)


class GenerateNativeRuntimeReleaseManifestTests(unittest.TestCase):
    def test_generated_manifest_is_single_valid_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            package_dir = root / "meshllm-native-runtime-linux-aarch64-cpu"
            package_dir.mkdir()
            (package_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "id": "meshllm-native-runtime-linux-aarch64-cpu",
                            "mesh_version": "0.68.0",
                            "skippy_abi": "0.1.25",
                            "platform": {
                                "os": "linux",
                                "arch": "aarch64",
                                "target": "aarch64-unknown-linux-gnu",
                            },
                            "backend": {"kind": "cpu"},
                            "rank": 0,
                            "libraries": ["lib/libllama.so"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            archive = root / "meshllm-native-runtime-linux-aarch64-cpu.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(package_dir, arcname=package_dir.name)

            out = root / "native-runtimes.json"
            subprocess.run(
                [
                    str(SCRIPT),
                    "--tag",
                    "v0.68.0",
                    "--out",
                    str(out),
                    str(archive),
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            with out.open(encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["mesh_version"], "0.68.0")
            self.assertEqual(len(manifest["artifacts"]), 1)


if __name__ == "__main__":
    unittest.main()
