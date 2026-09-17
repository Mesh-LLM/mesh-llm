import importlib.util
import json
import tempfile
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compose-product-bundle.py"
SPEC = importlib.util.spec_from_file_location("compose_product_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPOSE_PRODUCT_BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSE_PRODUCT_BUNDLE)


class ComposeProductBundleTests(unittest.TestCase):
    def test_independent_runtime_release_requires_the_compiled_host_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = pathlib.Path(directory)
            host = bundle / "mesh-llm"
            contract = {
                "schema_version": 1, "product_version": "9.0.0",
                "runtime_release": "1.0.0", "skippy_abi": "0.1.57",
            }
            host.write_text("#!/bin/sh\nprintf '%s\\n' '" + json.dumps(contract) + "'\n")
            host.chmod(0o755)
            runtime = bundle / "native-runtimes" / "cpu"
            runtime.mkdir(parents=True)
            for abi in ("0.1.57", "0.1.58"):
                with self.subTest(abi=abi):
                    (runtime / "manifest.json").write_text(json.dumps({"runtime": {
                        "id": "cpu", "mesh_version": "2.0.0", "skippy_abi": abi,
                        "backend": {"kind": "cpu"},
                    }}))
                    if abi != contract["skippy_abi"]:
                        with self.assertRaisesRegex(ValueError, "host-required ABI"):
                            COMPOSE_PRODUCT_BUNDLE.compose_manifest(bundle, host, runtime, "9.0.0", "cpu")
                    else:
                        result = COMPOSE_PRODUCT_BUNDLE.compose_manifest(bundle, host, runtime, "9.0.0", "cpu")
                        self.assertEqual(result["mesh_version"], "9.0.0")
                        self.assertEqual(result["runtime"]["release_version"], "2.0.0")
                        self.assertEqual(result["runtime"]["skippy_abi"], "0.1.57")
                        self.assertEqual(result["host"]["required_skippy_abi"], "0.1.57")
                        with self.assertRaisesRegex(ValueError, "product version"):
                            COMPOSE_PRODUCT_BUNDLE.compose_manifest(bundle, host, runtime, "8.0.0", "cpu")

    def test_host_contract_rejects_unknown_schema_and_missing_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            host = pathlib.Path(directory) / "host"
            for contract, message in (
                ({"schema_version": 2}, "unsupported host build contract schema"),
                ({"schema_version": 1, "product_version": "1.0.0", "runtime_release": "2.0.0"}, "missing skippy_abi"),
            ):
                with self.subTest(contract=contract):
                    host.write_text("#!/bin/sh\nprintf '%s\\n' '" + json.dumps(contract) + "'\n")
                    host.chmod(0o755)
                    with self.assertRaisesRegex(ValueError, message):
                        COMPOSE_PRODUCT_BUNDLE.read_host_build_contract(host)

    def test_tree_hash_uses_ordinal_relative_path_order(self) -> None:
        class CaseInsensitivePath(type(pathlib.Path())):
            def __lt__(self, other: object) -> bool:
                if not isinstance(other, pathlib.PurePath):
                    return NotImplemented
                return str(self).lower() < str(other).lower()

        fixture = CaseInsensitivePath(ROOT / "scripts" / "tests" / "fixtures" / "tree-hash")
        files = {
            fixture / "README.md": b"upper sorts first ordinally\n",
            fixture / "lib" / "runtime.dll": b"lower sorts second ordinally\n",
        }
        for path, contents in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        try:
            self.assertEqual(
                COMPOSE_PRODUCT_BUNDLE.tree_sha256(fixture),
                "01df8a658501c6798530548aa7ca5a15ce02059d66b8ab87df4150811b55c7e1",
            )
        finally:
            for path in files:
                path.unlink()
            (fixture / "lib").rmdir()
            fixture.rmdir()


if __name__ == "__main__":
    unittest.main()
