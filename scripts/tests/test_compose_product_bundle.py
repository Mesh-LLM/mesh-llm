import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "compose-product-bundle.py"
SPEC = importlib.util.spec_from_file_location("compose_product_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COMPOSE_PRODUCT_BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPOSE_PRODUCT_BUNDLE)


class ComposeProductBundleTests(unittest.TestCase):
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
