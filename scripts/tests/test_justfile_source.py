from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.tests.justfile_source import JustfileImportError, read_justfile_source


class JustfileSourceTests(unittest.TestCase):
    def test_flat_imports_are_resolved_relative_to_each_importing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "just" / "nested"
            nested.mkdir(parents=True)
            (root / "Justfile").write_text(
                "# root\n\nimport 'just/build.just'\n\ndefault: build\n",
                encoding="utf-8",
            )
            (root / "just" / "build.just").write_text(
                "# build\nimport 'nested/runtime.just'\n",
                encoding="utf-8",
            )
            (nested / "runtime.just").write_text(
                "build:\n    #!/usr/bin/env bash\n    true\n",
                encoding="utf-8",
            )

            source = read_justfile_source(root / "Justfile")

            self.assertEqual(
                source,
                "# root\n\n# build\nbuild:\n    #!/usr/bin/env bash\n    true\n\n"
                "default: build\n",
            )

    def test_missing_import_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            justfile = Path(directory) / "Justfile"
            justfile.write_text("import 'missing.just'\n", encoding="utf-8")

            with self.assertRaisesRegex(JustfileImportError, "missing import"):
                read_justfile_source(justfile)

    def test_import_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Justfile").write_text("import 'other.just'\n", encoding="utf-8")
            (root / "other.just").write_text("import 'Justfile'\n", encoding="utf-8")

            with self.assertRaisesRegex(JustfileImportError, "import cycle"):
                read_justfile_source(root / "Justfile")

    def test_non_canonical_import_cycle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "just").mkdir()
            (root / "Justfile").write_text("import 'just/build.just'\n", encoding="utf-8")
            (root / "just" / "build.just").write_text(
                "import '../Justfile'\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(JustfileImportError, "import cycle"):
                read_justfile_source(root / "Justfile")

    def test_namespaced_and_optional_imports_are_rejected(self) -> None:
        for directive in ("mod build\n", "import? 'build.just'\n"):
            with self.subTest(directive=directive), tempfile.TemporaryDirectory() as directory:
                justfile = Path(directory) / "Justfile"
                justfile.write_text(directive, encoding="utf-8")

                with self.assertRaisesRegex(JustfileImportError, "unsupported directive"):
                    read_justfile_source(justfile)

    def test_indented_shell_commands_are_not_treated_as_directives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            justfile = Path(directory) / "Justfile"
            justfile.write_text("build:\n    mod build\n", encoding="utf-8")

            self.assertEqual(read_justfile_source(justfile), "build:\n    mod build\n")


if __name__ == "__main__":
    unittest.main()
