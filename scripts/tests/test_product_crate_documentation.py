from __future__ import annotations

from pathlib import Path
import re
import tempfile
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)")


def _local_link_destinations(readme: Path):
    fence = None
    for line_number, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in ("```", "~~~"):
            fence = None if fence == marker else marker
            continue
        if fence is not None:
            continue
        for match in LINK.finditer(line):
            destination = match.group(1).strip("<>")
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            yield line_number, unquote(parsed.path)


def check_product_crates(root: Path) -> list[str]:
    """Check the shared documentation contract for Mesh and Skippy crates."""
    errors = []
    root = root.resolve()
    for product in ("mesh", "skippy"):
        crates_root = root / product / "crates"
        if not crates_root.is_dir():
            errors.append(f"{crates_root}: missing product crates directory")
            continue
        for manifest in sorted(crates_root.glob("*/Cargo.toml")):
            crate_dir = manifest.parent
            try:
                package = tomllib.loads(manifest.read_text(encoding="utf-8"))["package"]
            except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
                errors.append(f"{manifest}: cannot read package metadata: {exc}")
                continue
            if not isinstance(package.get("description"), str) or not package["description"].strip():
                errors.append(f"{manifest}: missing non-empty package description")
            if package.get("readme", "README.md") != "README.md":
                errors.append(f"{manifest}: package readme must be README.md")

            readme = crate_dir / "README.md"
            if not readme.is_file():
                errors.append(f"{readme}: missing crate README")
                continue
            for line_number, destination in _local_link_destinations(readme):
                target = (root / destination.lstrip("/") if destination.startswith("/") else crate_dir / destination).resolve()
                if not target.is_relative_to(root):
                    errors.append(f"{readme}:{line_number}: local link leaves repository: {destination}")
                elif not target.exists():
                    errors.append(f"{readme}:{line_number}: broken local link: {destination}")
    return errors


class ProductCrateDocumentationTests(unittest.TestCase):
    def test_workspace_product_crates(self) -> None:
        self.assertEqual(check_product_crates(ROOT), [])

    def test_reports_missing_description_readme_and_local_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for product in ("mesh", "skippy"):
                (root / product / "crates").mkdir(parents=True)
            no_readme = root / "mesh/crates/no-readme"
            no_readme.mkdir()
            (no_readme / "Cargo.toml").write_text('[package]\nname = "no-readme"\n')
            broken_link = root / "skippy/crates/broken-link"
            broken_link.mkdir()
            (broken_link / "Cargo.toml").write_text(
                '[package]\nname = "broken-link"\ndescription = "A fixture"\n'
            )
            (broken_link / "README.md").write_text("# Fixture\n[missing](../absent/README.md)\n")

            errors = "\n".join(check_product_crates(root))
            self.assertIn("missing non-empty package description", errors)
            self.assertIn("missing crate README", errors)
            self.assertIn("broken local link", errors)

    def test_accepts_existing_relative_links_and_ignores_external_or_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for product in ("mesh", "skippy"):
                (root / product / "crates").mkdir(parents=True)
            crate = root / "skippy/crates/example"
            crate.mkdir()
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "example"\ndescription = "A fixture"\n'
            )
            (crate / "README.md").write_text(
                "# Example\n[relative](Cargo.toml#package) "
                "[external](https://example.com/missing) [section](#example)\n"
                "```md\n[illustration](not-a-file.md)\n```\n"
            )

            self.assertEqual(check_product_crates(root), [])


if __name__ == "__main__":
    unittest.main()
