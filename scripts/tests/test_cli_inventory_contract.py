from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
WEBSITE = ROOT / "website"


class CliInventoryContractTests(unittest.TestCase):
    def test_npm_build_and_dev_always_generate_inventory(self) -> None:
        package = json.loads((WEBSITE / "package.json").read_text(encoding="utf-8"))
        scripts = package["scripts"]

        self.assertEqual(scripts["generate:cli"], "node ./scripts/generate-cli-inventory.mjs")
        self.assertEqual(scripts["check:cli"], "node ./scripts/generate-cli-inventory.mjs --check")
        self.assertIn("npm run generate:cli", scripts["build"])
        self.assertIn("npm run generate:cli", scripts["dev"])
        self.assertEqual(scripts["test:cli-explorer"], "node ./scripts/test-cli-explorer.mjs")

    def test_d3_is_locked_and_copied_from_local_node_modules(self) -> None:
        package = json.loads((WEBSITE / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((WEBSITE / "package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual(package["dependencies"]["d3"], "^7.9.0")
        self.assertEqual(lock["packages"][""]["dependencies"]["d3"], "^7.9.0")
        self.assertEqual(lock["packages"]["node_modules/d3"]["version"], "7.9.0")

        eleventy = (WEBSITE / ".eleventy.js").read_text(encoding="utf-8")
        self.assertIn('"node_modules/d3/dist/d3.min.js"', eleventy)
        self.assertIn('"assets/d3.min.js"', eleventy)
        self.assertIn('addFilter("jsonScript"', eleventy)
        self.assertIn('"<": "\\\\u003C"', eleventy)
        self.assertIn('"\\u2028": "\\\\u2028"', eleventy)
        self.assertNotIn("jsdelivr", eleventy.lower())

    def test_generated_inventory_is_ignored_and_cleaned(self) -> None:
        gitignore = (WEBSITE / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("src/_data/cliInventory.json", gitignore)

        cleaner = (WEBSITE / "scripts/clean-generated-site.mjs").read_text(encoding="utf-8")
        self.assertIn('"website/src/_data/cliInventory.json"', cleaner)

    def test_generator_invokes_locked_rust_exporter_and_validates_schema(self) -> None:
        generator = (WEBSITE / "scripts/generate-cli-inventory.mjs").read_text(encoding="utf-8")
        for token in (
            '"run"',
            '"--locked"',
            '"--quiet"',
            '"-p"',
            '"mesh-llm-cli"',
            '"--bin"',
            '"mesh-llm-cli-inventory"',
            '"--"',
            '"--check"',
            "schemaVersion",
            "document.root",
            "duplicate node path",
        ):
            self.assertIn(token, generator)

    def test_cli_crate_owns_the_cli_domain(self) -> None:
        ownership = json.loads((ROOT / "ci/ownership.yml").read_text(encoding="utf-8"))
        cli_crates = {
            crate
            for rule in ownership["crate_rules"]
            if rule["domain"] == "cli"
            for crate in rule["crates"]
        }
        self.assertIn("mesh-llm-cli", cli_crates)

    def test_nested_cli_paths_select_inventory_validation(self) -> None:
        derive_outputs = (ROOT / ".github/actions/compute-changes/derive-outputs.sh").read_text(encoding="utf-8")
        self.assertIn("^crates/mesh-llm-cli/", derive_outputs)

        affected = subprocess.run(
            ["bash", str(ROOT / "scripts/affected-crates.sh"), "crates/mesh-llm-cli/src/parser/commands.rs"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        affected_payload = json.loads(affected.stdout)
        self.assertFalse(affected_payload["website_changed"])
        self.assertIn("mesh-llm-cli", affected_payload["affected"])

    def test_ci_runs_inventory_and_browser_contracts(self) -> None:
        quality = (ROOT / ".github/workflows/ci-quality-slice.yml").read_text(encoding="utf-8")
        web = (ROOT / ".github/workflows/ci-web-slice.yml").read_text(encoding="utf-8")
        pages = (ROOT / ".github/workflows/website-pages.yml").read_text(encoding="utf-8")

        self.assertIn("Verify generated CLI inventory is deterministic and current", quality)
        self.assertIn("just cli-inventory-check", quality)
        self.assertNotIn("Require public website docs update", quality)
        self.assertIn("npm run test:cli-explorer", web)
        self.assertIn("npm run test:cli-explorer", pages)


if __name__ == "__main__":
    unittest.main()
