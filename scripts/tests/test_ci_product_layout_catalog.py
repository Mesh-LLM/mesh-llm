"""Ownership admission before the physical Mesh/Skippy source relocation."""
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from unittest import mock

from scripts.tests.test_plan_ci import PLANNER, ROOT, fixture

COMPAT_SPEC = importlib.util.spec_from_file_location(
    "ci_cargo_packages", ROOT / "scripts" / "ci-cargo-packages.py"
)
COMPAT = importlib.util.module_from_spec(COMPAT_SPEC)
COMPAT_SPEC.loader.exec_module(COMPAT)


class ProductLayoutCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ownership = json.loads((ROOT / "ci/ownership.yml").read_text())

    def test_relocated_paths_preserve_existing_semantic_domains(self) -> None:
        paths = [
            ("crates/mesh-llm-cli/src/parser.rs", "mesh/crates/mesh-llm-cli/src/parser.rs"),
            ("crates/mesh-llm-ui/src/App.tsx", "mesh/crates/mesh-llm-ui/src/App.tsx"),
            ("crates/mesh-llm-protocol/src/lib.rs", "mesh/crates/mesh-llm-protocol/src/lib.rs"),
            ("crates/skippy-ffi/src/abi.rs", "skippy/crates/skippy-ffi/src/abi.rs"),
            ("crates/skippy-server/src/lib.rs", "skippy/crates/skippy-serving/src/lib.rs"),
            ("crates/model-hf/src/lib.rs", "skippy/crates/skippy-model-hf/src/lib.rs"),
            ("third_party/llama.cpp/patches/0001.patch", "skippy/third_party/llama.cpp/patches/0001.patch"),
            ("scripts/prepare-llama.sh", "skippy/scripts/prepare-llama.sh"),
            ("scripts/build-host.sh", "mesh/scripts/build-host.sh"),
            ("sdk/kotlin/build.gradle.kts", "mesh/sdk/kotlin/build.gradle.kts"),
            ("website/src/index.njk", "mesh/website/src/index.njk"),
            ("docs/skippy/CONFIG.md", "skippy/docs/CONFIG.md"),
            ("evals/parity.json", "skippy/evals/parity.json"),
        ]
        for old, new in paths:
            with self.subTest(path=new):
                self.assertEqual(
                    PLANNER._matched_domains(self.ownership, [old], []),
                    PLANNER._matched_domains(self.ownership, [new], []),
                )

    def test_future_mesh_extractions_select_product_paths_and_named_crates(self) -> None:
        for name in (
            "mesh-llm-skippy-adapter", "mesh-llm-membership",
            "mesh-llm-control-api", "mesh-llm-runtime",
        ):
            for prefix in ("crates/", "mesh/crates/"):
                with self.subTest(crate=name, prefix=prefix):
                    path_domains = PLANNER._matched_domains(
                        self.ownership, [f"{prefix}{name}/src/lib.rs"], []
                    )
                    crate_domains = PLANNER._matched_domains(
                        self.ownership, [], [name]
                    )
                    for domains in (path_domains, crate_domains):
                        self.assertIn("rust", domains)
                        self.assertIn("runtime-product", domains)

    def test_relocation_preserves_complete_runtime_plan_selection(self) -> None:
        old = fixture("runtime.json")
        new = copy.deepcopy(old)
        new["changed_files"] = ["mesh/" + path for path in old["changed_files"]]
        for package in new["workspace_packages"]:
            owner = "skippy" if package["name"].startswith("skippy-") else "mesh"
            package["path"] = owner + "/" + package["path"]
        with mock.patch.object(PLANNER.subprocess, "run", side_effect=AssertionError("unexpected command")):
            old_plan = PLANNER.build_plan(old, root=ROOT)
            new_plan = PLANNER.build_plan(new, root=ROOT)
        for key in ("domains", "required_slices", "direct_crates", "affected_crates", "matrices", "signals"):
            with self.subTest(field=key):
                self.assertEqual(old_plan[key], new_plan[key])

    def test_renamed_crates_keep_their_direct_semantic_domains(self) -> None:
        for old, new in (
            ("openai-frontend", "skippy-openai-frontend"),
            ("skippy-server", "skippy-serving"),
            ("skippy-model-package", "skippy-package-builder"),
            ("model-hf", "skippy-model-hf"),
            ("model-artifact", "skippy-model-artifact"),
            ("model-ref", "skippy-model-ref"),
            ("model-resolver", "skippy-model-resolver"),
            ("mesh-llm-gpu-bench", "skippy-gpu-bench"),
        ):
            with self.subTest(crate=new):
                self.assertEqual(
                    PLANNER._matched_domains(self.ownership, [], [old]),
                    PLANNER._matched_domains(self.ownership, [], [new]),
                )

    def test_reused_package_name_preserves_main_and_routes_relocated_source(self) -> None:
        # Current main calls its package builder skippy-model-package; the
        # extraction reuses that name for acquisition. Keep the existing direct
        # rule until the follow-up catalog cleanup and add download ownership
        # only through the new product path. This conservatively selects both.
        self.assertEqual(
            PLANNER._matched_domains(self.ownership, [], ["skippy-model-package"]),
            ["rust", "split-serving"],
        )
        self.assertEqual(
            PLANNER._matched_domains(
                self.ownership,
                ["skippy/crates/skippy-model-package/src/lib.rs"],
                ["skippy-model-package"],
            ),
            ["rust", "split-serving", "model-download"],
        )

    def test_executor_successor_map_preserves_planner_semantic_coverage(self) -> None:
        """Every translated successor keeps the domains of the owner it replaces.

        ``scripts/ci-cargo-packages.py`` rewrites a protected batch into the
        extracted package names. Translation is only sound while the planner
        catalog routes a successor the way it routes the predecessor, because a
        crate that matches no direct rule falls through to generic ``rust``
        routing and its runtime-product/artifact/smoke consumers are skipped.

        The three exceptions below are the deferred catalog migration: those
        successors have no ``crate_rules`` entry yet, and ``ci/ownership.yml``
        can only change on the protected branch (the PR plan step requires a PR's
        catalogs to match the protected copies byte for byte). This test locks
        that exact set, so adding a new successor or closing a gap without
        emptying this list fails until the catalog catches up. The successor may
        only ever lose coverage here, never gain it.
        """
        reused_package_name = {("model-package", "skippy-model-package")}
        observed: dict[str, tuple[str, ...]] = {}
        for predecessor, successors in COMPAT.SUCCESSORS.items():
            expected = set(PLANNER._matched_domains(self.ownership, [], [predecessor]))
            for successor in successors:
                if (predecessor, successor) in reused_package_name:
                    # Documented separately: the extracted tree reuses the
                    # package-builder name for acquisition.
                    continue
                actual = set(PLANNER._matched_domains(self.ownership, [], [successor]))
                if actual == expected:
                    continue
                self.assertTrue(
                    actual < expected,
                    f"{successor} must not gain domains over {predecessor}: {sorted(actual)}",
                )
                observed[successor] = tuple(sorted(expected - actual))
        self.assertEqual(
            observed,
            {
                "skippy-hf-hub": ("model-download",),
                "skippy-api": ("split-serving",),
                "skippy-events": ("split-serving",),
            },
        )


    def test_unknown_product_paths_are_not_admitted_by_a_catch_all(self) -> None:
        for path in ("mesh/unowned/payload.bin", "skippy/unowned/payload.bin", "shared/crates/demo/src/lib.rs"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(PLANNER.PlanError, "ownership has no rule"):
                    PLANNER._matched_domains(self.ownership, [path], [])


if __name__ == "__main__":
    unittest.main()
