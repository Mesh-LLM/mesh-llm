#!/usr/bin/env python3
"""Structural ownership checks for the logging modules introduced by PR #1175."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_NEW_FILE_LINES = 999


class LoggingModuleBoundariesTest(unittest.TestCase):
    def source(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def assert_owner_module(
        self,
        parent: str,
        module_name: str,
        child: str,
        *,
        forbidden_parent_text: tuple[str, ...] = (),
    ) -> None:
        parent_source = self.source(parent)
        child_path = ROOT / child
        self.assertTrue(child_path.is_file(), f"missing owner module: {child}")
        self.assertIn(
            f"mod {module_name};",
            parent_source,
            f"{parent} must declare its {module_name} owner module",
        )
        for text in forbidden_parent_text:
            self.assertFalse(
                text in parent_source,
                f"{text!r} still lives in {parent}",
            )
        self.assertLessEqual(
            len(child_path.read_text(encoding="utf-8").splitlines()),
            MAX_NEW_FILE_LINES,
            f"new owner module exceeds {MAX_NEW_FILE_LINES} lines: {child}",
        )

    def test_production_logging_responsibilities_have_named_owners(self) -> None:
        test_extractions = (
            (
                "crates/mesh-llm-host-runtime/src/logging/webhook_delivery.rs",
                "crates/mesh-llm-host-runtime/src/logging/webhook_delivery/tests.rs",
            ),
            (
                "crates/mesh-llm-host-runtime/src/logging/cleanup.rs",
                "crates/mesh-llm-host-runtime/src/logging/cleanup/tests.rs",
            ),
            (
                "crates/mesh-llm-host-runtime/src/logging/raw_mesh_lifecycle.rs",
                "crates/mesh-llm-host-runtime/src/logging/raw_mesh_lifecycle/tests.rs",
            ),
            (
                "crates/mesh-llm-host-runtime/src/runtime/operational_logging.rs",
                "crates/mesh-llm-host-runtime/src/runtime/operational_logging/tests.rs",
            ),
            (
                "crates/mesh-llm-host-runtime/src/api/routes/logs/mod.rs",
                "crates/mesh-llm-host-runtime/src/api/routes/logs/tests.rs",
            ),
            (
                "crates/mesh-llm-host-runtime/src/api/routes/logs/events/session.rs",
                "crates/mesh-llm-host-runtime/src/api/routes/logs/events/session/tests.rs",
            ),
        )
        for parent, child in test_extractions:
            with self.subTest(parent=parent):
                self.assert_owner_module(
                    parent,
                    "tests",
                    child,
                    forbidden_parent_text=("mod tests {",),
                )

        self.assert_owner_module(
            "crates/mesh-llm-host-runtime/src/logging/runtime_state.rs",
            "query_facade",
            "crates/mesh-llm-host-runtime/src/logging/runtime_state/query_facade.rs",
            forbidden_parent_text=("struct LoggingQueryFacade",),
        )
        self.assert_owner_module(
            "crates/mesh-llm-host-runtime/src/logging/runtime_state.rs",
            "workers",
            "crates/mesh-llm-host-runtime/src/logging/runtime_state/workers.rs",
            forbidden_parent_text=("fn start_persistence_worker",),
        )
        self.assert_owner_module(
            "crates/mesh-llm-host-runtime/src/logging/runtime_state.rs",
            "tests",
            "crates/mesh-llm-host-runtime/src/logging/runtime_state/tests.rs",
            forbidden_parent_text=("mod tests {",),
        )
        self.assert_owner_module(
            "crates/mesh-llm-log-store/src/maintenance.rs",
            "execution",
            "crates/mesh-llm-log-store/src/maintenance/execution.rs",
            forbidden_parent_text=("impl ArtifactFileStore",),
        )

        for parent in (
            "crates/mesh-llm-host-runtime/src/logging/webhook_delivery.rs",
            "crates/mesh-llm-host-runtime/src/logging/cleanup.rs",
            "crates/mesh-llm-host-runtime/src/logging/raw_mesh_lifecycle.rs",
            "crates/mesh-llm-host-runtime/src/runtime/operational_logging.rs",
            "crates/mesh-llm-host-runtime/src/api/routes/logs/mod.rs",
            "crates/mesh-llm-host-runtime/src/api/routes/logs/events/session.rs",
            "crates/mesh-llm-host-runtime/src/logging/runtime_state.rs",
            "crates/mesh-llm-log-store/src/maintenance.rs",
        ):
            with self.subTest(coherent_parent=parent):
                self.assertLessEqual(
                    len(self.source(parent).splitlines()),
                    MAX_NEW_FILE_LINES,
                    f"new production owner remains oversized: {parent}",
                )

    def test_oversized_characterization_suites_are_split_by_concern(self) -> None:
        suites = (
            (
                "crates/mesh-llm-host-runtime/src/api/tests/logs_api_routes.rs",
                (
                    ("access_and_mutation", "access_and_mutation.rs"),
                    ("read_and_export", "read_and_export.rs"),
                    ("event_stream", "event_stream.rs"),
                ),
                "crates/mesh-llm-host-runtime/src/api/tests/logs_api_routes",
            ),
            (
                "crates/mesh-llm-log-store/src/maintenance/tests.rs",
                (("cleanup", "cleanup.rs"), ("delete_one", "delete_one.rs")),
                "crates/mesh-llm-log-store/src/maintenance/tests",
            ),
            (
                "crates/mesh-llm-host-runtime/src/network/openai/transport_tests.rs",
                (("lifecycle", "lifecycle.rs"), ("routing", "routing.rs")),
                "crates/mesh-llm-host-runtime/src/network/openai/transport_tests",
            ),
        )
        for parent, modules, child_directory in suites:
            for module_name, filename in modules:
                with self.subTest(parent=parent, module=module_name):
                    self.assert_owner_module(
                        parent,
                        module_name,
                        f"{child_directory}/{filename}",
                    )
            parent_source = self.source(parent)
            self.assertFalse("#[tokio::test]" in parent_source)
            self.assertFalse("#[test]" in parent_source)
            self.assertLessEqual(
                len(parent_source.splitlines()),
                MAX_NEW_FILE_LINES,
                f"test suite parent remains oversized: {parent}",
            )


if __name__ == "__main__":
    unittest.main()
