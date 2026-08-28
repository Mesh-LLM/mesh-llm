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

    def assert_semantic_owner(
        self,
        parent: str,
        module_name: str,
        child: str,
        owned_symbols: tuple[str, ...],
    ) -> None:
        self.assert_owner_module(
            parent,
            module_name,
            child,
            forbidden_parent_text=owned_symbols,
        )
        child_source = self.source(child)
        for symbol in owned_symbols:
            self.assertIn(symbol, child_source, f"{symbol!r} missing from {child}")
        pure_lines = [
            line
            for line in child_source.splitlines()
            if line.strip() and not line.lstrip().startswith("//")
        ]
        self.assertLessEqual(
            len(pure_lines),
            250,
            f"semantic production owner exceeds 250 pure lines: {child}",
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
        self.assert_semantic_owner(
            "crates/mesh-llm-host-runtime/src/logging/service/operational_audit.rs",
            "context",
            "crates/mesh-llm-host-runtime/src/logging/service/operational_audit/context.rs",
            (
                "OPERATIONAL_AUDIT_CONTEXT_VERSION",
                "MAX_CONTEXT_VALUE_CHARS",
                "MAX_NUMERIC_SUMMARIES",
                "pub enum OperationalAuditSubjectKind",
                "pub enum OperationalAuditPathType",
                "pub struct OperationalAuditContext",
                "fn insert_optional_string",
                "fn valid_static_code",
                "fn bounded_context_value",
            ),
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

    def test_log_store_repository_responsibilities_have_named_owners(self) -> None:
        repositories = "crates/mesh-llm-log-store/src/repositories.rs"
        audit = "crates/mesh-llm-log-store/src/repositories/audit.rs"
        self.assert_owner_module(
            repositories,
            "audit",
            audit,
            forbidden_parent_text=(
                "pub struct AuditEntryRow",
                "struct StoredAuditDetail",
                "fn audit_entry_query_parts",
                "fn audit_entry_row",
                "pub fn insert_audit_entry",
                "pub fn list_audit_entries(",
                "pub fn list_audit_entries_after_sequence(",
            ),
        )
        self.assert_owner_module(
            audit,
            "detail",
            "crates/mesh-llm-log-store/src/repositories/audit/detail.rs",
            forbidden_parent_text=(
                "struct StoredAuditDetail",
                "fn bounded_audit_value",
                "fn bounded_command_summary",
                "fn bounded_audit_code",
                "fn bounded_code",
            ),
        )
        self.assert_owner_module(
            repositories,
            "caller_metadata",
            "crates/mesh-llm-log-store/src/repositories/caller_metadata.rs",
            forbidden_parent_text=("pub fn upsert_summary_metadata(",),
        )

    def test_mesh_connection_responsibilities_have_named_owners(self) -> None:
        connections = "crates/mesh-llm-host-runtime/src/mesh/connections.rs"
        self.assert_semantic_owner(
            connections,
            "inbound",
            "crates/mesh-llm-host-runtime/src/mesh/connections/inbound.rs",
            (
                "pub(crate) async fn handle_incoming(",
                "pub(crate) async fn handle_control_incoming(",
                "pub(crate) async fn accept_mesh_stream(",
                "pub(crate) async fn admitted_mesh_stream(",
            ),
        )
        self.assert_semantic_owner(
            connections,
            "tunnel",
            "crates/mesh-llm-host-runtime/src/mesh/connections/tunnel.rs",
            (
                "pub(crate) async fn dispatch_mesh_stream(",
                "pub(crate) async fn forward_tunnel_stream(",
                "pub(crate) async fn forward_tunnel_http_stream(",
                "pub(crate) async fn _dispatch_streams(",
                "pub(crate) async fn authenticated_peer_path(",
                "pub(crate) async fn remove_connection_if_stable_id(",
            ),
        )
        self.assert_semantic_owner(
            "crates/mesh-llm-host-runtime/src/mesh/connections/inbound.rs",
            "stage",
            "crates/mesh-llm-host-runtime/src/mesh/connections/inbound/stage.rs",
            ("pub(crate) async fn handle_stage_alpn(",),
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

    def test_audit_and_gossip_suites_have_semantic_children(self) -> None:
        suites = (
            (
                "crates/mesh-llm-log-store/src/api_acceptance_tests/summary_audit.rs",
                (
                    ("mod basic;", "summary_audit/basic.rs"),
                    ("mod sanitization;", "summary_audit/sanitization.rs"),
                    ("mod query;", "summary_audit/query.rs"),
                ),
            ),
            (
                "crates/mesh-llm-host-runtime/src/mesh/tests/gossip.rs",
                (
                    (
                        'include!("gossip/merge_and_refresh.rs");',
                        "gossip/merge_and_refresh.rs",
                    ),
                    ('include!("gossip/admission.rs");', "gossip/admission.rs"),
                    ('include!("gossip/discovery.rs");', "gossip/discovery.rs"),
                ),
            ),
        )
        for parent, children in suites:
            parent_path = ROOT / parent
            parent_source = parent_path.read_text(encoding="utf-8")
            with self.subTest(parent=parent):
                self.assertTrue(parent_source.strip(), f"empty test parent: {parent}")
                self.assertNotIn("#[test]", parent_source)
                self.assertNotIn("#[tokio::test", parent_source)
                self.assertLessEqual(
                    len(parent_source.splitlines()),
                    MAX_NEW_FILE_LINES,
                    f"test suite parent remains oversized: {parent}",
                )
            for declaration, relative_child in children:
                child = str(Path(parent).parent / relative_child)
                child_path = ROOT / child
                with self.subTest(parent=parent, child=child):
                    self.assertIn(declaration, parent_source)
                    self.assertTrue(child_path.is_file(), f"missing test child: {child}")
                    child_source = child_path.read_text(encoding="utf-8")
                    self.assertTrue(child_source.strip(), f"empty test child: {child}")
                    self.assertLessEqual(
                        len(child_source.splitlines()),
                        MAX_NEW_FILE_LINES,
                        f"test suite child remains oversized: {child}",
                    )


if __name__ == "__main__":
    unittest.main()
