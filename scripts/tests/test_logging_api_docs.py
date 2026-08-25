"""Structural contract tests for the trusted-local logging API documentation."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "website" / "src" / "docs" / "pages" / "logging-api.md"
OPERATOR_GUIDE = ROOT / "docs" / "LOGGING.md"
DOCS_NAV = ROOT / "website" / "src" / "_data" / "docs.js"
API_REFERENCE = ROOT / "website" / "src" / "docs" / "pages" / "api-reference.md"


class LoggingApiDocumentationTests(unittest.TestCase):
    def page(self) -> str:
        self.assertTrue(PAGE.is_file(), "the authored logging API page must exist")
        return PAGE.read_text(encoding="utf-8")

    def test_page_keeps_operator_contract_sections(self) -> None:
        headings = {
            re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            for title in re.findall(r"^##\s+(.+?)\s*$", self.page(), flags=re.MULTILINE)
        }
        self.assertTrue(
            {
                "scope-and-trust-boundary",
                "status-and-health",
                "read-api",
                "live-sse",
                "export-and-artifact-controls",
                "cleanup-and-deletion",
                "terminal-webhooks",
                "privacy-and-errors",
                "configuration-and-limits",
                "compatibility",
            }.issubset(headings)
        )

    def test_page_covers_every_route_family(self) -> None:
        page = self.page()
        for route in (
            "GET /api/status",
            "GET /api/logs/requests",
            "GET /api/logs/requests/{requestId}",
            "GET /api/logs/requests/{requestId}/events",
            "GET /api/logs/requests/{requestId}/artifacts",
            "GET /api/logs/artifacts/{artifactId}",
            "GET /api/logs/proxy",
            "GET /api/logs/audit",
            "GET /api/logs/events",
            "POST /api/logs/requests/export",
            "POST /api/logs/cleanup/preview",
            "POST /api/logs/cleanup/run",
            "POST /api/logs/requests/{requestId}/delete",
            "POST /api/logs/webhooks/{deliveryId}/retry",
        ):
            with self.subTest(route=route):
                self.assertIn(route, page)

    def test_page_covers_recovery_privacy_and_configuration_contracts(self) -> None:
        page = self.page()
        for anchor in (
            "loopback",
            "`Host`",
            "`Origin`",
            "nextCursor",
            "Last-Event-ID",
            "v1:",
            "replay_gap",
            "stream_error",
            "operationId",
            "selection fingerprint",
            "request_id",
            "status_code",
            "scheduled",
            "already_scheduled",
            "dead-letter",
            "metadata_only",
            "redacted_artifacts",
            "completed",
            "failed",
            "rejected",
            "cancelled",
            "dropped",
            "logging.enabled",
            "logging.retention_ttl_secs",
            "logging.retention_max_rows",
            "logging.replay_capacity",
            "logging.queue_capacity",
            "logging.artifact.capture_mode",
            "logging.export_limit_bytes",
            "logging.cleanup_cadence_secs",
            "logging.webhook.enabled",
            "logging.webhook.url",
            "logging.webhook.max_attempts",
            "logging.webhook.timeout_secs",
            "logging.webhook.dead_letter_retention_secs",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, page)

    def test_stream_error_audit_recovery_contract_is_anchored(self) -> None:
        for document in (PAGE, OPERATOR_GUIDE):
            with self.subTest(document=document):
                text = document.read_text(encoding="utf-8")
                for anchor in (
                    "invalid_event",
                    "audit_reconcile_failed",
                    "a1:",
                    "GET /api/logs/audit",
                ):
                    self.assertIn(anchor, text)

    def test_page_is_linked_from_developer_navigation_and_api_reference(self) -> None:
        expected_url = "/docs/pages/logging-api/"
        self.assertIn(expected_url, DOCS_NAV.read_text(encoding="utf-8"))
        self.assertIn(expected_url, API_REFERENCE.read_text(encoding="utf-8"))

    def test_incompatible_version_audit_scope_is_explicit_in_authored_docs(self) -> None:
        for document in (OPERATOR_GUIDE, PAGE):
            with self.subTest(document=document):
                text = document.read_text(encoding="utf-8")
                normalized = re.sub(r"\s+", " ", text.lower())
                self.assertIn("`gossip_incompatible_version_rejected` is emitted when a direct peer is", text)
                self.assertIn("A transitive\nannouncement below the local version floor is dropped without emitting this", text)

    def test_schema_incompatibility_contract_is_typed_and_version_neutral(self) -> None:
        for document in (OPERATOR_GUIDE, PAGE):
            with self.subTest(document=document):
                text = document.read_text(encoding="utf-8")
                normalized = re.sub(r"\s+", " ", text.lower())
                for term in (
                    "logging_schema_incompatible",
                    "schema_version",
                    "supported_schema_version",
                    "left unchanged",
                    "inference remains available",
                    "log_store.db-wal",
                    "log_store.db-shm",
                    "PRAGMA user_version",
                ):
                    self.assertIn(term.lower(), normalized)
                self.assertRegex(normalized, r"logging metadata (?:is|becomes) unavailable")
                self.assertNotRegex(normalized, r"legacy (rows|entries)|previously retained|older schema|newer schema")

    def test_stable_503_table_lists_all_codes(self) -> None:
        page = self.page()
        row = re.search(r"^\|\s*503\s*\|\s*([^|]+?)\s*\|\s*$", page, flags=re.MULTILINE)
        self.assertIsNotNone(row, "the stable 503 error-code table row must exist")
        assert row is not None
        self.assertEqual(
            set(re.findall(r"`([^`]+)`", row.group(1))),
            {
                "artifact_deletion_unavailable",
                "export_timed_out",
                "maintenance_cancelled",
                "logging_unavailable",
                "store_unavailable",
                "logging_schema_incompatible",
            },
        )


if __name__ == "__main__":
    unittest.main()
