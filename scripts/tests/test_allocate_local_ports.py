"""Regression coverage for dynamic local certification ports."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "allocate_local_ports", ROOT / "scripts/lib/allocate_local_ports.py"
)
PORTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PORTS)


class AllocateLocalPortsTests(unittest.TestCase):
    def test_ports_are_distinct_and_released_for_immediate_use(self) -> None:
        ports = PORTS.allocate_ports(5)
        self.assertEqual(5, len(ports))
        self.assertEqual(5, len(set(ports)))
        listeners = []
        self.addCleanup(lambda: [listener.close() for listener in listeners])
        for port in ports:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", port))
            listeners.append(listener)

    def test_os_selection_avoids_an_occupied_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied_port = occupied.getsockname()[1]
            self.assertNotIn(occupied_port, PORTS.allocate_ports(5))

    def test_invalid_counts_fail_closed(self) -> None:
        for count in (0, -1, PORTS.MAX_PORTS + 1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                PORTS.allocate_ports(count)


if __name__ == "__main__":
    unittest.main()
