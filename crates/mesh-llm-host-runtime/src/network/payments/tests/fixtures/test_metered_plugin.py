"""Contract tests for the dependency-free external plugin fixture."""
import importlib.util
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import unittest
import urllib.request
import os

SPEC = importlib.util.spec_from_file_location(
    "metered_plugin", Path(__file__).with_name("metered_plugin.py")
)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


class FixtureContract(unittest.TestCase):
    def test_protobuf_roundtrip(self):
        for number in (0, 1, 127, 128, 65536, 2**63 - 1):
            self.assertEqual(fixture.fields(fixture.integer(3, number))[3], number)
        self.assertEqual(fixture.fields(fixture.data(5, "abc"))[5], b"abc")

    def test_registration_health_and_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / "plugin.sock")
            with socket.socket(socket.AF_INET) as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            with socket.socket(socket.AF_UNIX) as listener:
                listener.bind(address)
                listener.listen(1)
                listener.settimeout(5)
                env = dict(os.environ, MESH_LLM_PLUGIN_ENDPOINT=address,
                           MESH_LLM_PLUGIN_NAME="metering-fixture")
                process = subprocess.Popen(["python3", SPEC.origin, str(port)], env=env)
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(5)
                        response = self.exchange(connection, 10, b"", 1)
                        init = fixture.fields(response[11])
                        self.assertEqual(init[1], b"metering-fixture")
                        self.assertEqual(init[2], 3)
                        manifest = fixture.fields(init[6])
                        endpoint = fixture.fields(manifest[7])
                        self.assertEqual(endpoint[4], b"openai_compatible")
                        base = endpoint[5].decode()
                        with urllib.request.urlopen(base + "/models", timeout=5) as result:
                            self.assertIn(b'"test"', result.read())
                        body = b'{"stream_options":{"include_usage":true}}'
                        request = urllib.request.Request(base + "/chat/completions", body,
                                                         {"x-request-id": "contract"})
                        with urllib.request.urlopen(request, timeout=5) as result:
                            stream = result.read()
                            self.assertIn(b'"prompt_tokens":40', stream)
                            self.assertIn(b"[DONE]", stream)
                        health = self.exchange(connection, 12, b"", 2)
                        self.assertEqual(fixture.fields(health[13])[1], 1)
                        self.exchange(connection, 14, b"", 3)
                    self.assertEqual(process.wait(timeout=5), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def exchange(self, connection, field, payload, request_id):
        body = fixture.integer(1, 3) + fixture.integer(3, request_id)
        body += fixture.data(field, payload)
        connection.sendall(struct.pack("<I", len(body)) + body)
        size = struct.unpack("<I", fixture.read_exact(connection, 4))[0]
        result = fixture.fields(fixture.read_exact(connection, size))
        self.assertEqual(result[3], request_id)
        return result


if __name__ == "__main__":
    unittest.main()
