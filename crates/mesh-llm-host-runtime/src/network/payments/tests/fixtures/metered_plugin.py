"""External test adapter: real plugin IPC registration and HTTP inference.

Minimal protobuf encoding keeps this opt-in fixture dependency-free. Field numbers
follow mesh-llm-plugin/proto/plugin.proto; this is not a production plugin SDK.
"""
import json
import os
import socket
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def varint(number):
    result = bytearray()
    while number >= 128:
        result.append((number & 127) | 128)
        number >>= 7
    result.append(number)
    return bytes(result)


def integer(field, number):
    return varint(field << 3) + varint(number)


def data(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((field << 3) | 2) + varint(len(value)) + value


def decode_varint(body, offset):
    value = shift = 0
    while True:
        byte = body[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("invalid varint")


def fields(body):
    offset = 0
    result = {}
    while offset < len(body):
        key, offset = decode_varint(body, offset)
        if key & 7 == 0:
            value, offset = decode_varint(body, offset)
        elif key & 7 == 2:
            size, offset = decode_varint(body, offset)
            value = body[offset:offset + size]
            offset += size
        else:
            raise ValueError("unsupported protobuf field")
        result[key >> 3] = value
    return result


def read_exact(connection, count):
    result = bytearray()
    while len(result) < count:
        chunk = connection.recv(count - len(result))
        if not chunk:
            raise EOFError()
        result.extend(chunk)
    return bytes(result)


class Backend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/adapter/v1/models":
            self.send_error(404)
            return
        body = json.dumps({"data": [{"id": "test"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/adapter/v1/chat/completions"
        assert body["stream_options"]["include_usage"] is True
        assert self.headers["x-request-id"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\r\n\r\n')
        self.wfile.flush()
        self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":3}}\r\n\r\ndata: [DONE]\r\n\r\n')
        self.close_connection = True


def main():
    port = int(sys.argv[1])
    backend = ThreadingHTTPServer(("127.0.0.1", port), Backend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    name = os.environ["MESH_LLM_PLUGIN_NAME"]
    endpoint = data(1, "inference") + integer(2, 1) + integer(3, 1)
    endpoint += data(4, "openai_compatible") + data(5, f"http://127.0.0.1:{port}/adapter/v1")
    endpoint += integer(8, 1)
    manifest = data(7, endpoint)
    server_info = json.dumps({
        "protocolVersion": "2024-11-05", "capabilities": {},
        "serverInfo": {"name": name, "version": "fixture"},
    })
    initialize = data(1, name) + integer(2, 3) + data(3, "fixture")
    initialize += data(4, server_info) + data(6, manifest)
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(os.environ["MESH_LLM_PLUGIN_ENDPOINT"])
        try:
            while True:
                size = struct.unpack("<I", read_exact(connection, 4))[0]
                if size > 16 * 1024 * 1024:
                    raise ValueError("oversized control frame")
                request = fields(read_exact(connection, size))
                if 10 in request:
                    payload = data(11, initialize)
                elif 12 in request:
                    payload = data(13, integer(1, 1))
                elif 14 in request:
                    payload = data(15, b"")
                else:
                    continue
                response = integer(1, 3) + data(2, name) + integer(3, request[3]) + payload
                connection.sendall(struct.pack("<I", len(response)) + response)
                if 14 in request:
                    break
        except EOFError:
            pass
        finally:
            backend.shutdown()
            backend.server_close()


if __name__ == "__main__":
    main()
