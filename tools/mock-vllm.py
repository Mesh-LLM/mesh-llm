#!/usr/bin/env python3
"""Minimal mock of an OpenAI-compatible server for testing external backend plugins.

Mimics vLLM's response format (owned_by: "vllm", max_model_len, root field).

Usage:
    python3 tools/mock-vllm.py [PORT]    # default 8000
"""

import json
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

MODELS_RESPONSE = json.dumps({
    "object": "list",
    "data": [
        {
            "id": "meta-llama/Llama-3.1-8B-Instruct",
            "object": "model",
            "created": 1700000000,
            "owned_by": "vllm",
            "root": "meta-llama/Llama-3.1-8B-Instruct",
            "max_model_len": 131072,
        }
    ],
})

CHAT_RESPONSE = json.dumps({
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from mock vLLM! This is a test response.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
})

VERSION_RESPONSE = json.dumps({"version": "0.8.0-mock"})


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/v1/models/"):
            self._json_response(200, MODELS_RESPONSE)
        elif self.path in ("/health", "/health/"):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path in ("/version", "/version/"):
            self._json_response(200, VERSION_RESPONSE)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(content_length)
        if self.path in ("/v1/chat/completions", "/v1/chat/completions/"):
            self._json_response(200, CHAT_RESPONSE)
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, status, body_str):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        body = body_str.encode()
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[mock-vllm] {args[0]}")


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Mock vLLM server on http://127.0.0.1:{PORT}")
    print(f"  GET  /v1/models         → model list")
    print(f"  GET  /health            → 200")
    print(f"  GET  /version           → version")
    print(f"  POST /v1/chat/completions → mock response")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock server")
