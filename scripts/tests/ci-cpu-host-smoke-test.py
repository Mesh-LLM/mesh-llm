#!/usr/bin/env python3
"""Exercise the portable smoke runner with a local stand-in host."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci-cpu-host-smoke.py"
FAKE_HOST = """#!/usr/bin/env python3
import http.server
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[:2] == ["runtime", "install"]:
    cache = Path(args[args.index("--cache-dir") + 1])
    runtime = cache / "0.80.3" / "fixture-cpu"
    runtime.mkdir(parents=True)
    (runtime / "manifest.json").write_text(json.dumps({
        "runtime": {"id": "fixture-cpu", "backend": {"kind": os.getenv("FAKE_BACKEND", "cpu")}}
    }))
    sys.exit(0)

port = int(args[args.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"data": [{"id": "fixture-model"}]}
        self.reply(body)
    def do_POST(self):
        body = {"model": "fixture-model", "choices": [{"message": {"content": "Paris"}}],
                "usage": {"completion_tokens": 1}}
        self.reply(body)
    def reply(self, body):
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def run(smoke: Path, host: Path, model: Path, backend: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(smoke), str(host), str(model)],
        env={**os.environ, "FAKE_BACKEND": backend},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mesh-cpu-smoke-test-") as temporary:
        work = Path(temporary)
        host = work / "fake-host"
        host.write_text(FAKE_HOST, encoding="utf-8")
        host.chmod(0o755)
        model = work / "model.gguf"
        model.write_bytes(b"fixture")
        success = run(SMOKE, host, model, "cpu")
        if success.returncode or "CPU host smoke OK" not in success.stdout:
            raise AssertionError(f"CPU fixture failed: {success.stdout}\n{success.stderr}")
        wrong_backend = run(SMOKE, host, model, "cuda")
        if wrong_backend.returncode == 0 or "expected one installed CPU runtime" not in wrong_backend.stderr:
            raise AssertionError(f"wrong backend was accepted: {wrong_backend.stdout}\n{wrong_backend.stderr}")
    print("portable CPU host smoke: ok")


if __name__ == "__main__":
    main()
