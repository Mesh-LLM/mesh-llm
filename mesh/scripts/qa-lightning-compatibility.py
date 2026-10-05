#!/usr/bin/env python3
"""Exercise released/current peers without provisioning or funding wallets.

Supply two extracted product bundles and the SmolLM2-135M-Instruct GGUF used by
the opt-in payment gate test. Each bundle must retain its own native-runtimes/.
The output directory must be new: all profiles and logs remain isolated there.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


MODEL = "Payment-Compatibility-Smoke"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def http(port, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class Node:
    def __init__(self, output, name, binary, model, provider, invite=None):
        self.root = output / name
        self.root.mkdir(exist_ok=True)
        self.console, self.port = unused_port(), unused_port()
        while self.port == self.console:
            self.port = unused_port()
        config = (
            'version = 1\n[runtime.native_runtime]\nselection = "cpu"\n'
            f'[logging]\napplication_state_root = {json.dumps(str(self.root / "logging"))}\n'
        )
        if provider:
            config += (
                f'[[models]]\nmodel = "{MODEL}"\n[models.hardware]\n'
                f'model_path = {json.dumps(str(model))}\ngpu_layers = 0\n'
                '[models.skippy]\nsource_policy = "local-required"\n'
                '[models.model_fit]\nctx_size = 1024\nbatch = 128\nubatch = 64\n'
            )
        config_path = self.root / "config.toml"
        config_path.write_text(config)
        env = os.environ.copy()
        for key, relative in [
            ("MESH_LLM_RUNTIME_ROOT", "run"),
            ("MESH_LLM_DATA_DIR", "data"),
            ("MESH_LLM_PLUGIN_DIR", "plugins"),
            ("MESH_LLM_NODE_KEY_PATH", "node.key"),
            ("MESH_LLM_NATIVE_RUNTIME_CACHE_DIR", "runtime-cache"),
        ]:
            env[key] = str(self.root / relative)
        env["MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"] = str(binary.parent / "native-runtimes")
        cmd = [str(binary), "serve" if provider else "client", "--config", str(config_path),
               "--console", str(self.console), "--port", str(self.port), "--log-format", "json"]
        if provider:
            cmd += ["--device", "CPU", "--no-draft", "--mesh-name", "PaymentCompatibility"]
        if invite:
            cmd += ["--join", invite]
        with (self.root / "node.log").open("ab") as log:
            self.process = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                            start_new_session=True)

    def wait(self, port, path, condition):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Node exited; inspect {self.root / 'node.log'}")
            try:
                status, value = http(port, path)
                if status == 200 and condition(value):
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"Node not ready; inspect {self.root / 'node.log'}")

    def status(self):
        return self.wait(self.console, "/api/status", lambda _: True)

    def model_ready(self):
        self.wait(self.port, "/v1/models", lambda d: any(m["id"] == MODEL for m in d["data"]))

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def check_inference(node, expected, name, evidence):
    node.model_ready()
    status, response = http(node.port, "/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "temperature": 0, "stream": False,
    })
    evidence["cases"].append({"case": name, "status": status, "response": response})
    if status != expected:
        raise AssertionError(f"{name}: expected {expected}, got {status}: {response}")
    if expected == 200:
        assert response["choices"][0]["message"]["content"]
        assert response["usage"]["completion_tokens"] > 0
    print(f"PASS: {name} (HTTP {status})", flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    nodes = []
    evidence = {
        "current_binary_sha256": digest(args.current_binary),
        "released_binary_sha256": digest(args.released_binary),
        "model_sha256": digest(args.model),
        "released_version": subprocess.check_output(
            [str(args.released_binary), "--version"], text=True).strip(),
        "cases": [],
    }

    def start(name, binary, provider, invite=None):
        node = Node(args.output, name, binary, args.model, provider, invite)
        nodes.append(node)
        return node, node.status()

    try:
        provider, status = start("current-provider", args.current_binary, True)
        provider.model_ready()
        client, _ = start("released-client", args.released_binary, False, status["token"])
        check_inference(client, 200, "released client to current free provider", evidence)
        client.stop()
        code, _ = http(provider.console, "/api/wallet", {
            "command": "set_pricing", "model": MODEL,
            "value": {"input_msat_per_million": 10000000,
                      "output_msat_per_million": 30000000, "minimum_invoice_msat": 1000},
        })
        assert code == 200
        # A fresh join observes the updated offer without waiting for heartbeat.
        client, _ = start("released-client", args.released_binary, False, status["token"])
        check_inference(client, 402, "released client cannot bypass paid provider", evidence)
        client.stop()
        provider.stop()
        provider, status = start("released-provider", args.released_binary, True)
        provider.model_ready()
        client, _ = start("current-client", args.current_binary, False, status["token"])
        check_inference(client, 200, "current client to released free provider", evidence)
        evidence["passed"] = True
    finally:
        for node in reversed(nodes):
            node.stop()
        (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["current-binary", "released-binary", "model", "output"]:
        parser.add_argument(f"--{flag}", required=True, type=lambda s: Path(s).resolve())
    run(parser.parse_args())
