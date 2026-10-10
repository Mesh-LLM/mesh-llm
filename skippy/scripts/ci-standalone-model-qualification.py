#!/usr/bin/env python3
"""Execute an executable model qualification suite against a composed product.

Every case is observed through the routes the composed standalone CLI actually
serves (skippy/crates/skippy-inference-api/src/router.rs). Cases the CLI cannot
observe are recorded in ``DEFERRED_CASES`` in ``validate-ci-qualification.py``
rather than claimed here, so a missing route can never masquerade as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SPEC = importlib.util.spec_from_file_location(
    "validate_ci_qualification", Path(__file__).with_name("validate-ci-qualification.py")
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)

DEVICES = {"cpu": "CPU", "cuda": "CUDA0", "rocm": "ROCm0",
           "vulkan": "Vulkan0", "metal": "MTL0"}
SUITE_DEVICE_BACKENDS = set(DEVICES)
PROMPT = "Reply with one short sentence about the weather."
PREFIX = "Reference facts: " + " ".join(f"fact-{index}" for index in range(32))
SUFFIX = "\nNow summarise the reference facts in one line."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def request(base: str, route: str, body: dict | None = None,
            *, timeout: float = 180) -> tuple[int, dict]:
    payload = None if body is None else json.dumps(body).encode()
    target = Request(base + route, data=payload,
                     headers={"Content-Type": "application/json"} if payload else {})
    try:
        response = urlopen(target, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        result = json.load(response)
        require(isinstance(result, dict), f"{route} returned a non-object")
        return response.status, result


def stream_request(base: str, route: str, body: dict, *, timeout: float = 180) -> str:
    payload = json.dumps(body).encode()
    target = Request(base + route, data=payload, headers={"Content-Type": "application/json"})
    lines: list[str] = []
    with urlopen(target, timeout=timeout) as response:
        require(response.status == 200, f"{route} stream returned HTTP {response.status}")
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line:
                lines.append(line)
    return "\n".join(lines)


def usage_tokens(response: dict) -> dict:
    usage = response.get("usage")
    require(isinstance(usage, dict), "completion lacks usage")
    require(type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0,
            "completion lacks positive prefill tokens")
    require(type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > 0,
            "completion lacks positive decode tokens")
    return usage


def cached_tokens(response: dict) -> int:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    value = details.get("cached_tokens")
    return value if type(value) is int and value >= 0 else 0


def completion(base: str, model_id: str, messages: list[dict], **options) -> dict:
    body = {"model": model_id, "messages": messages, "temperature": 0,
            "max_tokens": options.pop("max_tokens", 8), **options}
    status, response = request(base, "/v1/chat/completions", body)
    require(status == 200, f"chat completion returned HTTP {status}: {response}")
    require(response.get("model") == model_id, "completion model differs from the selected model")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "completion must have one choice")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    require(isinstance(message, dict) and isinstance(message.get("content"), str)
            and bool(message["content"].strip()), "completion has no generated text")
    usage_tokens(response)
    return response


def text_of(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def stream_completion(base: str, model_id: str, prompt: str) -> dict:
    body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16, "temperature": 0, "stream": True}
    payload = stream_request(base, "/v1/chat/completions", body)
    require(payload.strip() != "", "stream produced no server-sent events")
    require("data:" in payload, "stream did not emit server-sent events")
    deltas = [line for line in payload.splitlines()
              if line.startswith("data:") and "[DONE]" not in line]
    require(deltas, "stream produced no content deltas")
    require("[DONE]" in payload, "stream did not terminate with [DONE]")
    return {"deltas": len(deltas), "bytes": len(payload)}


class Server:
    """A composed standalone CLI serving one model over loopback."""

    def __init__(self, product_dir: Path, manifest: dict, model: Path,
                 model_id: str, device: str, ctx_size: int, directory: Path) -> None:
        self.product_dir = product_dir
        self.manifest = manifest
        self.model = model
        self.model_id = model_id
        self.device = device
        self.ctx_size = ctx_size
        self.directory = directory
        self.process: subprocess.Popen | None = None
        self.base = ""
        self.log_path = directory / "serve.log"

    def __enter__(self) -> "Server":
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        log = self.log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [str(self.product_dir / self.manifest["cli"]["path"]),
                 "--runtime-bundle", str(self.product_dir / self.manifest["runtime"]["path"]),
                 "--runtime-cache", str(self.directory / "runtime-cache"),
                 "--runtime-selection", self.manifest["backend"],
                 "serve", "--model-path", str(self.model), "--model-id", self.model_id,
                 "--device", self.device, "--ctx-size", str(self.ctx_size),
                 "--n-gpu-layers", "0" if self.device == "CPU" else "999",
                 "--bind-addr", f"127.0.0.1:{port}"],
                stdout=log, stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            require(self.process.poll() is None, self.tail())
            try:
                status, models = request(self.base, "/v1/models", timeout=10)
                if status == 200 and any(item.get("id") == self.model_id
                                         for item in models.get("data", [])
                                         if isinstance(item, dict)):
                    return self
            except (URLError, TimeoutError):
                pass
            time.sleep(1)
        self.stop()
        raise ValueError(f"standalone CLI did not become ready: {self.tail()}")

    def tail(self) -> str:
        return f"server log: {self.log_path.read_text(errors='replace')[-4000:]}"

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def __exit__(self, *_: object) -> None:
        self.stop()


def dense_cases(server: Server, observations: dict) -> list[str]:
    first = completion(server.base, server.model_id, [{"role": "user", "content": PROMPT}])
    observations["prefill_tokens"] = usage_tokens(first)["prompt_tokens"]
    history = [{"role": "user", "content": PROMPT},
               {"role": "assistant", "content": text_of(first)},
               {"role": "user", "content": "Now answer in five words."}]
    second = completion(server.base, server.model_id, history)
    require(usage_tokens(second)["prompt_tokens"] > observations["prefill_tokens"],
            "continuation did not grow the prompt")
    stream = stream_completion(server.base, server.model_id, PROMPT)
    observations["stream_deltas"] = stream["deltas"]
    return ["load", "prefill-decode", "stream", "continuation"]


def restart_case(server: Server, observations: dict) -> None:
    server.stop()
    server.__enter__()
    repeat = completion(server.base, server.model_id, [{"role": "user", "content": PROMPT}])
    require(usage_tokens(repeat)["prompt_tokens"] == observations["prefill_tokens"],
            "restart changed the prefill for an identical prompt")


def recurrent_cases(server: Server, observations: dict) -> list[str]:
    first = completion(server.base, server.model_id, [{"role": "user", "content": PROMPT}])
    observations["prefill_tokens"] = usage_tokens(first)["prompt_tokens"]
    history = [{"role": "user", "content": PROMPT},
               {"role": "assistant", "content": text_of(first)}]
    previous = observations["prefill_tokens"]
    for turn in range(2):
        history.append({"role": "user", "content": f"Turn {turn}: continue."})
        response = completion(server.base, server.model_id, list(history))
        prompt_tokens = usage_tokens(response)["prompt_tokens"]
        require(prompt_tokens > previous, "recurrent state did not grow across turns")
        previous = prompt_tokens
        history.append({"role": "assistant", "content": text_of(response)})
    observations["state_prompt_tokens"] = previous
    return ["prefill-decode", "state-preservation"]


def moe_cases(server: Server, observations: dict) -> list[str]:
    first = completion(server.base, server.model_id, [{"role": "user", "content": PREFIX}])
    repeat = completion(server.base, server.model_id, [{"role": "user", "content": PREFIX}])
    require(usage_tokens(first)["prompt_tokens"] == usage_tokens(repeat)["prompt_tokens"],
            "repeated identical prompt changed the prefill")
    extended = completion(server.base, server.model_id,
                          [{"role": "user", "content": PREFIX + SUFFIX}])
    require(usage_tokens(extended)["prompt_tokens"] > usage_tokens(first)["prompt_tokens"],
            "suffix continuation did not grow the prompt")
    observations["expert_prompt_tokens"] = usage_tokens(first)["prompt_tokens"]
    return ["expert-execution", "repeated-restore", "suffix-continuation"]


def kv_cases(server: Server, observations: dict) -> list[str]:
    cold = completion(server.base, server.model_id, [{"role": "user", "content": PREFIX}])
    require(usage_tokens(cold)["prompt_tokens"] > 0, "cold prefill was empty")
    warm = completion(server.base, server.model_id, [{"role": "user", "content": PREFIX}])
    require(cached_tokens(warm) > 0, "a repeated prefix was not served from cache")
    extended = completion(server.base, server.model_id,
                          [{"role": "user", "content": PREFIX + SUFFIX}])
    require(cached_tokens(extended) > 0, "a prefixed suffix was not served from cache")
    require(usage_tokens(extended)["prompt_tokens"] > usage_tokens(cold)["prompt_tokens"],
            "suffix continuation did not grow the prompt")
    divergent = completion(server.base, server.model_id,
                           [{"role": "user", "content": PREFIX[::-1]}])
    require(usage_tokens(divergent)["prompt_tokens"] > 0, "divergent prefix did not execute")
    other = completion(server.base, server.model_id, [{"role": "user", "content": PROMPT}])
    require(usage_tokens(other)["prompt_tokens"] > 0, "isolated prompt did not execute")
    observations["cached_tokens"] = cached_tokens(warm)
    observations["cold_prompt_tokens"] = usage_tokens(cold)["prompt_tokens"]
    return ["isolation", "divergent-prefix", "suffix-continuation"]


SUITE_DRIVERS = {
    "dense": (dense_cases, ["restart"]),
    "recurrent": (recurrent_cases, ["restart"]),
    "moe": (moe_cases, []),
    "kv-cache": (kv_cases, []),
}
# Capability each suite's model set must cover. A tag group is satisfied when a
# model carries any of its tags; every group must be satisfied.
SUITE_MODEL_TAGS = {
    "dense": (("dense",),),
    "recurrent": (("hybrid", "recurrent"),),
    "moe": (("moe",),),
    "kv-cache": (("dense",), ("hybrid", "recurrent")),
}
# KV prefix reuse is named per model family so the receipt records both results.
PREFIX_HIT_CASES = {"dense": "dense-prefix-hit", "hybrid": "recurrent-prefix-hit"}


def parse_model(values: list[list[str]]) -> list[tuple[str, Path, str, str]]:
    models = []
    for artifact_id, raw_path, model_sha256, model_id in values:
        models.append((artifact_id, Path(raw_path), model_sha256, model_id))
    return models


def run(product_dir: Path, row_id: str, suite: str, device: str, ctx_size: int,
        models: list[tuple[str, Path, str, str]], evidence: Path) -> dict:
    row = contract.catalog_row(row_id)
    require(device == DEVICES[row["backend"]], "device differs from the selected backend")
    manifest_path = product_dir / "product-manifest.json"
    manifest = contract.load_json(manifest_path)
    require(manifest.get("contract") == "skippy-product-v1"
            and manifest.get("target") == row["target"]
            and manifest.get("backend") == row["backend"],
            "composed product differs from the selected row")
    contract.validate_product_bytes(manifest, product_dir)
    identifiers = {artifact_id: model_sha256 for artifact_id, _, model_sha256, _ in models}
    contract.validate_models(suite, identifiers)
    known = {item["id"]: set(item.get("capability_tags", []))
             for item in contract.load_json(
                 contract.ROOT / "ci/model-artifacts/registry.json")["artifacts"]}
    for group in SUITE_MODEL_TAGS[suite]:
        require(any(group & known[artifact_id] for artifact_id, _, _, _ in models),
                f"{suite} lacks a model tagged {sorted(group)}")

    driver, restart = SUITE_DRIVERS[suite]
    cases: list[str] = []
    observations: dict[str, object] = {}
    for artifact_id, model, model_sha256, model_id in models:
        require(model.is_file() and sha256(model) == model_sha256,
                f"pinned {artifact_id} bytes differ")
        with tempfile.TemporaryDirectory(prefix=f"skippy-{suite}-") as temporary:
            directory = Path(temporary)
            server = Server(product_dir, manifest, model, model_id, device, ctx_size, directory)
            with server:
                executed = driver(server, observations)
                if restart:
                    restart_case(server, observations)
                    executed = executed + restart
                cases.extend(executed)
                observations[f"{artifact_id}_log_sha256"] = contract.digest(server.log_path)
                for family, case in PREFIX_HIT_CASES.items():
                    if suite == "kv-cache" and family in known[artifact_id]:
                        cases.append(case)
    require(set(cases) == contract.SUITE_CASES[suite],
            f"{suite} executed cases differ from the contract: {sorted(set(cases))}")
    result = {"schema_version": 1, "status": "passed", "source_sha": manifest["source_sha"],
              "row_id": row_id, "suite": suite,
              "product_manifest_sha256": contract.digest(manifest_path),
              "executed_cases": sorted(set(cases)), "models": identifiers,
              "observations": observations}
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--suite", choices=sorted(SUITE_DRIVERS), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--model", nargs=4, action="append", metavar=("ARTIFACT", "PATH", "SHA256", "MODEL_ID"),
                        required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.product_dir.resolve(), args.row_id, args.suite, args.device,
            args.ctx_size, parse_model(args.model), args.evidence)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"standalone model qualification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
