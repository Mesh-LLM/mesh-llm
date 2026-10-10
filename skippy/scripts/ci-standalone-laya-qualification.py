#!/usr/bin/env python3
"""Execute System One and Decisions against one composed standalone Laya product."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_ci_qualification", Path(__file__).with_name("validate-ci-qualification.py")
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)

DEVICES = {"cpu": "CPU", "cuda": "CUDA0", "rocm": "ROCm0",
           "vulkan": "Vulkan0", "metal": "MTL0"}
MODEL_ARTIFACT = "family-laya-multilingual"
QUESTION_STATE = "I was charged twice this month. Please refund the second payment."
SYSTEM_QUESTIONS = {
    "question_0": {"type": "noul", "instructions": "Does this need action today?"},
    "question_1": {"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Payments and refunds", "support": "Technical help"}},
    "question_2": {"type": "score", "instructions": "How frustrated?",
                   "criteria": ["Calm", "Frustrated"]},
}
DECISION_QUESTIONS = [
    {"type": "predicate", "name": "urgent", "instructions": "Does this need action today?"},
    {"type": "choice", "name": "team", "instructions": "Which team?",
     "choices": [{"value": key, "description": value}
                 for key, value in SYSTEM_QUESTIONS["question_1"]["criteria"].items()]},
    {"type": "score", "name": "frustration", "instructions": "How frustrated?",
     "levels": [{"label": label, "description": label}
                for label in SYSTEM_QUESTIONS["question_2"]["criteria"]]},
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def request(base: str, route: str, body: dict | None = None) -> tuple[int, dict]:
    payload = None if body is None else json.dumps(body).encode()
    target = Request(base + route, data=payload,
                     headers={"Content-Type": "application/json"} if payload else {})
    try:
        response = urlopen(target, timeout=300)
    except HTTPError as error:
        response = error
    with response:
        result = json.load(response)
        require(isinstance(result, dict), f"{route} returned a non-object")
        return response.status, result


def positive(base: str, route: str, body: dict) -> dict:
    status, response = request(base, route, body)
    require(status == 200, f"{route} returned HTTP {status}: {response}")
    return response


def negative(base: str, route: str, body: dict | None, expected: int) -> None:
    status, response = request(base, route, body)
    require(status == expected, f"{route} returned HTTP {status}; expected {expected}")
    error = response.get("error")
    require(isinstance(error, dict) and bool(error.get("message")),
            f"{route} lacked a typed error: {response}")


def system_body(model_id: str) -> dict:
    return {"model": model_id, "state": QUESTION_STATE, "questions": SYSTEM_QUESTIONS}


def decisions_body(model_id: str) -> dict:
    return {"model": model_id, "input": QUESTION_STATE,
            "questions": copy.deepcopy(DECISION_QUESTIONS)}


def equivalence(system: dict, decisions: dict, model_id: str) -> None:
    require(system.get("model") == decisions.get("model") == model_id,
            "System One and Decisions model identities differ")
    answers = decisions.get("answers")
    require(isinstance(answers, list) and [(x.get("type"), x.get("name")) for x in answers] ==
            [("predicate", "urgent"), ("choice", "team"), ("score", "frustration")],
            "Decisions answer order or names differ")
    original = system.get("answers")
    require(isinstance(original, dict) and set(original) == set(SYSTEM_QUESTIONS),
            "System One answer set differs")
    predicate, choice, score = answers
    require(isinstance(predicate.get("probability"), (int, float))
            and math.isfinite(predicate["probability"])
            and 0 <= predicate["probability"] <= 1,
            "predicate probability is not finite and bounded")
    require(abs(predicate["probability"] - original["question_0"]["noul"]) <= 1e-6,
            "predicate probability differs from System One")
    require(choice["choice"] == original["question_1"]["choice"],
            "choice answer differs from System One")
    require([item.get("value") for item in choice["probabilities"]] == ["billing", "support"],
            "choice values or order differ")
    choice_probabilities = [item["probability"] for item in choice["probabilities"]]
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                for value in choice_probabilities)
            and abs(sum(choice_probabilities) - 1) <= 1e-3,
            "choice probabilities are not normalized")
    require(choice["choice"] == choice["probabilities"][max(
        range(len(choice_probabilities)), key=choice_probabilities.__getitem__)]["value"],
        "choice does not select the highest probability")
    for item in choice["probabilities"]:
        require(abs(item["probability"] - original["question_1"]["probabilities"][item["value"]]) <= 1e-6,
                "choice probability differs from System One")
    require(abs(score["score"] - original["question_2"]["score"]) <= 1e-6,
            "score expectation differs from System One")
    require([(item.get("value"), item.get("label")) for item in score["probabilities"]] ==
            [(0, "Calm"), (1, "Frustrated")], "score levels or order differ")
    score_probabilities = [item["probability"] for item in score["probabilities"]]
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                for value in score_probabilities)
            and abs(sum(score_probabilities) - 1) <= 1e-3,
            "score probabilities are not normalized")
    require(abs(score["score"] - sum(index * value for index, value in enumerate(score_probabilities))) <= 1e-3,
            "score is not the distribution expectation")
    for item in score["probabilities"]:
        require(abs(item["probability"] - original["question_2"]["probabilities"][str(item["value"])]) <= 1e-6,
                "score probability differs from System One")
    usage = decisions.get("usage")
    require(isinstance(usage, dict) and type(usage.get("input_tokens")) is int
            and usage["input_tokens"] > 0 and usage.get("output_tokens") == 0,
            "Decisions must perform a read without generation")


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def start(product: Path, manifest: dict, model: Path, model_id: str,
          device: str, evidence_dir: Path, label: str) -> tuple[subprocess.Popen, str, Path]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = evidence_dir / f"laya-{label}.log"
    command = [str(product / manifest["cli"]["path"]),
               "--runtime-bundle", str(product / manifest["runtime"]["path"]),
               "--runtime-selection", manifest["backend"],
               "serve", "--model-path", str(model), "--model-id", model_id,
               "--device", device, "--bind-addr", f"127.0.0.1:{port}"]
    log = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    finally:
        log.close()
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ValueError(f"Laya CLI exited during startup: {log_path.read_text(errors='replace')[-4000:]}")
            try:
                status, models = request(base, "/v1/models")
                if status == 200 and any(item.get("id") == model_id for item in models.get("data", [])
                                         if isinstance(item, dict)):
                    return process, base, log_path
            except (URLError, TimeoutError):
                pass
            time.sleep(1)
        raise ValueError(f"Laya CLI did not become ready: {log_path.read_text(errors='replace')[-4000:]}")
    except BaseException:
        stop(process)
        raise


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_driver(script: Path, base: str, model_id: str, output: Path,
               *extra: str) -> dict:
    subprocess.run([sys.executable, str(script), "--base-url", base,
                    "--model", model_id, "--json-out", str(output), *extra],
                   check=True, timeout=900)
    report = contract.load_json(output)
    require(report.get("passed") is True or report.get("status") == "pass",
            f"{script.name} did not pass")
    return report


def evidence(suite: str, source_sha: str, row_id: str, product_digest: str,
             model_hash: str, observations: dict) -> dict:
    return {"schema_version": 1, "status": "passed", "source_sha": source_sha,
            "row_id": row_id, "suite": suite,
            "product_manifest_sha256": product_digest,
            "executed_cases": sorted(contract.SUITE_CASES[suite]),
            "models": {MODEL_ARTIFACT: model_hash}, "observations": observations}


def run(product: Path, model: Path, model_hash: str, model_id: str,
        row_id: str, device: str | None, evidence_dir: Path) -> None:
    manifest_path = product / "product-manifest.json"
    manifest = contract.load_json(manifest_path)
    row = contract.catalog_row(row_id)
    require(manifest.get("source_sha") and manifest.get("target") == row["target"]
            and manifest.get("backend") == row["backend"], "product differs from selected row")
    device = device or DEVICES[row["backend"]]
    require(device == DEVICES[row["backend"]], "device differs from selected backend")
    contract.validate_product_bytes(manifest, product)
    require(model.is_file() and contract.digest(model) == model_hash, "pinned Laya bytes differ")
    contract.validate_models("system-one", {MODEL_ARTIFACT: model_hash})
    evidence_dir.mkdir(parents=True, exist_ok=True)
    observations: dict[str, str] = {}
    process, base, log_path = start(product, manifest, model, model_id, device, evidence_dir, "initial")
    try:
        parity = evidence_dir / "laya-goldens.json"
        report = run_driver(ROOT / "scripts/skippy-laya-parity.py", base, model_id, parity)
        require(len(report.get("results", [])) >= 7, "Laya golden battery is incomplete")
        observations["goldens_sha256"] = contract.digest(parity)
        reader = evidence_dir / "laya-reader-contract.json"
        report = run_driver(Path(__file__).with_name("skippy-system-one-cases.py"),
                            base, model_id, reader, "--mode", "full-read", "--alias", model_id)
        require(len(report.get("cases", [])) >= 6, "Laya reader contract is incomplete")
        observations["reader_sha256"] = contract.digest(reader)
        system = positive(base, "/systemone", system_body(model_id))
        decisions = positive(base, "/v1/decisions", decisions_body(model_id))
        equivalence(system, decisions, model_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = list(pool.map(lambda _: positive(base, "/systemone", system_body(model_id)), range(2)))
        require(all(item.get("answers") == system.get("answers") for item in concurrent),
                "concurrent System One reads changed answers")
        negative(base, "/systemone", {"model": model_id, "state": QUESTION_STATE, "questions": {}}, 400)
        negative(base, "/systemone", system_body("missing-laya-model"), 400)
        negative(base, "/systemone", None, 405)
        negative(base, "/v1/decisions", {"model": model_id, "input": QUESTION_STATE, "questions": []}, 400)
        duplicate = decisions_body(model_id)
        duplicate["questions"][1]["choices"].append(
            duplicate["questions"][1]["choices"][0]
        )
        negative(base, "/v1/decisions", duplicate, 400)
    finally:
        stop(process)
    observations["initial_log_sha256"] = contract.digest(log_path)
    process, base, log_path = start(product, manifest, model, model_id, device, evidence_dir, "restart")
    try:
        resumed_system = positive(base, "/systemone", system_body(model_id))
        resumed_decisions = positive(base, "/v1/decisions", decisions_body(model_id))
        equivalence(resumed_system, resumed_decisions, model_id)
        require(resumed_system.get("answers") == system.get("answers"),
                "System One answers changed after process restart")
    finally:
        stop(process)
    observations["restart_log_sha256"] = contract.digest(log_path)
    for suite in ("system-one", "decisions"):
        result = evidence(suite, manifest["source_sha"], row_id,
                          contract.digest(manifest_path), model_hash, observations)
        (evidence_dir / f"{suite}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--device", help="Backend device token; defaults to the selected row's device.")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.product_dir.resolve(), args.model.resolve(), args.model_sha256,
            args.model_id, args.row_id, args.device, args.evidence_dir)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError,
            json.JSONDecodeError) as error:
        print(f"standalone Laya qualification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
