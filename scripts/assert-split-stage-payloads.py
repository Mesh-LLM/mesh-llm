#!/usr/bin/env python3
"""Certify Auto payload selection for every stage in one split-smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class PayloadError(ValueError):
    """A stage selection disagrees with the pinned smoke expectation."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PayloadError(f"{path} must contain a JSON object")
    return value


def log_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("attributes"), dict):
            events.append(event)
    return events


def observer_for_stage(stage: dict[str, Any], observers: dict[str, Any]) -> str:
    matches = [
        node for node in ("seed", "worker")
        if stage["node_id"].startswith(observers[node]["node_id"])
    ]
    if len(matches) != 1:
        raise PayloadError(f"stage {stage['stage_id']} has ambiguous node: {matches}")
    return matches[0]


def selection(attributes: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "payload": "skippy.kv.payload",
        "reason": "skippy.kv.payload_selection_reason",
        "fallbacks": "skippy.kv.payload_fallbacks",
        "admitted_graph_state": "skippy.kv.admitted_graph_state",
        "loaded_state_kind": "skippy.kv.loaded_state_kind",
        "resident": "skippy.kv.loaded_memory_cache_resident",
        "kv_recurrent": "skippy.kv.loaded_memory_cache_kv_recurrent",
        "graph_loaded_state_mismatches": "skippy.kv.graph_loaded_state_mismatches",
    }
    return {name: attributes.get(key) for name, key in keys.items()}


def certify(
    evidence: dict[str, Any], expectation: dict[str, Any],
    logs: dict[str, list[dict[str, Any]]], artifact_id: str, model_sha256: str,
) -> dict[str, Any]:
    if expectation.get("artifact_id") != artifact_id or expectation.get("sha256") != model_sha256:
        raise PayloadError("model artifact ID or SHA-256 differs from the pinned expectation")
    topology = evidence["topology"]
    if evidence.get("status") != "ready" or evidence.get("model_id") != topology["model_id"]:
        raise PayloadError("split topology evidence is not ready for one model")
    stages = topology["stages"]
    expected_stages = expectation["stages"]
    if len(stages) != len(expected_stages) or {
        stage["stage_index"] for stage in stages
    } != set(range(len(expected_stages))):
        raise PayloadError("topology stages differ from the explicit expectation")
    stage_by_id = {stage["stage_id"]: stage for stage in stages}
    if len(stage_by_id) != len(stages):
        raise PayloadError("duplicate stage ID in topology")
    matched: dict[str, list[dict[str, Any]]] = {stage_id: [] for stage_id in stage_by_id}
    exact_kinds: dict[str, set[str]] = {stage_id: set() for stage_id in stage_by_id}
    lookups: dict[str, list[dict[str, Any]]] = {stage_id: [] for stage_id in stage_by_id}
    identity = {
        "skippy.run_id": topology["run_id"],
        "skippy.model_id": topology["model_id"],
        "skippy.topology_id": topology["topology_id"],
    }
    for node, events in logs.items():
        for event in events:
            attrs = event["attributes"]
            if any(attrs.get(key) != value for key, value in identity.items()):
                continue
            is_selection = event.get("event") == "stage.kv_payload_selected"
            is_lookup = event.get("event") in {
                "stage.binary_kv_lookup_decision", "stage.openai_kv_lookup_decision",
            }
            if not is_selection and "skippy.exact_cache.payload_kind" not in attrs and not is_lookup:
                continue
            stage_id = attrs.get("skippy.stage_id")
            if stage_id not in stage_by_id:
                raise PayloadError(f"unexpected stage selection for {stage_id!r}")
            stage = stage_by_id[stage_id]
            if node != observer_for_stage(stage, evidence["observers"]):
                raise PayloadError(f"stage {stage_id} emitted on wrong node {node}")
            if attrs.get("skippy.stage_index") != stage["stage_index"]:
                raise PayloadError(f"stage index mismatch for {stage_id}")
            exact_kind = attrs.get("skippy.exact_cache.payload_kind")
            if isinstance(exact_kind, str):
                exact_kinds[stage_id].add(exact_kind)
            if is_lookup:
                lookups[stage_id].append({
                    "event": event.get("event"),
                    "request_id": attrs.get("skippy.request_id"),
                    "decision": attrs.get("skippy.kv.decision"),
                    "restored_tokens": attrs.get("skippy.kv.restored_tokens"),
                })
            if is_selection:
                matched[stage_id].append(selection(attrs))
    results = []
    for stage in stages:
        stage_id = stage["stage_id"]
        stage_index = stage["stage_index"]
        outcomes = matched[stage_id]
        if not outcomes:
            raise PayloadError(f"missing selection for stage {stage_id}")
        unique = {json.dumps(outcome, sort_keys=True) for outcome in outcomes}
        if len(unique) != 1:
            raise PayloadError(f"contradictory selections for stage {stage_id}: {sorted(unique)}")
        if len(outcomes) != 1:
            raise PayloadError(f"duplicate selection for stage {stage_id}")
        observed = outcomes[0]
        expected = expected_stages[stage_index]
        for field, value in expected.items():
            if field == "exact_payload_kind":
                expected_kinds = set() if value is None else {value}
                if exact_kinds[stage_id] != expected_kinds:
                    raise PayloadError(
                        f"stage {stage_id} exact payload: expected {value!r}, "
                        f"observed {sorted(exact_kinds[stage_id])}"
                    )
            elif observed.get(field) != value:
                raise PayloadError(
                    f"stage {stage_id} {field}: expected {value!r}, observed {observed.get(field)!r}"
                )
        results.append({
            "stage_id": stage_id, "stage_index": stage_index,
            "node": observer_for_stage(stage, evidence["observers"]),
            "expected": expected, "observed": observed,
            "exact_payload_kinds": sorted(exact_kinds[stage_id]),
            "selection_event_count": len(outcomes),
            "lookups": lookups[stage_id],
        })
    return {
        "status": "pass", "artifact_id": artifact_id, "sha256": model_sha256,
        "model_id": topology["model_id"], "run_id": topology["run_id"],
        "topology_id": topology["topology_id"], "stages": results,
    }


def certify_from_files(args: argparse.Namespace) -> dict[str, Any]:
    expectations = load_json(args.expectations)
    expectation = expectations["models"].get(args.artifact_id)
    if not isinstance(expectation, dict):
        raise PayloadError(f"no pinned Auto expectation for {args.artifact_id}")
    manifest = load_json(args.model_manifest)
    artifacts = manifest.get("artifacts")
    artifact = next(
        (item for item in artifacts if item.get("id") == args.artifact_id), None
    ) if isinstance(artifacts, list) else None
    if not isinstance(artifact, dict) or any(
        expectation.get(key) != artifact.get(key)
        for key in ("revision", "sha256")
    ):
        raise PayloadError("Auto expectation differs from the immutable model manifest")
    backend_device = getattr(args, "backend_device", "").lower()
    metal = backend_device.startswith("mtl") or "metal" in backend_device
    if metal and "metal_stages" in expectation:
        expectation = {**expectation, "stages": expectation["metal_stages"]}
    result = certify(
        load_json(args.evidence), expectation,
        {"seed": log_events(args.seed_log), "worker": log_events(args.worker_log)},
        args.artifact_id, args.model_sha256,
    )
    result["revision"] = expectation["revision"]
    result["tested_commit"] = args.tested_commit
    result["native_recipe"] = load_json(args.roster)["native_recipe"]
    manifests = sorted(args.runtime_bundle.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise PayloadError(f"expected one packaged native runtime, found {len(manifests)}")
    runtime = load_json(manifests[0])["runtime"]
    if runtime.get("skippy_abi") != result["native_recipe"]["skippy_abi"]:
        raise PayloadError("loaded native runtime ABI differs from the checked-in roster")
    result["native_runtime"] = {
        "id": runtime.get("id"), "skippy_abi": runtime.get("skippy_abi"),
    }
    responses = [
        load_json(args.responses_dir / f"response-{index}.json")
        for index in (1, 2)
    ]
    cold_warm = []
    for response in responses:
        usage = response.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        content = response["choices"][0]["message"]["content"]
        if not isinstance(cached, int) or not isinstance(content, str):
            raise PayloadError("cold/warm response omitted cache usage or output")
        cold_warm.append({
            "request_id": response.get("id"), "cached_tokens": cached,
            "prompt_tokens": usage.get("prompt_tokens"), "output": content,
        })
    if cold_warm[0]["cached_tokens"] != 0 or cold_warm[1]["cached_tokens"] <= 0:
        raise PayloadError("expected a cold miss followed by a warm restore")
    if cold_warm[0]["output"] != cold_warm[1]["output"]:
        raise PayloadError("warm continuation differs from clean cold continuation")
    result["cold_warm"] = cold_warm
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--tested-commit", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--backend-device", default="")
    parser.add_argument("--seed-log", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = certify_from_files(args)
    except (OSError, KeyError, TypeError, ValueError) as error:
        try:
            topology = load_json(args.evidence).get("topology")
            models = load_json(args.expectations).get("models")
            pin = models.get(args.artifact_id) if isinstance(models, dict) else None
        except (OSError, TypeError, ValueError):
            topology, pin = None, None
        topology = topology if isinstance(topology, dict) else {}
        pin = pin if isinstance(pin, dict) else {}
        try:
            native_recipe = load_json(args.roster).get("native_recipe")
        except (OSError, ValueError):
            native_recipe = None
        args.output.write_text(json.dumps({
            "status": "fail", "error": str(error),
            "artifact_id": args.artifact_id, "sha256": args.model_sha256,
            "revision": pin.get("revision"),
            "tested_commit": args.tested_commit,
            "native_recipe": native_recipe,
            "model_id": topology.get("model_id"),
            "run_id": topology.get("run_id"),
            "stages": topology.get("stages"),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"certified Auto payload for {len(result['stages'])} loaded stages")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"split Auto payload certification failed: {error}") from error
