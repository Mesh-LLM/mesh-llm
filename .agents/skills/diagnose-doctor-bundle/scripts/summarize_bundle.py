#!/usr/bin/env python3
"""Summarize a mesh-llm doctor bundle for support triage."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

PATTERNS = [
    ("panic", re.compile(r"\bpanic(?:ked)?\b", re.IGNORECASE)),
    ("error", re.compile(r"\b(error|failed|failure)\b", re.IGNORECASE)),
    ("oom", re.compile(r"(out of memory|\boom\b|cannot allocate)", re.IGNORECASE)),
    ("port", re.compile(r"(address already in use|bind|listen|port \d+)", re.IGNORECASE)),
    ("gpu", re.compile(r"(cuda|rocm|hip|metal|vulkan|gpu|vram)", re.IGNORECASE)),
    ("model", re.compile(r"(gguf|model load|tokenizer|tensor|layer package)", re.IGNORECASE)),
    ("skippy", re.compile(r"(skippy|stage|activation|kv cache)", re.IGNORECASE)),
    ("network", re.compile(r"(quic|relay|join|peer|mesh|connection refused)", re.IGNORECASE)),
    ("plugin", re.compile(r"(plugin|mcp|command not found)", re.IGNORECASE)),
]

MAX_LOG_BYTES = 256_000
MAX_FINDINGS = 80
MAX_LINE_CHARS = 240


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="mesh-llm doctor zip path")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    try:
        summary = summarize_bundle(args.bundle)
    except Exception as exc:  # noqa: BLE001 - support helper should be direct.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_markdown(summary)
    return 0


def summarize_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a zip file: {path}")

    with zipfile.ZipFile(path) as bundle:
        names = sorted(info.filename for info in bundle.infolist() if not info.is_dir())
        json_files = {
            name: read_json(bundle, name)
            for name in names
            if name.endswith(".json") and bundle.getinfo(name).file_size <= 20_000_000
        }
        logs = [name for name in names if name.startswith("runtime/logs/")]
        return {
            "bundle": str(path),
            "files": names,
            "file_count": len(names),
            "manifest": manifest_summary(json_files.get("manifest.json")),
            "system": system_summary(json_files.get("system.json")),
            "gpus": gpu_summary(json_files.get("gpus.json")),
            "plugins": plugin_summary(json_files.get("plugins.json")),
            "runtime": runtime_summary(json_files),
            "api": api_summary(json_files),
            "log_findings": scan_logs(bundle, logs),
        }


def read_json(bundle: zipfile.ZipFile, name: str) -> Any:
    try:
        return json.loads(bundle.read(name).decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001 - keep corrupt JSON visible.
        return {"_parse_error": str(exc)}


def manifest_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": False}
    selected = value.get("selected_instance") or {}
    return {
        "present": True,
        "generated_at": value.get("generated_at"),
        "target": value.get("target"),
        "warnings": value.get("warnings") or [],
        "selected_pid": selected.get("pid"),
        "selected_live": selected.get("is_live"),
        "selected_api_port": selected.get("api_port"),
        "included_files": len(value.get("included_files") or []),
    }


def system_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": False}
    mesh = value.get("mesh_llm") or {}
    platform = value.get("platform") or {}
    flavor = value.get("flavor") or {}
    system = value.get("system") or {}
    memory = system.get("memory") or {}
    cpu = system.get("cpu") or {}
    return {
        "present": True,
        "version": mesh.get("version"),
        "current_exe": mesh.get("current_exe"),
        "current_dir": mesh.get("current_dir"),
        "os": platform.get("os"),
        "arch": platform.get("arch"),
        "requested_flavor": flavor.get("requested_llama_flavor"),
        "detected_flavor": flavor.get("detected_host_flavor"),
        "memory_total_gb": bytes_to_gb(memory.get("total_bytes")),
        "memory_available_gb": bytes_to_gb(memory.get("available_bytes")),
        "cpu_logical": cpu.get("logical_count"),
        "cpu_physical": cpu.get("physical_count"),
        "cpu_brand": cpu.get("brand"),
        "api_port": (value.get("runtime") or {}).get("api_port"),
    }


def gpu_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": False, "count": 0, "devices": []}
    devices = value.get("gpus") or value.get("devices") or []
    if not isinstance(devices, list):
        devices = []
    return {
        "present": True,
        "count": len(devices),
        "devices": [compact_gpu(device) for device in devices[:16]],
    }


def compact_gpu(device: Any) -> dict[str, Any]:
    if not isinstance(device, dict):
        return {"raw": device}
    return {
        key: device.get(key)
        for key in [
            "name",
            "backend_device",
            "vendor",
            "total_vram_bytes",
            "free_vram_bytes",
            "memory_total_bytes",
            "memory_available_bytes",
        ]
        if key in device
    }


def plugin_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"present": False}
    installed = value.get("installed") or []
    resolved = value.get("resolved") or {}
    return {
        "present": True,
        "installed": compact_plugin_list(installed),
        "active_runtime": compact_plugin_list(resolved.get("active") or []),
        "inactive_runtime": compact_plugin_list(resolved.get("inactive") or []),
    }


def compact_plugin_list(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    compacted = []
    for item in items[:64]:
        if isinstance(item, dict):
            compacted.append(
                {
                    key: item.get(key)
                    for key in ["name", "version", "enabled", "kind", "command", "url", "env_keys"]
                    if key in item
                }
            )
    return compacted


def runtime_summary(json_files: dict[str, Any]) -> dict[str, Any]:
    instances = json_files.get("runtime/instances.json")
    owner = json_files.get("runtime/owner.json")
    return {
        "instances": instances if isinstance(instances, dict) else None,
        "owner": owner if isinstance(owner, dict) else None,
    }


def api_summary(json_files: dict[str, Any]) -> dict[str, Any]:
    api = {}
    for name, value in sorted(json_files.items()):
        if not name.startswith("api/") or not isinstance(value, dict):
            continue
        api[name] = {
            "ok": value.get("ok"),
            "status": value.get("status"),
            "url": value.get("url"),
            "error": value.get("error"),
            "body_type": type(value.get("body")).__name__,
        }
    return api


def scan_logs(bundle: zipfile.ZipFile, logs: list[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for name in logs:
        data = bundle.read(name)
        if len(data) > MAX_LOG_BYTES:
            data = data[-MAX_LOG_BYTES:]
        text = data.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(
                        {
                            "file": name,
                            "line": line_number,
                            "pattern": label,
                            "text": line.strip()[:MAX_LINE_CHARS],
                        }
                    )
                    break
            if len(findings) >= MAX_FINDINGS:
                return findings
    return findings


def bytes_to_gb(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 1024 / 1024 / 1024, 2)


def print_markdown(summary: dict[str, Any]) -> None:
    print(f"# mesh-llm doctor bundle summary\n")
    print(f"Bundle: `{summary['bundle']}`")
    print(f"Files: {summary['file_count']}")

    manifest = summary["manifest"]
    if manifest.get("present"):
        print(f"Generated: {manifest.get('generated_at')}")
        print(f"Selected runtime: pid={manifest.get('selected_pid')} live={manifest.get('selected_live')}")
        warnings = manifest.get("warnings") or []
        if warnings:
            print("\n## Manifest warnings")
            for warning in warnings:
                print(f"- {warning}")

    print_section("System", summary["system"])
    print_section("GPUs", summary["gpus"])
    print_section("Plugins", summary["plugins"])
    print_section("API", summary["api"])

    findings = summary["log_findings"]
    if findings:
        print("\n## Log findings")
        for item in findings[:MAX_FINDINGS]:
            print(
                f"- `{item['file']}:{item['line']}` [{item['pattern']}]: {item['text']}"
            )
    else:
        print("\n## Log findings\n- No high-signal log patterns found.")


def print_section(title: str, value: Any) -> None:
    print(f"\n## {title}")
    print("```json")
    print(json.dumps(value, indent=2, sort_keys=True))
    print("```")


if __name__ == "__main__":
    raise SystemExit(main())
