#!/usr/bin/env python3
"""Exact localhost ports required by a resolved family-certification shard."""
from __future__ import annotations

import csv
from pathlib import Path
import sys


CAUSAL_OFFSETS = (1, 11, 12, 31, 32)
MODEL_CLASSES = {
    "causal_generation",
    "embedding",
    "rerank",
    "encoder_decoder",
    "ocr",
    "speech_synthesis",
    "speech_recognition",
}
SERVER_ORACLE_CLASSES = {"embedding", "rerank", "ocr", "speech_recognition"}


def valid_port(value: int, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError(f"invalid {name}")
    return value


def required_ports(path: Path, candidate_port: int, oracle_port: int) -> list[int]:
    candidate_port = valid_port(candidate_port, "workload candidate port")
    oracle_port = valid_port(oracle_port, "workload oracle port")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(), delimiter="|"))
    if not rows or len(rows[0]) < 2 or rows[0][:2] != ["family", "class"]:
        raise ValueError("resolved family manifest has an invalid header")
    ports: set[int] = set()
    total = 0
    for row in rows[1:]:
        if len(row) < 2 or not row[0]:
            raise ValueError("resolved family manifest has an invalid row")
        model_class = row[1]
        if model_class not in MODEL_CLASSES:
            raise ValueError(f"resolved family manifest has unknown class: {model_class}")
        total += 1
        if model_class == "causal_generation":
            port_base = 19000 + ((total - 1) % 20) * 50
            ports.update(port_base + offset for offset in CAUSAL_OFFSETS)
        else:
            ports.add(candidate_port)
            if model_class in SERVER_ORACLE_CLASSES:
                if candidate_port == oracle_port:
                    raise ValueError("workload candidate and oracle ports conflict")
                ports.add(oracle_port)
    if not ports:
        raise ValueError("resolved family manifest has no certification rows")
    return sorted(ports)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: canary_family_ports.py RESOLVED_MANIFEST CANDIDATE_PORT ORACLE_PORT"
        )
    try:
        ports = required_ports(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(",".join(str(port) for port in ports))


if __name__ == "__main__":
    main()
