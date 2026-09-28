#!/usr/bin/env python3
"""Compare Laya System One answers with the upstream PyTorch golden fixtures.

Two sources, one comparison:

* ``--base-url URL --model ID`` posts each fixture to a mesh ``/systemone``
  endpoint serving a converted ``laya-multilingual`` GGUF.
* ``--cli PATH --gguf PATH`` runs a ``llama-laya-cli`` build on each fixture.
  The CLI also reports token ids, which are checked for an exact match.

Each fixture may differ from its golden by the error upstream's own CPU
runtime shows on it (``UPSTREAM_CPU_ERROR``, measured on the same GGUF) plus
``ALLOWANCE``. The goldens live in ``ci/llama-canary/fixtures/laya-golden``.

Exit status: 0 when every fixture is within its allowance, 1 when one is not,
2 for a usage or environment error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES = ROOT / "ci" / "llama-canary" / "fixtures" / "laya-golden"

# Max |p - golden| of upstream ggml-org/llama.cpp#29363 on CPU, per fixture,
# with laya-multilingual converted to F16 (jianyang job_cryr7vn4pwrvf0s).
UPSTREAM_CPU_ERROR = {
    "choice_single_zh": 0.0039,
    "choice_multi_zh": 0.0087,
    "score_zh": 0.0021,
    "noul_zh": 0.0579,
    "choice_single_en": 0.0027,
    "score_en": 0.0131,
    "noul_en": 0.0012,
}
ALLOWANCE = 0.005


def read_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def answer_probabilities(answer: dict[str, Any]) -> dict[str, float]:
    """The probabilities a Jev answer reports, keyed by option."""
    if "noul" in answer:
        return {"true": float(answer["noul"])}
    return {key: float(value) for key, value in answer["probabilities"].items()}


def max_probability_error(got: dict[str, Any], want: dict[str, Any]) -> float:
    """Largest probability difference across every question and option."""
    worst = 0.0
    for key, expected in want.items():
        actual = answer_probabilities(got[key])
        for option, probability in answer_probabilities(expected).items():
            worst = max(worst, abs(actual[option] - probability))
    return worst


def compare(name: str, golden: dict[str, Any], answers: dict[str, Any],
            input_ids: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Judge one fixture's answers (and optional token ids) against its golden."""
    result: dict[str, Any] = {"fixture": name, "failures": []}
    error = max_probability_error(answers, golden["answers"])
    allowed = UPSTREAM_CPU_ERROR.get(name, 0.0) + ALLOWANCE
    result["max_abs_diff"] = round(error, 4)
    result["allowed"] = round(allowed, 4)
    if error > allowed:
        result["failures"].append(f"max |dp| {error:.4f} exceeds {allowed:.4f}")
    for key, expected in golden["answers"].items():
        if "choice" in expected and answers[key].get("choice") != expected["choice"]:
            result["failures"].append(f"{key}: choice {answers[key].get('choice')!r} != {expected['choice']!r}")
    if input_ids is not None:
        for key, question in golden["per_question"].items():
            if input_ids.get(key) != question["input_ids"]:
                result["failures"].append(f"{key}: token ids differ from the golden")
    return result


def read_via_http(base_url: str, model: str, golden: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps({"model": model, "state": golden["state"], "questions": golden["questions"]}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/systemone", body, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["answers"]


def read_via_cli(
    cli: str,
    gguf: str,
    fixture: Path,
    timeout: float,
    device: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    command = [cli, "-m", gguf, "-f", str(fixture)]
    if device:
        command.extend(("--device", device))
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=True
    )
    output = json.loads(completed.stdout)
    ids = {key: question["input_ids"] for key, question in output["per_question"].items()}
    return output["answers"], ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--base-url", help="mesh endpoint, e.g. http://127.0.0.1:9337")
    parser.add_argument("--model", help="served model id or alias, with --base-url")
    parser.add_argument("--cli", help="llama-laya-cli binary")
    parser.add_argument("--gguf", help="converted laya-multilingual GGUF, with --cli")
    parser.add_argument("--device", help="explicit llama-laya-cli backend device")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if bool(args.base_url) == bool(args.cli):
        parser.error("give exactly one of --base-url or --cli")
    if args.base_url and not args.model:
        parser.error("--base-url needs --model")
    if args.cli and not args.gguf:
        parser.error("--cli needs --gguf")
    fixtures = sorted(path for path in args.fixtures.glob("*.json") if path.stem != "manifest")
    if not fixtures:
        print(f"no fixtures under {args.fixtures}", file=sys.stderr)
        return 2

    results = []
    for path in fixtures:
        golden = read_fixture(path)
        try:
            if args.base_url:
                answers, ids = read_via_http(args.base_url, args.model, golden, args.timeout), None
            else:
                answers, ids = read_via_cli(args.cli, args.gguf, path, args.timeout, args.device)
        except (urllib.error.URLError, subprocess.SubprocessError, OSError, KeyError, ValueError) as error:
            print(f"{path.stem}: could not read: {error}", file=sys.stderr)
            return 2
        result = compare(path.stem, golden, answers, ids)
        results.append(result)
        status = "ok" if not result["failures"] else "FAIL " + "; ".join(result["failures"])
        print(f"{path.stem:18s} max|dp|={result['max_abs_diff']:.4f} (allowed {result['allowed']:.4f}) {status}")

    failed = [result for result in results if result["failures"]]
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"results": results, "passed": not failed}, indent=2),
            encoding="utf-8",
        )
    print("laya parity:", "pass" if not failed else f"{len(failed)} fixture(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
