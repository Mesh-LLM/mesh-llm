#!/usr/bin/env python3
"""Compare native skippy-quantize output with llama.cpp reference tools.

The script has two independent checks:

* conversion: convert a SafeTensors checkpoint with upstream
  convert_hf_to_gguf.py and with `skippy-quantize convert`, then compare GGUF
  tensor names, shapes, types, and payload bytes.
* quantization: quantize a BF16/FP16 GGUF with standalone `llama-quantize` and
  with `skippy-quantize quantize --backend llama-api` for every requested quant
  mode, then require byte-identical split outputs.

It intentionally records failures instead of stopping at the first unsupported
mode so a full quant catalog run produces actionable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    log: str


@dataclass
class ConversionResult:
    status: str
    python_gguf: str | None
    skippy_gguf: str | None
    tensor_order_equal: bool | None
    tensor_name_set_equal: bool | None
    tensor_mismatch_count: int | None
    byte_equal: bool | None
    error: str | None


@dataclass
class QuantResult:
    quant: str
    status: str
    reference_output: str | None
    skippy_output: str | None
    reference_sha256: str | None
    skippy_sha256: str | None
    byte_equal: bool | None
    reference_returncode: int | None
    skippy_returncode: int | None
    error: str | None


def main() -> int:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "conversion": None,
        "quantization": [],
    }

    if args.checkpoint is not None:
        report["conversion"] = asdict(run_conversion(args, work_dir))

    quant_input = args.quant_input
    if quant_input is None and report["conversion"] is not None:
        conversion = report["conversion"]
        if conversion["status"] == "ok":
            quant_input = Path(conversion["skippy_gguf"])

    if quant_input is not None and not args.skip_quantization:
        for quant in selected_quants(args):
            result = run_quant(args, work_dir, quant, quant_input.resolve())
            report["quantization"].append(asdict(result))

    report_path = work_dir / "comparison-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"comparison_report={report_path}")

    return 0 if report_passed(report, args.allow_matching_failures) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--quant-input", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--skippy-quantize", type=Path, required=True)
    parser.add_argument("--llama-quantize", type=Path, required=True)
    parser.add_argument("--python-converter", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-type", default="bf16")
    parser.add_argument("--nthreads", default="8")
    parser.add_argument("--quant", action="append")
    parser.add_argument("--skip-quant", action="append", default=[])
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument(
        "--generate-imatrix",
        action="store_true",
        help="Generate a deterministic all-ones legacy imatrix from --quant-input.",
    )
    parser.add_argument("--native-runtime-library", action="append", type=Path, default=[])
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--allow-matching-failures",
        action="store_true",
        help="Treat modes where both reference and native fail as non-fatal.",
    )
    return parser.parse_args()


def run_conversion(args: argparse.Namespace, work_dir: Path) -> ConversionResult:
    if args.python_converter is None:
        return ConversionResult(
            status="skipped",
            python_gguf=None,
            skippy_gguf=None,
            tensor_order_equal=None,
            tensor_name_set_equal=None,
            tensor_mismatch_count=None,
            byte_equal=None,
            error="--python-converter not provided",
        )

    conversion_dir = work_dir / "conversion"
    python_dir = conversion_dir / "python"
    skippy_dir = conversion_dir / "skippy"
    python_dir.mkdir(parents=True, exist_ok=True)
    skippy_dir.mkdir(parents=True, exist_ok=True)

    basename = args.checkpoint.name
    python_gguf = python_dir / f"{basename}-{args.output_type}.gguf"
    skippy_gguf = skippy_dir / f"{basename}-{args.output_type}.gguf"

    python_cmd = [
        args.python,
        str(args.python_converter),
        "--outtype",
        args.output_type,
        "--outfile",
        str(python_gguf),
        str(args.checkpoint),
    ]
    result = run_logged(python_cmd, conversion_dir / "python-convert.log")
    if result.returncode != 0:
        return conversion_error("python_conversion_failed", python_gguf, skippy_gguf, result)

    skippy_cmd = [
        str(args.skippy_quantize),
        "convert",
        "--output-type",
        args.output_type,
        "--expected-splits",
        "1",
        "--no-verify-on-complete",
        str(args.checkpoint),
        str(skippy_gguf),
    ]
    result = run_logged(skippy_cmd, conversion_dir / "skippy-convert.log")
    if result.returncode != 0:
        return conversion_error("skippy_conversion_failed", python_gguf, skippy_gguf, result)

    comparison = compare_gguf_tensors(python_gguf, skippy_gguf)
    return ConversionResult(
        status="ok" if comparison["tensor_mismatch_count"] == 0 else "tensor_mismatch",
        python_gguf=str(python_gguf),
        skippy_gguf=str(skippy_gguf),
        tensor_order_equal=comparison["tensor_order_equal"],
        tensor_name_set_equal=comparison["tensor_name_set_equal"],
        tensor_mismatch_count=comparison["tensor_mismatch_count"],
        byte_equal=same_file_bytes(python_gguf, skippy_gguf),
        error=None,
    )


def conversion_error(
    status: str, python_gguf: Path, skippy_gguf: Path, result: CommandResult
) -> ConversionResult:
    return ConversionResult(
        status=status,
        python_gguf=str(python_gguf),
        skippy_gguf=str(skippy_gguf),
        tensor_order_equal=None,
        tensor_name_set_equal=None,
        tensor_mismatch_count=None,
        byte_equal=None,
        error=f"returncode={result.returncode} log={result.log}",
    )


def selected_quants(args: argparse.Namespace) -> list[str]:
    if args.quant:
        names = args.quant
    else:
        proc = subprocess.run(
            [str(args.skippy_quantize), "list-quants", "--json"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        names = json.loads(proc.stdout)["whole_model_quant_modes"]
    skipped = set(args.skip_quant)
    return [name for name in names if name not in skipped]


def run_quant(
    args: argparse.Namespace, work_dir: Path, quant: str, quant_input: Path
) -> QuantResult:
    quant_dir = work_dir / "quant" / safe_name(quant)
    ref_dir = quant_dir / "reference"
    skippy_dir = quant_dir / "skippy"
    ref_dir.mkdir(parents=True, exist_ok=True)
    skippy_dir.mkdir(parents=True, exist_ok=True)

    reference_prefix = ref_dir / f"model-{safe_name(quant)}.gguf"
    skippy_prefix = skippy_dir / f"model-{safe_name(quant)}.gguf"
    imatrix = imatrix_path(args, work_dir, quant_input)
    reference_cmd = [
        str(args.llama_quantize),
        "--keep-split",
        "--first-split",
        "1",
        "--last-split",
        "1",
    ]
    if imatrix is not None:
        reference_cmd.extend(["--imatrix", str(imatrix)])
    reference_cmd.extend([
        str(quant_input),
        str(reference_prefix),
        quant,
        str(args.nthreads),
    ])
    reference = run_logged(reference_cmd, quant_dir / "reference.log")

    skippy_cmd = [
        str(args.skippy_quantize),
        "quantize",
        "--backend",
        "llama-api",
        "--no-stage-source",
        "--no-verify-on-complete",
        "--work-dir",
        str(quant_dir / "work"),
    ]
    for library in args.native_runtime_library:
        skippy_cmd.extend(["--native-runtime-library", str(library)])
    if imatrix is not None:
        skippy_cmd.extend(["--imatrix", str(imatrix)])
    skippy_cmd.extend([str(quant_input), str(skippy_prefix), quant, str(args.nthreads)])
    skippy = run_logged(skippy_cmd, quant_dir / "skippy.log")

    ref_output = split_output_path(reference_prefix)
    skippy_output = split_output_path(strip_gguf_suffix(skippy_prefix))

    if reference.returncode != 0 or skippy.returncode != 0:
        status = "matching_failure" if reference.returncode == skippy.returncode else "failure"
        return QuantResult(
            quant=quant,
            status=status,
            reference_output=str(ref_output) if ref_output.exists() else None,
            skippy_output=str(skippy_output) if skippy_output.exists() else None,
            reference_sha256=None,
            skippy_sha256=None,
            byte_equal=None,
            reference_returncode=reference.returncode,
            skippy_returncode=skippy.returncode,
            error=f"reference_log={reference.log} skippy_log={skippy.log}",
        )

    if not ref_output.exists() or not skippy_output.exists():
        return QuantResult(
            quant=quant,
            status="missing_output",
            reference_output=str(ref_output),
            skippy_output=str(skippy_output),
            reference_sha256=None,
            skippy_sha256=None,
            byte_equal=None,
            reference_returncode=reference.returncode,
            skippy_returncode=skippy.returncode,
            error="expected split output missing",
        )

    ref_sha = sha256(ref_output)
    skippy_sha = sha256(skippy_output)
    byte_equal = ref_sha == skippy_sha and same_file_bytes(ref_output, skippy_output)
    return QuantResult(
        quant=quant,
        status="ok" if byte_equal else "byte_mismatch",
        reference_output=str(ref_output),
        skippy_output=str(skippy_output),
        reference_sha256=ref_sha,
        skippy_sha256=skippy_sha,
        byte_equal=byte_equal,
        reference_returncode=reference.returncode,
        skippy_returncode=skippy.returncode,
        error=None,
    )


def run_logged(argv: list[str], log_path: Path) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            argv,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    return CommandResult(argv=argv, returncode=proc.returncode, log=str(log_path))


def compare_gguf_tensors(left: Path, right: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from gguf import GGUFReader
    except Exception as exc:  # pragma: no cover - dependency error path
        raise SystemExit(f"install gguf and numpy for tensor comparison: {exc}") from exc

    left_reader = GGUFReader(left)
    right_reader = GGUFReader(right)
    left_tensors = {tensor.name: tensor for tensor in left_reader.tensors}
    right_tensors = {tensor.name: tensor for tensor in right_reader.tensors}
    tensor_order_equal = list(left_tensors) == list(right_tensors)
    tensor_name_set_equal = set(left_tensors) == set(right_tensors)
    mismatches = []
    for name, left_tensor in left_tensors.items():
        right_tensor = right_tensors.get(name)
        if right_tensor is None:
            mismatches.append(name)
            continue
        same_shape = left_tensor.shape.tolist() == right_tensor.shape.tolist()
        same_type = left_tensor.tensor_type == right_tensor.tensor_type
        same_data = np.array_equal(left_tensor.data, right_tensor.data)
        if not (same_shape and same_type and same_data):
            mismatches.append(name)
    for name in right_tensors:
        if name not in left_tensors:
            mismatches.append(name)
    return {
        "tensor_order_equal": tensor_order_equal,
        "tensor_name_set_equal": tensor_name_set_equal,
        "tensor_mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
    }


def report_passed(report: dict[str, Any], allow_matching_failures: bool) -> bool:
    conversion = report.get("conversion")
    if conversion is not None and conversion["status"] not in ("ok", "skipped"):
        return False
    for result in report["quantization"]:
        if result["status"] == "ok":
            continue
        if allow_matching_failures and result["status"] == "matching_failure":
            continue
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_file_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def imatrix_path(args: argparse.Namespace, work_dir: Path, quant_input: Path) -> Path | None:
    if args.imatrix is not None:
        return args.imatrix.resolve()
    if not args.generate_imatrix:
        return None
    path = work_dir / "generated-imatrix.dat"
    if not path.exists():
        write_legacy_all_ones_imatrix(quant_input, path)
    return path


def write_legacy_all_ones_imatrix(quant_input: Path, output: Path) -> None:
    try:
        from gguf import GGUFReader
    except Exception as exc:  # pragma: no cover - dependency error path
        raise SystemExit(f"install gguf to generate imatrix fixture: {exc}") from exc

    reader = GGUFReader(quant_input)
    entries: list[tuple[str, int]] = []
    for tensor in reader.tensors:
        shape = tensor.shape.tolist()
        if len(shape) < 2:
            continue
        imatrix_width = int(shape[0])
        if len(shape) >= 3:
            imatrix_width *= int(shape[2])
        entries.append((tensor.name, imatrix_width))

    if not entries:
        raise SystemExit(f"no rank >= 2 tensors found for imatrix in {quant_input}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<i", len(entries)))
        for name, width in entries:
            encoded = name.encode("utf-8")
            handle.write(struct.pack("<i", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<i", 1))
            handle.write(struct.pack("<i", width))
            handle.write(struct.pack(f"<{width}f", *([1.0] * width)))
        dataset = b"deterministic-all-ones"
        handle.write(struct.pack("<i", 1))
        handle.write(struct.pack("<i", len(dataset)))
        handle.write(dataset)


def split_output_path(prefix_or_path: Path) -> Path:
    prefix = strip_gguf_suffix(prefix_or_path)
    return prefix.with_name(f"{prefix.name}-00001-of-00001.gguf")


def strip_gguf_suffix(path: Path) -> Path:
    if path.suffix == ".gguf":
        return path.with_suffix("")
    return path


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    raise SystemExit(main())
