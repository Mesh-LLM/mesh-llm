#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def invoke(
    tool: Path,
    source_root: Path,
    report: Path,
    sources: list[Path],
    *,
    apply: bool,
) -> dict:
    command = [
        str(tool),
        "--source-root",
        str(source_root),
        "--llama-commit",
        "fixture",
        "--report",
        str(report),
    ]
    if apply:
        command.append("--apply")
    command.extend(str(source) for source in sources)
    command.extend(["--", "-std=c++17"])
    subprocess.run(command, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def run(
    tool: Path,
    source_root: Path,
    report: Path,
    *,
    source_name: str,
    apply: bool = False,
) -> dict:
    return invoke(
        tool,
        source_root,
        report,
        [source_root / "src/models" / source_name],
        apply=apply,
    )


def builder(report: dict) -> dict:
    assert len(report["builders"]) == 1
    return report["builders"][0]


def edit_kinds(record: dict) -> list[str]:
    return [edit["kind"] for edit in record["edits"]]


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_rewriter.py TOOL FIXTURE_ROOT")
    tool = Path(sys.argv[1]).resolve()
    fixture_root = Path(sys.argv[2]).resolve()

    with tempfile.TemporaryDirectory(prefix="skippy-rewriter-") as temporary:
        source_root = Path(temporary) / "source"
        shutil.copytree(fixture_root, source_root)
        report_root = Path(temporary)

        conventional = builder(
            run(
                tool,
                source_root,
                report_root / "conventional.json",
                source_name="conventional.cpp",
            )
        )
        assert conventional["verdict"] == "transformable"
        assert conventional["proof"]["loop"] == {
            "end": "n_layer",
            "start": "0",
            "var": "il",
        }
        assert conventional["proof"]["activation_in"] == "inpL"
        assert conventional["proof"]["activation_out"] == "inpL"
        assert edit_kinds(conventional) == ["insert_begin_block", "insert_end_block"]

        run(
            tool,
            source_root,
            report_root / "conventional-applied.json",
            source_name="conventional.cpp",
            apply=True,
        )
        transformed = (source_root / "src/models/conventional.cpp").read_text(
            encoding="utf-8"
        )
        assert "begin_block(inpL, il);" in transformed
        assert "end_block(inpL, il);" in transformed
        assert "for (int il = 0; il < n_layer; ++il)" in transformed
        second = builder(
            run(
                tool,
                source_root,
                report_root / "conventional-second.json",
                source_name="conventional.cpp",
            )
        )
        assert second["verdict"] == "already_transformed"
        assert second["edits"] == []

        continued = builder(
            run(
                tool,
                source_root,
                report_root / "continue.json",
                source_name="continue-path.cpp",
            )
        )
        assert continued["verdict"] == "transformable"
        assert edit_kinds(continued).count("insert_end_block_before_continue") == 1
        run(
            tool,
            source_root,
            report_root / "continue-applied.json",
            source_name="continue-path.cpp",
            apply=True,
        )
        continued_source = (source_root / "src/models/continue-path.cpp").read_text(
            encoding="utf-8"
        )
        assert "end_block(inpL, il);\n      continue;" in continued_source

        unbraced_continue = builder(
            run(
                tool,
                source_root,
                report_root / "continue-unbraced.json",
                source_name="continue-unbraced.cpp",
            )
        )
        assert unbraced_continue["verdict"] == "transformable"
        assert (
            edit_kinds(unbraced_continue).count("wrap_end_block_before_continue")
            == 1
        )
        run(
            tool,
            source_root,
            report_root / "continue-unbraced-applied.json",
            source_name="continue-unbraced.cpp",
            apply=True,
        )
        unbraced_source = (
            source_root / "src/models/continue-unbraced.cpp"
        ).read_text(encoding="utf-8")
        assert (
            "if (il == 2) {\n"
            "        end_block(inpL, il);\n"
            "        continue;\n"
            "    }"
        ) in unbraced_source

        for source_name, evidence in (
            ("glm-dsa.cpp", "glm_dsa_top_k_sideband"),
            ("kimi-k3.cpp", "kimi_k3_residual_sideband"),
            ("hyperconnection.cpp", "hyperconnection_activation_frontier"),
            ("rwkv-first-value.cpp", "rwkv_first_value_sideband"),
        ):
            record = builder(
                run(
                    tool,
                    source_root,
                    report_root / f"{source_name}.json",
                    source_name=source_name,
                )
            )
            assert record["verdict"] == "transformable"
            assert evidence in record["proof"]["scope_evidence"]
            assert edit_kinds(record) == ["insert_begin_block", "insert_end_block"]

        auxiliary = builder(
            run(
                tool,
                source_root,
                report_root / "auxiliary.json",
                source_name="auxiliary.cpp",
            )
        )
        assert auxiliary["verdict"] == "supported_auxiliary"
        assert auxiliary["proof"]["execution_scope"] == "final_stage_sidecar"
        assert auxiliary["edits"] == []

        whole_model = builder(
            run(
                tool,
                source_root,
                report_root / "multiple-domains.json",
                source_name="multiple-domains.cpp",
            )
        )
        assert whole_model["verdict"] == "supported_whole_model"
        assert whole_model["edits"] == []

        for source_name, reason in (
            ("filter-only.cpp", "legacy model-local stage filter is not supported"),
            ("two-loops.cpp", "multiple equally ranked layer block loops"),
            ("nonlocal-exit.cpp", "block loop contains a non-local exit"),
            (
                "embedding-prelude-else.cpp",
                "pre-loop activation conditional has an else branch",
            ),
        ):
            record = builder(
                run(
                    tool,
                    source_root,
                    report_root / f"{source_name}.json",
                    source_name=source_name,
                )
            )
            assert record["verdict"] == "unsupported_shape"
            assert record["unsupported_reason"] == reason
            assert record["edits"] == []

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
