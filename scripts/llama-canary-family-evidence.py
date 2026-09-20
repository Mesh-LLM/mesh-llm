#!/usr/bin/env python3
"""Immutable canary build handoff and fail-closed per-family aggregation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

BINS = ("skippy-correctness", "skippy-server", "skippy-model-package", "skippy-topology-plan")
CORE = {"single-step", "chain", "state-handoff"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def output(**values) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def load_planner(root: Path):
    spec = importlib.util.spec_from_file_location("family_planner", root / "scripts/plan-family-battery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plan(plan: dict) -> dict[str, dict]:
    models = {model["family"]: model for model in plan["selected_models"]}
    rows = plan["github_matrix"]["include"]
    if not 1 <= len(models) <= 256 or len(models) != len(plan["selected_models"]):
        raise ValueError("family matrix must contain 1..256 unique families")
    families = [row["families"] for row in rows]
    if len(families) != len(models) or set(families) != set(models):
        raise ValueError("exactly one matrix job is required per family")
    if len({row["shard_index"] for row in rows}) != len(rows):
        raise ValueError("duplicate shard index")
    for row in rows:
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", row["families"]):
            raise ValueError("unsafe family identity")
        shards = [s for s in plan["shards"] if s["shard_index"] == row["shard_index"]]
        if len(shards) != 1 or shards[0]["families"] != [row["families"]]:
            raise ValueError("matrix and shard membership disagree")
    if set(plan["required_certification_lanes"]) != CORE:
        raise ValueError("required lane contract changed")
    return models


def check_binary(path: Path) -> None:
    arches = subprocess.check_output(["lipo", "-archs", str(path)], text=True).strip()
    if arches != "arm64":
        raise ValueError(f"non-arm64 certification executable: {path.name}")
    # Producer binaries must be relocatable. Do not silently depend on a
    # producer-local Homebrew or native build directory on another machine.
    deps = subprocess.check_output(["otool", "-L", str(path)], text=True).splitlines()[1:]
    for line in deps:
        dep = line.strip().split(" (", 1)[0]
        if not dep.startswith(("/usr/lib/", "/System/Library/")):
            raise ValueError(f"unpackaged dynamic dependency in {path.name}: {dep}")


def build(args) -> None:
    mode = os.environ["CANARY_HARNESS_MODE"]
    pass_id = os.environ["CANARY_PASS_ID"]
    if mode not in {"repair-build", "verify-build", "pinned-build"}:
        raise ValueError("invalid distributed build mode")
    if not re.fullmatch(r"(?:repair|verify)-[1-3]", pass_id):
        raise ValueError("invalid bounded pass identity")
    root = Path.cwd()
    env = dict(os.environ)
    previous = env.get("CANARY_PREVIOUS_PACKAGE")
    if previous:
        package = Path(previous)
        identity, _ = verify_package(package, env["CANARY_PREVIOUS_IDENTITY"])
        if identity["candidate"] != env["CANARY_CANDIDATE_SHA"] or identity["base"] != git(root, "rev-parse", "HEAD"):
            raise ValueError("previous candidate/base does not match dependency outputs")
        env["CANARY_INPUT_BUNDLE"] = str(package / "candidate.bundle")
    elif mode == "verify-build":
        raise ValueError("independent verification requires a candidate")
    subprocess.run([str(root / "scripts/llama-canary-agent-repair.sh")], env=env, check=True)


def publication(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    if not identity["pass_id"].startswith("verify-") or not identity["bundle_sha256"]:
        raise ValueError("publication requires the independent verifier package")
    url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{identity['run_id']}"
    (args.package / "pr-body.md").write_text(
        "Update the llama.cpp pin and its patch queue to the independently certified candidate.\n\n"
        f"Candidate: `{identity['candidate']}`. Both complete per-family passes succeeded on this exact tree. "
        "Each pass independently rebuilt the native/Rust binaries and ran the complete roster, including "
        "single-step, chain, state-handoff, native draft requirements and applicable multimodal smokes.\n\n"
        f"Evidence: {url} ({identity['pass_id']}; {len(plan['selected_models'])} families).\n\n"
        + (args.package / "upstream-summary.md").read_text()
    )


def pack(args) -> None:
    root, dest = args.root.resolve(), args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    git(root, "diff", "--exit-code", args.candidate, "--")
    planner = load_planner(root)
    plan = planner.build_plan(root / "ci/llama-canary/family-certified.json", shard_count=256,
                              cache_root=Path(os.environ["HF_CACHE"]))
    validate_plan(plan)
    write(dest / "plan.json", plan)
    payload = dest / "payload"
    payload.mkdir()
    for name in BINS:
        source = root / "target/debug" / name
        check_binary(source)
        shutil.copy2(source, payload / name)
    tests = [row["executable"] for row in map(json.loads, args.test_build.read_text().splitlines())
             if row.get("reason") == "compiler-artifact" and row.get("executable")
             and row.get("target", {}).get("name") == "skippy_server" and row.get("profile", {}).get("test")]
    if len(tests) != 1:
        raise ValueError("expected exactly one prebuilt skippy-server library test executable")
    check_binary(Path(tests[0]))
    shutil.copy2(tests[0], payload / "skippy-mm-test")
    # Metal is embedded in the static native library. Reject nonrelocatable
    # dylibs above rather than rebuilding native code in family consumers.
    with tarfile.open(dest / "binaries.tar", "w") as archive:
        for path in sorted(payload.iterdir()):
            archive.add(path, arcname=path.name, recursive=False)
    shutil.rmtree(payload)
    shutil.copyfile(args.summary, dest / "upstream-summary.md")
    if args.bundle.is_file():
        shutil.copyfile(args.bundle, dest / "candidate.bundle")
    identity = {"schema": 1, "candidate": args.candidate, "base": args.base,
                "branch": args.branch, "pass_id": args.pass_id,
                "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                "platform": "macos-arm64-metal", "summary_sha256": sha(dest / "upstream-summary.md"), "plan_sha256": sha(dest / "plan.json"),
                "manifest_sha256": plan["manifest_sha256"], "binaries_sha256": sha(dest / "binaries.tar"),
                "bundle_sha256": sha(dest / "candidate.bundle") if (dest / "candidate.bundle").exists() else None}
    write(dest / "identity.json", identity)
    output(matrix=json.dumps(plan["github_matrix"], separators=(",", ":")),
           identity_sha256=sha(dest / "identity.json"), candidate=args.candidate, branch=args.branch)


def verify_package(directory: Path, expected: str) -> tuple[dict, dict]:
    if sha(directory / "identity.json") != expected:
        raise ValueError("build identity digest mismatch")
    identity, plan = read(directory / "identity.json"), read(directory / "plan.json")
    if identity["schema"] != 1 or identity["platform"] != "macos-arm64-metal":
        raise ValueError("unknown build identity")
    for key in ("candidate", "base"):
        if not re.fullmatch(r"[0-9a-f]{40}", identity[key]):
            raise ValueError("invalid source identity")
    for env, key in (("GITHUB_RUN_ID", "run_id"), ("GITHUB_RUN_ATTEMPT", "run_attempt")):
        if identity[key] != os.environ[env]:
            raise ValueError("foreign workflow run or attempt")
    for name, key in (("plan.json", "plan_sha256"), ("binaries.tar", "binaries_sha256")):
        if sha(directory / name) != identity[key]:
            raise ValueError(f"{name} digest mismatch")
    if identity.get("summary_sha256") and sha(directory / "upstream-summary.md") != identity["summary_sha256"]:
        raise ValueError("upstream summary digest mismatch")
    if identity["bundle_sha256"] and sha(directory / "candidate.bundle") != identity["bundle_sha256"]:
        raise ValueError("candidate bundle digest mismatch")
    validate_plan(plan)
    return identity, plan


def restore(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    root = args.root.resolve()
    if git(root, "rev-parse", "HEAD") != identity["base"]:
        raise ValueError("consumer checkout differs from frozen trusted base")
    if identity["candidate"] != identity["base"]:
        git(root, "bundle", "verify", str(args.package / "candidate.bundle"))
        git(root, "fetch", str(args.package / "candidate.bundle"), identity["branch"])
        if git(root, "rev-parse", "FETCH_HEAD") != identity["candidate"]:
            raise ValueError("bundle head mismatch")
        if git(root, "rev-parse", identity["candidate"] + "^") != identity["base"]:
            raise ValueError("candidate is not a direct child of frozen base")
        protected = git(root, "diff", "--name-only", identity["base"], identity["candidate"], "--",
                        ".github", ".agents", "scripts", ".gitattributes", "ci/ci.md", "ci/llama-canary/agent-repair-prompt.md")
        if protected:
            raise ValueError("candidate modified trusted orchestration")
        git(root, "checkout", "--detach", identity["candidate"])
    if sha(root / "ci/llama-canary/family-certified.json") != identity["manifest_sha256"]:
        raise ValueError("candidate manifest mismatch")
    binary_dir = root / "target/debug"
    binary_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.package / "binaries.tar") as archive:
        members = archive.getmembers()
        if {m.name for m in members} != {*BINS, "skippy-mm-test"} or len(members) != len(BINS) + 1:
            raise ValueError("incomplete or unexpected executable set")
        for member in members:
            if not member.isfile() or member.name != Path(member.name).name:
                raise ValueError("unsafe executable archive")
            target = binary_dir / member.name
            if target.is_symlink():
                target.unlink()
            with archive.extractfile(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            target.chmod(0o755)
    output(candidate=identity["candidate"])


def receipt(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    models = validate_plan(plan)
    if args.family not in models:
        raise ValueError("unplanned family")
    args.evidence.mkdir(parents=True, exist_ok=True)
    path = args.evidence / "results.jsonl"
    write(args.evidence / "receipt.json", {"identity_sha256": args.identity, "family": args.family,
          "candidate": identity["candidate"], "pass_id": identity["pass_id"],
          "runner": os.environ.get("RUNNER_NAME", "unknown"), "outcome": args.outcome,
          "results_sha256": sha(path) if path.is_file() else None})


def validate_results(path: Path, family: str, model: dict) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or any(row.get("family") != family or row.get("exit_code") != 0 for row in rows):
        raise ValueError(f"{family}: missing, foreign, or failed results")
    core_rows = [row for row in rows if row.get("split_layer") is not None]
    # The battery runs one consolidated certification per family and itself
    # reconciles product-selected cuts and native draft requirements.
    if len(core_rows) != 1:
        raise ValueError(f"{family}: expected one consolidated certification")
    outcomes = core_rows[0]["outcomes"]
    for lane in CORE:
        matches = [item for item in outcomes if item.get("name") == lane]
        if len(matches) != 1 or matches[0].get("status") != "pass" or matches[0].get("exit_code") != 0:
            raise ValueError(f"{family}: required lane {lane} incomplete")
    mm = [row for row in rows if row.get("mmproj_smoke")]
    if len(mm) != int(bool(model.get("mmproj_artifact"))):
        raise ValueError(f"{family}: multimodal evidence incomplete")


def aggregate(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    models = validate_plan(plan)
    receipts = sorted(args.evidence.glob("*/receipt.json"))
    seen = set()
    errors = []
    passed = []
    for path in receipts:
        family = path.parent.name
        try:
            item = read(path)
            family = item["family"]
            if family not in models or family in seen:
                raise ValueError("duplicate or unplanned family receipt")
            seen.add(family)
            if (item["identity_sha256"] != args.identity or item["candidate"] != identity["candidate"]
                    or item["pass_id"] != identity["pass_id"] or item["outcome"] != "success"):
                raise ValueError("failed or mismatched worker receipt "
                                 f"(runner={item.get('runner', 'unknown')}, outcome={item.get('outcome', 'unknown')})")
            results = path.parent / "results.jsonl"
            if sha(results) != item["results_sha256"]:
                raise ValueError("worker results digest mismatch")
            validate_results(results, family, models[family])
            passed.append(family)
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(f"{family}: {error}")
    if seen != set(models):
        errors.append(f"missing family receipts: {sorted(set(models) - seen)}")
    report = (f"Canary {identity['pass_id']}: {len(passed)}/{len(models)} family receipts passed "
              f"for {identity['candidate']}\n")
    if errors:
        report += "\n" + "\n".join(f"- {error}" for error in errors) + "\n"
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(report + "\n")
    if errors:
        raise ValueError("family aggregation failed:\n" + "\n".join(errors))
    output(green="true", candidate=identity["candidate"], branch=identity["branch"])
    print(f"All {len(seen)} families passed for {identity['candidate']} ({identity['pass_id']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("pack")
    for name in ("root", "output", "test-build", "bundle", "summary"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("candidate", "base", "branch", "pass-id"):
        p.add_argument("--" + name, required=True)
    subs.add_parser("build")
    for command in ("restore", "receipt", "aggregate", "publication"):
        p = subs.add_parser(command)
        p.add_argument("--package", type=Path, required=True)
        p.add_argument("--identity", required=True)
        if command == "restore":
            p.add_argument("--root", type=Path, required=True)
        elif command != "publication":
            p.add_argument("--evidence", type=Path, required=True)
        if command == "receipt":
            p.add_argument("--family", required=True)
            p.add_argument("--outcome", required=True, choices=("success", "failure", "cancelled", "skipped"))
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
