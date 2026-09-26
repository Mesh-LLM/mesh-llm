#!/usr/bin/env python3
"""Immutable canary build handoff and fail-closed per-family aggregation."""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import tempfile
import sys

_MEMORY_SPEC = importlib.util.spec_from_file_location(
    "canary_family_memory", Path(__file__).resolve().parent / "lib/canary_family_memory.py")
MEMORY = importlib.util.module_from_spec(_MEMORY_SPEC)
_MEMORY_SPEC.loader.exec_module(MEMORY)

BINS = ("skippy-correctness", "skippy-server", "skippy-model-package", "skippy-topology-plan")
CORE = {"single-step", "chain", "state-handoff"}
# The workload oracle closure ships in the handoff so family workers consume
# the CPU candidate and native oracle executables without compiling. Paths are
# the producer manifest's canonical relatives; workers export them as
# SKIPPY_WORKLOAD_* below this restore root (see llama-canary-family-pass.yml).
WORKLOAD_CLOSURE_ROOT = ".deps/canary-workload-oracles"
WORKLOAD_ORACLES_TAR = "workload-oracles.tar"
LLAMA_BUNDLE = "llama-source.bundle"
LLAMA_PROVENANCE = "llama-source.json"
LLAMA_MARKERS = (".mesh-llm-upstream-sha", ".mesh-llm-patch-digest",
                 ".mesh-llm-patched-sha", ".mesh-llm-prepare-schema")
WORKLOAD_ORACLE_CANONICAL = (
    "native/bin/llama-server",
    "native/bin/llama-completion",
    "native/bin/llama-tts",
    "cargo/debug/skippy-server",
)


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


def source_plan(root: Path, destination: Path, *, check_cache: bool = False) -> dict:
    """The selected battery's planner owns canonical plan bytes and source paths."""
    command = [sys.executable, str(root / "scripts/plan-family-battery.py"),
               "--manifest", str(root / "ci/llama-canary/family-certified.json")]
    generate = [*command, "--shard-count", "256", "--output", str(destination)]
    if check_cache:
        generate += ["--check-cache", "--cache-root", os.environ["HF_CACHE"]]
    subprocess.run(generate, cwd=root, check=True, timeout=600)
    subprocess.run([*command, "--verify-plan", str(destination)], cwd=root, check=True, timeout=60)
    plan = read(destination)
    validate_plan(plan)
    return plan


def scheduling_matrix(plan: dict) -> dict:
    # Scheduling is controller policy, never part of a historical source plan.
    models = validate_plan(plan)
    rows = [{**row, **MEMORY.placement(models[row["families"]])}
            for row in plan["github_matrix"]["include"]]
    return {"include": sorted(rows, key=lambda row: (row["estimated_work_bytes"], row["families"]))}


def certify(args) -> None:
    _, plan = verify_package(args.package, args.identity)
    models = validate_plan(plan)
    rows = [row for row in plan["github_matrix"]["include"] if row["shard_index"] == args.shard_index]
    if len(rows) != 1:
        raise ValueError("unplanned family shard")
    model = models[rows[0]["families"]]
    command = ["arch", "-arm64", str(args.root.resolve() / "scripts/skippy-family-battery.sh"),
               "--skip-build", "--plan", str(args.package.resolve() / "plan.json"),
               "--shard-index", str(args.shard_index)]
    result = MEMORY.guarded_run(model, args.memory_tier, command, args.evidence, cwd=args.root.resolve())
    if result:
        raise SystemExit(result if result > 0 else 1)


def preflight_battery(root: Path) -> None:
    """Exercise the actual selected consumer before building or fanning out."""
    print(f"Checking selected battery/plan contract: {root}", flush=True)
    with tempfile.TemporaryDirectory(prefix="canary-contract-") as directory:
        temporary = Path(directory)
        plan = source_plan(root, temporary / "plan.json")
        scheduling_matrix(plan)
        env = dict(os.environ)
        env.pop("HF_CACHE", None)
        env.pop("SKIPPY_WORKLOAD_PRODUCER_MANIFEST", None)
        env.update(FAMILY_BATTERY_ARTIFACT_ROOT=str(temporary / "evidence"),
                   FAMILY_BATTERY_RUN_ID="contract")
        subprocess.run(["bash", str(root / "scripts/skippy-family-battery.sh"),
                        "--skip-build", "--dry-run", "--plan", str(temporary / "plan.json")],
                       cwd=root, env=env, check=True, timeout=120,
                       stdout=subprocess.DEVNULL)


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
    root = Path(os.environ.get("CANARY_SOURCE_ROOT", Path.cwd()))
    env = dict(os.environ)
    selected = env.get("CANARY_MESH_SOURCE")
    if selected and (mode != "pinned-build" or env.get("CANARY_PREVIOUS_PACKAGE")
                     or git(root, "rev-parse", "HEAD") != selected):
        raise ValueError("selected source requires an unchanged pinned-build checkout")
    previous = env.get("CANARY_PREVIOUS_PACKAGE")
    if previous:
        package = Path(previous)
        identity, _ = verify_package(package, env["CANARY_PREVIOUS_IDENTITY"])
        if identity["candidate"] != env["CANARY_CANDIDATE_SHA"] or identity["base"] != git(root, "rev-parse", "HEAD"):
            raise ValueError("previous candidate/base does not match dependency outputs")
        env["CANARY_INPUT_BUNDLE"] = str(package / "candidate.bundle")
        env["CANARY_CANDIDATE_BRANCH"] = identity["branch"]
    elif mode == "verify-build":
        raise ValueError("independent verification requires a candidate")
    preflight_battery(root)
    subprocess.run([str(Path(__file__).resolve().with_name("llama-canary-agent-repair.sh"))], env=env, check=True)


def publication(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    if not identity["pass_id"].startswith("verify-") or not identity["bundle_sha256"]:
        raise ValueError("publication requires the independent verifier package")
    url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{identity['run_id']}"
    (args.package / "pr-body.md").write_text(
        "Update the llama.cpp pin and its patch queue to the independently certified candidate.\n\n"
        f"Candidate: `{identity['candidate']}`. Both complete per-family passes succeeded on this exact tree. "
        "Each pass independently rebuilt the native/Rust binaries and ran the complete roster, including "
        "single-step, chain, state-handoff, native draft requirements, applicable multimodal smokes, and "
        "class-specific workload smoke/oracle lanes for the non-chat families.\n\n"
        f"Evidence: {url} ({identity['pass_id']}; {len(plan['selected_models'])} families).\n\n"
        + (args.package / "upstream-summary.md").read_text()
    )


def workload_closure_members(closure: Path) -> list[str]:
    """Validate a workload oracle closure and return its exact archive members.

    The member set is the producer manifest plus exactly the files it binds.
    Canonical oracle/candidate executables must be present at their documented
    relatives so worker environments can point SKIPPY_WORKLOAD_* at them.
    """
    manifest = closure / "producer.json"
    if not manifest.is_file():
        raise ValueError("workload oracle closure has no producer.json manifest")
    payload = read(manifest)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("unknown workload producer manifest schema")
    referenced = [record.get("path") for record in payload["files"].values()]
    for relative in (*WORKLOAD_ORACLE_CANONICAL, *(path for path in referenced if path)):
        if not isinstance(relative, str) or ".." in relative.split("/") or re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", relative
        ) is None:
            raise ValueError(f"unsafe workload closure path: {relative!r}")
        if not (closure / relative).is_file():
            raise ValueError(f"workload closure file missing: {relative}")
    if len(set(referenced)) != len(referenced):
        raise ValueError("duplicate workload closure manifest entries")
    unbound = [relative for relative in WORKLOAD_ORACLE_CANONICAL if relative not in referenced]
    if unbound:
        raise ValueError(f"workload closure manifest does not bind canonical executables: {unbound}")
    return ["producer.json", *sorted(set(referenced))]


def pack_llama_source(root: Path, destination: Path) -> None:
    source = (root / ".deps/llama.cpp").resolve()
    # The source-owned verifier checks the pin, ordered patch digest, preparation
    # schema, exact HEAD and clean tracked source before exporting it.
    head = subprocess.check_output([sys.executable, str(root / "scripts/llama-oracle-source.py")],
                                   cwd=root, text=True, timeout=60).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("invalid prepared llama source head")
    with tempfile.TemporaryDirectory(prefix="canary-llama-") as temporary:
        clone = Path(temporary) / "source"
        subprocess.run(["git", "clone", "--no-checkout", "--no-tags", "--depth=1",
                        source.as_uri(), str(clone)], check=True, timeout=120)
        if git(clone, "rev-parse", "HEAD") != head:
            raise ValueError("prepared llama source changed during export")
        git(clone, "bundle", "create", str(destination / LLAMA_BUNDLE), "HEAD")
    write(destination / LLAMA_PROVENANCE,
          {"head": head, "markers": {name: (source / name).read_text() for name in LLAMA_MARKERS}})


def restore_llama_source(root: Path, package: Path) -> None:
    provenance = read(package / LLAMA_PROVENANCE)
    head = provenance["head"]
    if not re.fullmatch(r"[0-9a-f]{40}", head) or set(provenance["markers"]) != set(LLAMA_MARKERS):
        raise ValueError("invalid prepared llama source provenance")
    target = root / ".deps/llama.cpp"
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--no-checkout",
                    str(package / LLAMA_BUNDLE), str(target)], check=True, timeout=120)
    if git(target, "rev-parse", "HEAD") != head:
        raise ValueError("prepared llama bundle head mismatch")
    # The bundle deliberately contains one source tree, not upstream history.
    (target / ".git/shallow").write_text(head + "\n")
    git(target, "-c", "core.hooksPath=/dev/null", "checkout", "--detach", head)
    for name, value in provenance["markers"].items():
        (target / name).write_text(value)
    actual = subprocess.check_output([sys.executable, str(root / "scripts/llama-oracle-source.py")],
                                     cwd=root, text=True, timeout=60).strip()
    if actual != head:
        raise ValueError("restored llama source identity mismatch")


def packaged_workload_manifest(root: Path, closure: Path, candidate: str) -> bytes:
    """Bind already verified dirty-tree outputs to the identical snapshot tree."""
    manifest = closure / "producer.json"
    payload = read(manifest)
    seal = os.environ.get("CANARY_VERIFIED_WORKLOAD_PRODUCER")
    if seal:
        if sha(manifest) != seal:
            raise ValueError("workload producer changed after snapshot verification")
        if git(root, "write-tree") != git(root, "rev-parse", candidate + "^{tree}"):
            raise ValueError("workload snapshot tree differs from candidate")
        git(root, "diff", "--exit-code", candidate, "--")
        if git(root, "ls-files", "--others", "--exclude-standard"):
            raise ValueError("untracked source after workload snapshot")
        payload["source"] = {"head": candidate, "worktree_sha256": hashlib.sha256(b"").hexdigest()}
    else:
        subprocess.run([sys.executable, str(root / "scripts/check-skippy-workload-candidate.py"),
                        "--candidate-binary", str(closure / "cargo/debug/skippy-server"),
                        "--native-build-dir", str(closure / "native"),
                        "--producer-manifest", str(manifest)], cwd=root, check=True, timeout=60)
    for record in payload["files"].values():
        if sha(closure / record["path"]) != record["sha256"]:
            raise ValueError("workload producer artifact changed during packaging")
    return (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()


def pack(args) -> None:
    root, dest = args.root.resolve(), args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    git(root, "diff", "--exit-code", args.candidate, "--")
    plan = source_plan(root, dest / "plan.json", check_cache=True)
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
    pack_llama_source(root, dest)
    closure_root = args.workload_oracles.resolve()
    closure_members = workload_closure_members(closure_root)
    for relative in WORKLOAD_ORACLE_CANONICAL:
        check_binary(closure_root / relative)
    producer_bytes = packaged_workload_manifest(root, closure_root, args.candidate)
    with tarfile.open(dest / WORKLOAD_ORACLES_TAR, "w") as archive:
        for relative in closure_members:
            if relative == "producer.json":
                member = tarfile.TarInfo(relative)
                member.size = len(producer_bytes)
                archive.addfile(member, io.BytesIO(producer_bytes))
            else:
                archive.add(closure_root / relative, arcname=relative, recursive=False)
    shutil.copyfile(args.summary, dest / "upstream-summary.md")
    if args.bundle.is_file():
        shutil.copyfile(args.bundle, dest / "candidate.bundle")
    identity = {"schema": 3, "candidate": args.candidate, "base": args.base,
                "branch": args.branch, "pass_id": args.pass_id,
                "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                "platform": "macos-arm64-metal", "summary_sha256": sha(dest / "upstream-summary.md"), "plan_sha256": sha(dest / "plan.json"),
                "manifest_sha256": plan["manifest_sha256"], "binaries_sha256": sha(dest / "binaries.tar"),
                "workload_oracles_sha256": sha(dest / WORKLOAD_ORACLES_TAR),
                "llama_bundle_sha256": sha(dest / LLAMA_BUNDLE),
                "llama_provenance_sha256": sha(dest / LLAMA_PROVENANCE),
                "bundle_sha256": sha(dest / "candidate.bundle") if (dest / "candidate.bundle").exists() else None}
    identity["controller"] = os.environ.get("CANARY_CONTROLLER_SHA", args.base)
    identity["mesh_source"] = os.environ.get("CANARY_MESH_SOURCE", "")
    if identity["mesh_source"] and (args.candidate != identity["mesh_source"] or args.base != args.candidate
                                    or git(root, "rev-parse", "HEAD") != args.candidate):
        raise ValueError("certify-only source changed during build")
    write(dest / "identity.json", identity)
    output(matrix=json.dumps(scheduling_matrix(plan), separators=(",", ":")),
           identity_sha256=sha(dest / "identity.json"), candidate=args.candidate, branch=args.branch)


def run_attempt(value) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("invalid workflow run attempt")
    return int(value)


def verify_package(directory: Path, expected: str) -> tuple[dict, dict]:
    if sha(directory / "identity.json") != expected:
        raise ValueError("build identity digest mismatch")
    identity, plan = read(directory / "identity.json"), read(directory / "plan.json")
    if identity["schema"] != 3 or identity["platform"] != "macos-arm64-metal":
        raise ValueError("unknown build identity")
    for key in ("candidate", "base"):
        if not re.fullmatch(r"[0-9a-f]{40}", identity[key]):
            raise ValueError("invalid source identity")
    controller = os.environ.get("CANARY_CONTROLLER_SHA")
    if controller and identity.get("controller") != controller:
        raise ValueError("controller revision mismatch")
    selected = os.environ.get("CANARY_MESH_SOURCE", "")
    if identity.get("mesh_source", "") != selected:
        raise ValueError("selected source identity mismatch")
    if selected and (identity["candidate"] != selected or identity["base"] != selected
                     or identity["bundle_sha256"] or identity["pass_id"] != "repair-1"):
        raise ValueError("certify-only package changed selected source")
    # Failed-job reruns retain the successful producer and its immutable digest.
    # The consumer attempt advances; the producer's provenance must not change.
    if (identity["run_id"] != os.environ["GITHUB_RUN_ID"]
            or run_attempt(identity["run_attempt"]) > run_attempt(os.environ["GITHUB_RUN_ATTEMPT"])):
        raise ValueError("foreign workflow run or attempt")
    for name, key in (("plan.json", "plan_sha256"), ("binaries.tar", "binaries_sha256"),
                      (WORKLOAD_ORACLES_TAR, "workload_oracles_sha256"),
                      (LLAMA_BUNDLE, "llama_bundle_sha256"), (LLAMA_PROVENANCE, "llama_provenance_sha256")):
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
    restore_llama_source(root, args.package)
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
    closure_root = root / WORKLOAD_CLOSURE_ROOT
    if closure_root.exists():
        shutil.rmtree(closure_root)
    closure_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.package / WORKLOAD_ORACLES_TAR) as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members) or any(not m.isfile() for m in members):
            raise ValueError("unsafe workload closure archive")
        producer_stream = archive.extractfile("producer.json")
        if producer_stream is None:
            raise ValueError("workload closure archive has no producer manifest")
        payload = json.loads(producer_stream.read())
        referenced = sorted({record["path"] for record in payload["files"].values()})
        if sorted(m.name for m in members) != sorted({"producer.json", *referenced}):
            raise ValueError("workload closure archive does not match its producer manifest")
        for member in members:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", member.name) or ".." in member.name.split("/"):
                raise ValueError("unsafe workload closure path")
        for member in members:
            target = closure_root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            target.chmod(0o755)
    # check_candidate requires every executable to postdate the stamped native
    # ABI. Extraction mtimes are host-dependent, so pin the stamp into the past
    # and every other member to the same extraction instant.
    restored = [path for path in closure_root.rglob("*") if path.is_file()]
    stamp_relative = payload["files"]["native_stamp"]["path"]
    moment = time.time()
    for path in restored:
        os.utime(path, (moment, moment))
    os.utime(closure_root / stamp_relative, (moment - 120.0, moment - 120.0))
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
          "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
          "runner": os.environ.get("RUNNER_NAME", "unknown"), "outcome": args.outcome,
          "results_sha256": sha(path) if path.is_file() else None})


def read_json_documents(path: Path) -> list[dict]:
    """Read adjacent JSON objects regardless of whether jq printed them compactly."""
    content = path.read_text()
    decoder = json.JSONDecoder()
    rows = []
    offset = 0
    while True:
        while offset < len(content) and content[offset].isspace():
            offset += 1
        if offset == len(content):
            return rows
        row, offset = decoder.raw_decode(content, offset)
        if not isinstance(row, dict):
            raise ValueError(f"{path}: expected a stream of JSON objects")
        rows.append(row)


def validate_results(path: Path, family: str, model: dict) -> None:
    rows = read_json_documents(path)
    if not rows or any(row.get("family") not in {family, "battery"} or row.get("exit_code") != 0 for row in rows):
        raise ValueError(f"{family}: missing, foreign, or failed results")
    battery_rows = [row for row in rows if row.get("family") == "battery"]
    if battery_rows:
        if len(battery_rows) != 1:
            raise ValueError(f"{family}: expected one global battery preflight")
        outcomes = battery_rows[0].get("outcomes", [])
        preflight = [item for item in outcomes if item.get("name") == "environment-preflight"]
        if (len(preflight) != 1 or preflight[0].get("status") != "pass"
                or preflight[0].get("exit_code") != 0):
            raise ValueError(f"{family}: global battery preflight incomplete")
    rows = [row for row in rows if row.get("family") == family]
    if not rows:
        raise ValueError(f"{family}: missing family-scoped results")
    if model["class"] == "causal_generation":
        core_rows = [row for row in rows if row.get("split_layer") is not None]
        # The battery runs one consolidated certification per family and itself
        # reconciles product-selected cuts and native draft requirements.
        if len(core_rows) != 1:
            raise ValueError(f"{family}: expected one consolidated certification")
        outcomes = core_rows[0]["outcomes"]
    else:
        workload_rows = [row for row in rows if row.get("workload_class") is not None]
        if len(workload_rows) != 1:
            raise ValueError(f"{family}: expected one workload certification")
        if workload_rows[0].get("workload_class") != model["class"]:
            raise ValueError(f"{family}: workload class mismatch")
        outcomes = workload_rows[0]["outcomes"]
    # Required lanes come from the plan's per-model certification contract, so
    # causal rows demand their split-parity lanes and non-chat rows demand
    # their class-specific smoke and (for workload-oracle profiles) oracle
    # lanes. A missing or failed required lane can never certify.
    for lane in model["certification_lanes"]:
        matches = [item for item in outcomes if item.get("name") == lane]
        if len(matches) != 1 or matches[0].get("status") != "pass" or matches[0].get("exit_code") != 0:
            raise ValueError(f"{family}: required lane {lane} incomplete")
    mm = [row for row in rows if row.get("mmproj_smoke")]
    # Projector smokes are consolidated into the workload lane for non-chat
    # classes; only causal families emit separate mmproj smoke rows.
    if model["class"] == "causal_generation" and len(mm) != int(bool(model.get("mmproj_artifact"))):
        raise ValueError(f"{family}: multimodal evidence incomplete")


def aggregate(args) -> None:
    identity, plan = verify_package(args.package, args.identity)
    models = validate_plan(plan)
    receipts = sorted(args.evidence.glob("*/receipt.json"))
    seen = set()
    errors = []
    passed = []
    latest = {}
    attempts = set()
    for path in receipts:
        family = path.parent.name
        try:
            item = read(path)
            family = item["family"]
            if family not in models:
                raise ValueError("unplanned family receipt")
            if (item["identity_sha256"] != args.identity or item["candidate"] != identity["candidate"]
                    or item["pass_id"] != identity["pass_id"] or item["run_id"] != identity["run_id"]):
                raise ValueError("mismatched worker receipt")
            attempt = run_attempt(item["run_attempt"])
            if not run_attempt(identity["run_attempt"]) <= attempt <= run_attempt(os.environ["GITHUB_RUN_ATTEMPT"]):
                raise ValueError("worker attempt outside producer/current bounds")
            if (family, attempt) in attempts:
                raise ValueError("duplicate family receipt in workflow attempt")
            attempts.add((family, attempt))
            if family not in latest or attempt > latest[family][0]:
                latest[family] = (attempt, path, item)
        except (ValueError, OSError, KeyError, TypeError) as error:
            errors.append(f"{family}: {error}")
    # Select before validation: a newer failure or corrupt result must never
    # silently fall back to a successful receipt from an earlier attempt.
    for family, (_, path, item) in sorted(latest.items()):
        seen.add(family)
        try:
            if item["outcome"] != "success":
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
    for name in ("root", "output", "test-build", "bundle", "summary", "workload-oracles"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("candidate", "base", "branch", "pass-id"):
        p.add_argument("--" + name, required=True)
    subs.add_parser("build")
    for command in ("restore", "receipt", "aggregate", "publication", "certify"):
        p = subs.add_parser(command)
        p.add_argument("--package", type=Path, required=True)
        p.add_argument("--identity", required=True)
        if command in {"restore", "certify"}:
            p.add_argument("--root", type=Path, required=True)
        if command not in {"restore", "publication"}:
            p.add_argument("--evidence", type=Path, required=True)
        if command == "certify":
            p.add_argument("--shard-index", type=int, required=True)
            p.add_argument("--memory-tier", required=True)
        if command == "receipt":
            p.add_argument("--family", required=True)
            p.add_argument("--outcome", required=True, choices=("success", "failure", "cancelled", "skipped"))
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
