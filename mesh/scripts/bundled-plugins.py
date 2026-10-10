#!/usr/bin/env python3
"""Plugins bundled into each Mesh release archive, pinned in ci/bundled-plugins.json.

A default-enabled plugin ships inside the release; a node never downloads it.

  fetch   Download each pinned plugin archive from its GitHub release at the
          pinned tag, check it against that release's SHA256SUMS and against
          the pin, and verify its build-provenance attestation with the pinned
          signer workflow (gh attestation verify).
  place   Put the archives for one target into a product bundle's plugins/
          directory, with plugins/manifest.json. A target the pin lists as
          absent gets a manifest saying so and no archive.
  verify  Check every published release archive: each one for a pinned target
          carries each plugin's archive at its pinned digest; one for a target
          listed as absent carries no plugins/.

The archive is bundled exactly as the plugin's release published it (not
unpacked), so the bytes a node loads are the attested bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

DEFAULT_PINS = Path(__file__).resolve().parents[2] / "ci" / "bundled-plugins.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# The release archives' targets, as their file names spell them.
RELEASE_TARGETS = (
    "aarch64-apple-darwin",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-windows-msvc",
)
# Every release asset named like a product archive; one this file cannot read
# a target from fails verification rather than going unchecked.
PRODUCT_ARCHIVE_NAME = re.compile(r"^mesh-llm-(?!node-sdk-addon-).*\.(?:tar\.gz|zip)$")
PRODUCT_ARCHIVE = re.compile(
    r"^mesh-llm-(?:v[0-9][^-]*-)?(?P<target>" + "|".join(re.escape(t) for t in RELEASE_TARGETS) + r")(?:-[a-z0-9.-]+)?\.(?P<kind>tar\.gz|zip)$"
)


class BundleError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(path: Path) -> list[dict]:
    try:
        pins = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BundleError(f"cannot read the pin file {path}: {error}") from error
    if pins.get("schema_version") != 1 or not isinstance(pins.get("plugins"), list):
        raise BundleError(f"{path}: expected schema_version 1 and a plugins list")
    names = set()
    for plugin in pins["plugins"]:
        for key in ("name", "repository", "tag", "version", "signer_workflow", "archives"):
            if not plugin.get(key):
                raise BundleError(f"{path}: a plugin entry has no {key}")
        if plugin["name"] in names:
            raise BundleError(f"{path}: {plugin['name']} is pinned twice")
        names.add(plugin["name"])
        if plugin["tag"] != "v" + plugin["version"]:
            raise BundleError(f"{path}: {plugin['name']} tag {plugin['tag']} is not v{plugin['version']}")
        absent = plugin.get("absent", {})
        for target, digest in plugin["archives"].items():
            if target not in RELEASE_TARGETS or not SHA256.match(digest):
                raise BundleError(f"{path}: {plugin['name']} pins {target!r} with {digest!r}")
            if target in absent:
                raise BundleError(f"{path}: {plugin['name']} lists {target} as both pinned and absent")
        for target in absent:
            if target not in RELEASE_TARGETS:
                raise BundleError(f"{path}: {plugin['name']} lists an unknown absent target {target!r}")
        missing = [t for t in RELEASE_TARGETS if t not in plugin["archives"] and t not in absent]
        if missing:
            raise BundleError(f"{path}: {plugin['name']} neither pins nor lists as absent: {', '.join(missing)}")
    return pins["plugins"]


def asset_name(plugin: dict, target: str) -> str:
    return f"{plugin['name']}-{plugin['version']}-{target}.tar.gz"


def gh_binary() -> str:
    return os.environ.get("BUNDLED_PLUGINS_GH", "gh")


def fetch(pins_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for plugin in load_pins(pins_path):
        assets = {asset_name(plugin, t): (t, d) for t, d in plugin["archives"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            command = [gh_binary(), "release", "download", plugin["tag"], "-R", plugin["repository"], "-D", tmp, "-p", "SHA256SUMS"]
            for asset in sorted(assets):
                command += ["-p", asset]
            if subprocess.run(command, check=False).returncode != 0:
                raise BundleError(f"could not download {plugin['repository']} {plugin['tag']}")
            sums = {}
            for line in Path(tmp, "SHA256SUMS").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    sums[parts[1].lstrip("*")] = parts[0].lower()
            for asset, (target, pinned) in sorted(assets.items()):
                path = Path(tmp, asset)
                if not path.is_file():
                    raise BundleError(f"{plugin['tag']} has no {asset}")
                actual = sha256_file(path)
                if sums.get(asset) != actual:
                    raise BundleError(f"{asset}: SHA256SUMS says {sums.get(asset)}, the file is {actual}")
                if actual != pinned:
                    raise BundleError(f"{asset}: pinned {pinned}, the release serves {actual}")
                # The attestation must come from the pinned signer workflow,
                # run for the pinned tag on a GitHub-hosted runner.
                verify = [
                    gh_binary(), "attestation", "verify", str(path), "--repo", plugin["repository"],
                    "--signer-workflow", plugin["signer_workflow"], "--source-ref", f"refs/tags/{plugin['tag']}",
                    "--deny-self-hosted-runners",
                ]
                if subprocess.run(verify, check=False).returncode != 0:
                    raise BundleError(f"{asset}: its build-provenance attestation does not verify for {plugin['signer_workflow']}")
                destination = out / plugin["name"] / asset
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                print(f"verified {asset} {actual} ({plugin['repository']} {plugin['tag']})")


def place(pins_path: Path, source: Path, target: str, bundle: Path) -> None:
    if target not in RELEASE_TARGETS:
        raise BundleError(f"unknown release target {target!r}")
    plugins_dir = bundle / "plugins"
    placed, absent = [], []
    for plugin in load_pins(pins_path):
        if target in plugin.get("absent", {}):
            absent.append({"name": plugin["name"], "version": plugin["version"], "reason": plugin["absent"][target]})
            continue
        asset = asset_name(plugin, target)
        path = source / plugin["name"] / asset
        if not path.is_file():
            raise BundleError(f"no verified {asset} under {source}: run fetch first")
        actual = sha256_file(path)
        if actual != plugin["archives"][target]:
            raise BundleError(f"{asset}: pinned {plugin['archives'][target]}, found {actual}")
        plugins_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, plugins_dir / asset)
        placed.append({
            "name": plugin["name"], "version": plugin["version"], "repository": plugin["repository"],
            "tag": plugin["tag"], "target": target, "archive": asset, "sha256": actual,
        })
    plugins_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "target": target, "plugins": placed, "absent": absent}
    (plugins_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for entry in placed:
        print(f"bundled {entry['archive']} {entry['sha256']}")
    for entry in absent:
        print(f"not bundled for {target}: {entry['name']} ({entry['reason']})")


def archive_members(path: Path, kind: str) -> dict[str, bytes]:
    members = {}
    if kind == "tar.gz":
        with tarfile.open(path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile() and "/plugins/" in "/" + member.name:
                    handle = tf.extractfile(member)
                    members[member.name] = handle.read() if handle else b""
    else:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if "/plugins/" in "/" + name and not name.endswith("/"):
                    members[name] = zf.read(name)
    return members


def verify(pins_path: Path, artifacts: Path) -> None:
    plugins = load_pins(pins_path)
    checked = 0
    seen_targets = set()
    for path in sorted(artifacts.iterdir()):
        match = PRODUCT_ARCHIVE.match(path.name)
        if not match:
            if PRODUCT_ARCHIVE_NAME.match(path.name):
                raise BundleError(f"{path.name}: a product archive whose target is not recognized; it cannot be checked")
            continue
        target, kind = match.group("target"), match.group("kind")
        members = archive_members(path, kind)
        for plugin in plugins:
            if target in plugin.get("absent", {}):
                if members:
                    raise BundleError(f"{path.name}: carries plugins/ for {target}, which the pin lists as absent")
                continue
            asset = asset_name(plugin, target)
            found = [data for name, data in members.items() if name.endswith("/plugins/" + asset)]
            if len(found) != 1:
                raise BundleError(f"{path.name}: does not carry plugins/{asset}")
            actual = hashlib.sha256(found[0]).hexdigest()
            if actual != plugin["archives"][target]:
                raise BundleError(f"{path.name}: plugins/{asset} is {actual}, pinned {plugin['archives'][target]}")
            manifests = [data for name, data in members.items() if name.endswith("/plugins/manifest.json")]
            if len(manifests) != 1 or not any(
                e.get("archive") == asset and e.get("sha256") == actual for e in json.loads(manifests[0]).get("plugins", [])
            ):
                raise BundleError(f"{path.name}: plugins/manifest.json does not list {asset} at {actual}")
        seen_targets.add(target)
        checked += 1
        print(f"ok {path.name}")
    if checked == 0:
        raise BundleError(f"no release archives under {artifacts}")
    for plugin in plugins:
        unseen = sorted(t for t in plugin["archives"] if t not in seen_targets)
        if unseen:
            raise BundleError(f"no release archive for {', '.join(unseen)}, which {plugin['name']} is pinned for")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--out", type=Path, required=True)
    p_place = sub.add_parser("place")
    p_place.add_argument("--from", dest="source", type=Path, required=True)
    p_place.add_argument("--target", required=True)
    p_place.add_argument("--bundle", type=Path, required=True)
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("artifacts", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch(args.pins, args.out)
        elif args.command == "place":
            place(args.pins, args.source, args.target, args.bundle)
        else:
            verify(args.pins, args.artifacts)
    except BundleError as error:
        print(f"bundled-plugins: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
