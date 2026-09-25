#!/usr/bin/env python3
"""Expose existing runner HF configuration to later canary steps; never provision it."""

import os
from pathlib import Path
import sys


def configuration(env):
    default = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    home = Path(env.get("HF_HOME") or env.get("HF_CACHE") or default).expanduser()
    hub = Path(env.get("HF_HUB_CACHE") or home / "hub").expanduser()
    # The certification planner consumes a cache root containing hub/. Do not
    # silently discard an explicit Hub override or point it at another cache.
    if hub.resolve() != (home / "hub").resolve():
        raise ValueError("HF_HUB_CACHE must resolve to HF_HOME/hub for family certification")
    if not hub.is_dir() or not os.access(hub, os.R_OK | os.X_OK):
        raise ValueError(f"configured Hugging Face hub cache is unavailable: {hub}; check the runner mount/configuration")
    values = {
        "HF_CACHE": str(home),
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(hub),
        # Offline certification is a job policy, not a provisioning prerequisite.
        "HF_HUB_OFFLINE": "1",
    }
    for name in ("HF_TOKEN", "HF_TOKEN_PATH"):
        if env.get(name):
            values[name] = env[name]
    for name, value in values.items():
        if any(char in value for char in "\r\n\0"):
            raise ValueError(f"invalid multiline value for {name}")
    return values


def main():
    try:
        values = configuration(os.environ)
        token = values.get("HF_TOKEN")
        if token:
            # Mask before handing the existing shell credential to later steps.
            print("::add-mask::" + token.replace("%", "%25"), flush=True)
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
            for name, value in values.items():
                output.write(f"{name}={value}\n")
        print(f"Using existing Hugging Face cache: {values['HF_HUB_CACHE']} (offline certification)")
    except (ValueError, OSError, KeyError) as error:
        print(f"canary cache preflight: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
