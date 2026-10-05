#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives with its owning product."""
from pathlib import Path as _Path

# The shared website entrypoint owns the Mesh destination; the Skippy generator
# requires an explicit output and has no dependency on the Mesh tree.
if __name__ == "__main__":
    import sys as _sys
    if not any(arg == "--output" or arg.startswith("--output=") for arg in _sys.argv[1:]):
        _sys.argv.extend(["--output", str(_Path(__file__).resolve().parents[1] / "mesh/website/src/docs/pages/skippy-api.md")])

__file__ = str(_Path(__file__).resolve().parents[1] / 'skippy/scripts/generate-skippy-api-doc.py')
exec(compile(_Path(__file__).read_bytes(), __file__, "exec"), globals())
