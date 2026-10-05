#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives with its owning product."""
from pathlib import Path as _Path

__file__ = str(_Path(__file__).resolve().parents[1] / 'skippy/scripts/generate-ocr-oracle-fixture.py')
exec(compile(_Path(__file__).read_bytes(), __file__, "exec"), globals())
