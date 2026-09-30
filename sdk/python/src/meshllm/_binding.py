from __future__ import annotations

from importlib import import_module
from types import ModuleType

_native_module: ModuleType | object | None = None


def native() -> ModuleType | object:
    """Load the generated binding only when a native-backed API is used."""
    global _native_module
    if _native_module is None:
        _native_module = import_module("meshllm._generated.mesh_ffi")
    return _native_module


def _set_native_for_testing(module: object | None) -> None:
    global _native_module
    _native_module = module
